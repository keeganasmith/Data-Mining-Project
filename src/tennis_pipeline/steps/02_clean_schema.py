"""Step 02: rename/standardize columns and enforce required schema."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pandas as pd

from tennis_pipeline.data_contracts import RAW_INPUT_CONTRACT, normalize_columns

_DEFAULT_CONFIG: dict[str, Any] = {
    "column_aliases": {},
    "required_columns": None,
    "enforce_required_schema": True,
}


def _normalize_config(config: Mapping[str, Any] | None) -> dict[str, Any]:
    normalized = dict(_DEFAULT_CONFIG)
    if config:
        normalized.update(dict(config))

    aliases = normalized.get("column_aliases")
    if aliases is None:
        normalized["column_aliases"] = {}
    elif not isinstance(aliases, Mapping):
        raise TypeError("config['column_aliases'] must be a mapping")
    else:
        normalized["column_aliases"] = dict(aliases)

    required_columns = normalized.get("required_columns")
    if required_columns is not None and not isinstance(required_columns, (list, tuple, set)):
        raise TypeError("config['required_columns'] must be a list/tuple/set when provided")

    return normalized


def _dedupe_columns_keep_last(df: pd.DataFrame) -> pd.DataFrame:
    if not df.columns.duplicated().any():
        return df
    return df.loc[:, ~df.columns.duplicated(keep="last")].copy(deep=True)


def _default_required_columns() -> tuple[str, ...]:
    required: list[str] = []
    for src_col in RAW_INPUT_CONTRACT.required_columns:
        required.append(RAW_INPUT_CONTRACT.canonical_column_names.get(src_col, src_col))
    return tuple(dict.fromkeys(required))


def run(df_or_path: pd.DataFrame, config: Mapping[str, Any] | None = None) -> pd.DataFrame:
    """Standardize column names and enforce required schema.

    This step is intentionally pure and only accepts a dataframe input.
    """

    if not isinstance(df_or_path, pd.DataFrame):
        raise TypeError("02_clean_schema.run expects a pandas DataFrame input")

    cfg = _normalize_config(config)
    df = df_or_path.copy(deep=True)

    df = normalize_columns(df, RAW_INPUT_CONTRACT)

    custom_aliases: dict[str, str] = cfg.get("column_aliases", {})
    if custom_aliases:
        applicable_aliases = {src: dst for src, dst in custom_aliases.items() if src in df.columns and src != dst}
        if applicable_aliases:
            df = df.rename(columns=applicable_aliases)

    df = _dedupe_columns_keep_last(df)

    if cfg.get("enforce_required_schema", True):
        if cfg.get("required_columns") is None:
            required = _default_required_columns()
        else:
            required = tuple(cfg["required_columns"])

        missing = [col for col in required if col not in df.columns]
        if missing:
            raise ValueError(f"[clean_schema] Missing required columns after normalization: {missing}")

    return df
