"""Step 03: clean values (missing data, dtypes, duplicates, invalid rows)."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pandas as pd

_DEFAULT_CONFIG: dict[str, Any] = {
    "drop_missing_required": True,
    "required_non_null_columns": (
        "match_id",
        "match_date",
        "winner_player_id",
        "team1_player_id",
        "team2_player_id",
    ),
    "coerce_datetime_columns": ("match_date",),
    "coerce_numeric_columns": ("team1_sgl_roll_rank", "team2_sgl_roll_rank"),
    "drop_duplicate_subset": ("match_id", "team1_player_id", "team2_player_id"),
    "drop_canonical_pair_duplicates": True,
    "canonical_pair_subset": ("match_id", "team1_player_id", "team2_player_id"),
    "canonical_pair_keep": "first",
    "invalid_filters": {
        "team1_not_team2": True,
        "winner_in_teams": True,
        "positive_rank": True,
    },
    "coerce_string_columns": (
        "match_id",
        "winner_player_id",
        "team1_player_id",
        "team2_player_id",
    )
}


def _normalize_config(config: Mapping[str, Any] | None) -> dict[str, Any]:
    normalized = dict(_DEFAULT_CONFIG)
    if config:
        normalized.update(dict(config))

    for key in (
        "required_non_null_columns",
        "coerce_datetime_columns",
        "coerce_numeric_columns",
        "coerce_string_columns",
        "drop_duplicate_subset",
        "canonical_pair_subset",
    ):
        val = normalized.get(key)
        if val is None:
            normalized[key] = ()
        elif not isinstance(val, (list, tuple, set)):
            raise TypeError(f"config['{key}'] must be a list/tuple/set")
        else:
            normalized[key] = tuple(val)

    if normalized.get("canonical_pair_keep") not in {"first", "last", False}:
        raise TypeError("config['canonical_pair_keep'] must be one of {'first', 'last', False}")

    invalid_filters = normalized.get("invalid_filters")
    if invalid_filters is None:
        normalized["invalid_filters"] = {}
    elif not isinstance(invalid_filters, Mapping):
        raise TypeError("config['invalid_filters'] must be a mapping")
    else:
        normalized["invalid_filters"] = dict(invalid_filters)

    return normalized


def run(df_or_path: pd.DataFrame, config: Mapping[str, Any] | None = None) -> pd.DataFrame:
    """Apply value-level cleaning to an already schema-normalized dataframe."""

    if not isinstance(df_or_path, pd.DataFrame):
        raise TypeError("03_clean_values.run expects a pandas DataFrame input")

    cfg = _normalize_config(config)
    df = df_or_path.copy(deep=True)

    for col in cfg["coerce_datetime_columns"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce", utc=False)

    for col in cfg["coerce_numeric_columns"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    for col in cfg["coerce_string_columns"]:
        if col in df.columns:
            df[col] = df[col].astype("string")
            
    if cfg.get("drop_missing_required", True):
        required_present = [c for c in cfg["required_non_null_columns"] if c in df.columns]
        if required_present:
            df = df.dropna(subset=required_present).copy(deep=True)

    dup_subset = [c for c in cfg["drop_duplicate_subset"] if c in df.columns]
    if dup_subset:
        df = df.drop_duplicates(subset=dup_subset, keep="first").copy(deep=True)

    canonical_subset = [c for c in cfg["canonical_pair_subset"] if c in df.columns]
    can_dedupe_enabled = cfg.get("drop_canonical_pair_duplicates", True)
    if can_dedupe_enabled and len(canonical_subset) == 3:
        match_col, team1_col, team2_col = canonical_subset
        pair_frame = df[[team1_col, team2_col]].astype("string")
        df["_p_low"] = pair_frame.min(axis=1)
        df["_p_high"] = pair_frame.max(axis=1)

        before = len(df)
        df = df.drop_duplicates(
            subset=[match_col, "_p_low", "_p_high"],
            keep=cfg.get("canonical_pair_keep", "first"),
        ).copy(deep=True)
        dropped = before - len(df)
        df.attrs["canonical_pair_duplicates_dropped"] = dropped

        df = df.drop(columns=["_p_low", "_p_high"])

    filters = cfg.get("invalid_filters", {})

    if filters.get("team1_not_team2", True):
        needed = {"team1_player_id", "team2_player_id"}
        if needed.issubset(df.columns):
            df = df[df["team1_player_id"] != df["team2_player_id"]].copy(deep=True)

    if filters.get("winner_in_teams", True):
        needed = {"winner_player_id", "team1_player_id", "team2_player_id"}
        if needed.issubset(df.columns):
            winner_in_team = (df["winner_player_id"] == df["team1_player_id"]) | (
                df["winner_player_id"] == df["team2_player_id"]
            )
            df = df[winner_in_team].copy(deep=True)

    if filters.get("positive_rank", True):
        for rank_col in ("team1_sgl_roll_rank", "team2_sgl_roll_rank"):
            if rank_col in df.columns:
                df = df[df[rank_col].isna() | (df[rank_col] > 0)].copy(deep=True)

    return df
