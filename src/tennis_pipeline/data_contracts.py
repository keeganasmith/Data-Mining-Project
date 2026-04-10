"""Reusable schema contracts and validators for pipeline tables.

This module centralizes column contracts for three table stages:
- raw input table
- cleaned/interim table
- final model-ready table

Step modules and ``validation/checks.py`` can import these definitions to
validate required fields, dtypes, key constraints, and canonical naming.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd
from pandas.api.types import is_bool_dtype, is_datetime64_any_dtype, is_numeric_dtype, is_string_dtype


# Canonical columns used across contracts.
TARGET_COLUMN = "team1_wins"
TEMPORAL_ORDER_COLUMNS: tuple[str, ...] = ("match_date", "match_seq")
SURFACE_COLUMNS: tuple[str, ...] = ("surface_context", "court_context")


@dataclass(frozen=True)
class KeyConstraint:
    """Represents key requirements for a table.

    Attributes:
        required_columns: columns that must be present and non-null.
        unique_together: column sets that must be unique if present.
    """

    required_columns: tuple[str, ...] = ()
    unique_together: tuple[tuple[str, ...], ...] = ()


@dataclass(frozen=True)
class TableContract:
    """Contract for a dataframe schema at a pipeline stage."""

    name: str
    required_columns: tuple[str, ...]
    canonical_column_names: dict[str, str]
    dtypes: dict[str, str]
    keys: KeyConstraint


RAW_INPUT_CONTRACT = TableContract(
    name="raw_input",
    required_columns=(
        "EventId",
        "StartDate",
        "WinningPlayerId",
        "PlayerTeam1.PlayerId",
        "PlayerTeam2.PlayerId",
        "PlayerTeam1.SglRollRank",
        "PlayerTeam2.SglRollRank",
    ),
    canonical_column_names={
        "EventId": "match_id",
        "StartDate": "match_date",
        "WinningPlayerId": "winner_player_id",
        "PlayerTeam1.PlayerId": "team1_player_id",
        "PlayerTeam2.PlayerId": "team2_player_id",
        "PlayerTeam1.SglRollRank": "team1_sgl_roll_rank",
        "PlayerTeam2.SglRollRank": "team2_sgl_roll_rank",
        # Surface/court source aliases from notebook logic.
        "CourtSurface": "surface_context",
        "Court.Surface": "surface_context",
        "Court": "court_context",
    },
    dtypes={
        "EventId": "string",
        "StartDate": "datetime",
        "WinningPlayerId": "string",
        "PlayerTeam1.PlayerId": "string",
        "PlayerTeam2.PlayerId": "string",
        "PlayerTeam1.SglRollRank": "numeric",
        "PlayerTeam2.SglRollRank": "numeric",
    },
    keys=KeyConstraint(
        required_columns=(
            "EventId",
            "StartDate",
            "PlayerTeam1.PlayerId",
            "PlayerTeam2.PlayerId",
        ),
        unique_together=(("EventId", "PlayerTeam1.PlayerId", "PlayerTeam2.PlayerId"),),
    ),
)


CLEANED_INTERIM_CONTRACT = TableContract(
    name="cleaned_interim",
    required_columns=(
        "match_id",
        "match_date",
        "winner_player_id",
        "team1_player_id",
        "team2_player_id",
        "team1_sgl_roll_rank",
        "team2_sgl_roll_rank",
        TARGET_COLUMN,
        "surface_context",
        "court_context",
    ),
    canonical_column_names={
        "match_id": "match_id",
        "match_date": "match_date",
        "winner_player_id": "winner_player_id",
        "team1_player_id": "team1_player_id",
        "team2_player_id": "team2_player_id",
        "team1_sgl_roll_rank": "team1_sgl_roll_rank",
        "team2_sgl_roll_rank": "team2_sgl_roll_rank",
        "surface_context": "surface_context",
        "court_context": "court_context",
        TARGET_COLUMN: TARGET_COLUMN,
    },
    dtypes={
        "match_id": "string",
        "match_date": "datetime",
        "winner_player_id": "string",
        "team1_player_id": "string",
        "team2_player_id": "string",
        "team1_sgl_roll_rank": "numeric",
        "team2_sgl_roll_rank": "numeric",
        "surface_context": "string",
        "court_context": "string",
        TARGET_COLUMN: "bool_or_int",
    },
    keys=KeyConstraint(
        required_columns=("match_id", "match_date", "team1_player_id", "team2_player_id"),
        unique_together=(("match_id", "team1_player_id", "team2_player_id"),),
    ),
)


FINAL_MODEL_CONTRACT = TableContract(
    name="final_model_ready",
    required_columns=(
        "match_id",
        "match_date",
        "match_seq",
        "team1_player_id",
        "team2_player_id",
        "rank_diff",
        "abs_rank_diff",
        "elo_diff_team1",
        "surface_context",
        "court_context",
        TARGET_COLUMN,
    ),
    canonical_column_names={
        "match_id": "match_id",
        "match_date": "match_date",
        "match_seq": "match_seq",
        "team1_player_id": "team1_player_id",
        "team2_player_id": "team2_player_id",
        "rank_diff": "rank_diff",
        "abs_rank_diff": "abs_rank_diff",
        "elo_diff_team1": "elo_diff_team1",
        "surface_context": "surface_context",
        "court_context": "court_context",
        TARGET_COLUMN: TARGET_COLUMN,
    },
    dtypes={
        "match_id": "string",
        "match_date": "datetime",
        "match_seq": "numeric",
        "team1_player_id": "string",
        "team2_player_id": "string",
        "rank_diff": "numeric",
        "abs_rank_diff": "numeric",
        "elo_diff_team1": "numeric",
        "surface_context": "string",
        "court_context": "string",
        TARGET_COLUMN: "bool_or_int",
    },
    keys=KeyConstraint(
        required_columns=("match_id", "match_date", "match_seq", "team1_player_id", "team2_player_id"),
        unique_together=(("match_id", "match_seq"),),
    ),
)


CONTRACTS: dict[str, TableContract] = {
    RAW_INPUT_CONTRACT.name: RAW_INPUT_CONTRACT,
    CLEANED_INTERIM_CONTRACT.name: CLEANED_INTERIM_CONTRACT,
    FINAL_MODEL_CONTRACT.name: FINAL_MODEL_CONTRACT,
}


def get_contract(name: str) -> TableContract:
    """Return a contract by name and raise a helpful error if unknown."""
    try:
        return CONTRACTS[name]
    except KeyError as exc:
        available = ", ".join(sorted(CONTRACTS))
        raise KeyError(f"Unknown contract '{name}'. Available: {available}") from exc


def normalize_columns(df: pd.DataFrame, contract: TableContract) -> pd.DataFrame:
    """Rename source columns into contract canonical names where aliases exist."""
    applicable_map = {src: dst for src, dst in contract.canonical_column_names.items() if src in df.columns and src != dst}
    if not applicable_map:
        return df
    return df.rename(columns=applicable_map)


def validate_required_columns(df: pd.DataFrame, contract: TableContract) -> None:
    """Validate that all required columns defined by the contract are present."""
    missing = [col for col in contract.required_columns if col not in df.columns]
    if missing:
        raise ValueError(f"[{contract.name}] Missing required columns: {missing}")


def validate_dtypes(df: pd.DataFrame, contract: TableContract) -> None:
    """Validate dtypes based on semantic dtype groups in a contract."""
    problems: list[str] = []
    for col, expected in contract.dtypes.items():
        if col not in df.columns:
            continue

        series = df[col]
        ok = True
        if expected == "numeric":
            ok = is_numeric_dtype(series)
        elif expected == "string":
            ok = is_string_dtype(series) or pd.api.types.is_object_dtype(series)
        elif expected == "datetime":
            ok = is_datetime64_any_dtype(series)
        elif expected == "bool_or_int":
            ok = is_bool_dtype(series) or is_numeric_dtype(series)
        else:
            problems.append(f"{col}: unsupported expected dtype '{expected}'")
            continue

        if not ok:
            problems.append(f"{col}: expected {expected}, got {series.dtype}")

    if problems:
        details = "; ".join(problems)
        raise TypeError(f"[{contract.name}] Dtype violations: {details}")


def validate_key_constraints(df: pd.DataFrame, contract: TableContract) -> None:
    """Validate non-null key fields and uniqueness constraints."""
    missing_in_keys = [col for col in contract.keys.required_columns if col not in df.columns]
    if missing_in_keys:
        raise ValueError(f"[{contract.name}] Missing key columns: {missing_in_keys}")

    null_key_cols = [col for col in contract.keys.required_columns if df[col].isna().any()]
    if null_key_cols:
        raise ValueError(f"[{contract.name}] Null values found in key columns: {null_key_cols}")

    for key_cols in contract.keys.unique_together:
        if not all(col in df.columns for col in key_cols):
            continue
        duplicate_mask = df.duplicated(subset=list(key_cols), keep=False)
        if duplicate_mask.any():
            dup_count = int(duplicate_mask.sum())
            raise ValueError(
                f"[{contract.name}] Duplicate key rows for {key_cols}; violating rows={dup_count}"
            )


def validate_temporal_ordering(df: pd.DataFrame, date_col: str = "match_date", seq_col: str = "match_seq") -> None:
    """Validate temporal ordering columns needed for leakage-safe Elo updates."""
    missing = [c for c in (date_col, seq_col) if c not in df.columns]
    if missing:
        raise ValueError(f"Missing temporal ordering columns: {missing}")

    if not is_datetime64_any_dtype(df[date_col]):
        raise TypeError(f"Temporal column '{date_col}' must be datetime-like")
    if not is_numeric_dtype(df[seq_col]):
        raise TypeError(f"Temporal sequence column '{seq_col}' must be numeric")

    ordered = df.sort_values([date_col, seq_col], kind="mergesort")
    if not ordered.index.equals(df.index):
        raise ValueError(f"DataFrame is not sorted by [{date_col}, {seq_col}] required for Elo")


def validate_table(df: pd.DataFrame, contract: TableContract, *, check_temporal: bool = False) -> None:
    """Run full contract checks for a dataframe."""
    validate_required_columns(df, contract)
    validate_dtypes(df, contract)
    validate_key_constraints(df, contract)
    if check_temporal:
        validate_temporal_ordering(df)


def contract_summary(contract: TableContract) -> dict[str, Any]:
    """Return a lightweight serializable summary for logging/reporting."""
    return {
        "name": contract.name,
        "required_columns": list(contract.required_columns),
        "target_column": TARGET_COLUMN,
        "temporal_order_columns": list(TEMPORAL_ORDER_COLUMNS),
        "surface_columns": list(SURFACE_COLUMNS),
        "key_required_columns": list(contract.keys.required_columns),
        "unique_together": [list(cols) for cols in contract.keys.unique_together],
    }
