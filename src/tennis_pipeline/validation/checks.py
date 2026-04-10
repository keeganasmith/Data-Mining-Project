"""Pipeline validation checks with fail-fast, human-readable errors."""

from __future__ import annotations

import re
from dataclasses import dataclass

import pandas as pd
from pandas.api.types import is_datetime64_any_dtype, is_numeric_dtype, is_string_dtype

from tennis_pipeline.data_contracts import (
    CLEANED_INTERIM_CONTRACT,
    FINAL_MODEL_CONTRACT,
    RAW_INPUT_CONTRACT,
    TableContract,
    validate_dtypes,
    validate_key_constraints,
    validate_required_columns,
)

_NULL_RATE_THRESHOLDS: dict[str, float] = {
    "EventId": 0.0,
    "StartDate": 0.0,
    "WinningPlayerId": 1.0,
    "PlayerTeam1.PlayerId": 0.0,
    "PlayerTeam2.PlayerId": 0.0,
    "PlayerTeam1.SglRollRank": 0.10,
    "PlayerTeam2.SglRollRank": 0.10,
    "match_id": 0.0,
    "match_date": 0.0,
    "team1_player_id": 0.0,
    "team2_player_id": 0.0,
    "winner_player_id": 0.02,
    "team1_sgl_roll_rank": 0.10,
    "team2_sgl_roll_rank": 0.10,
    "rank_diff": 0.05,
    "abs_rank_diff": 0.05,
    "elo_diff_team1": 0.02,
    "surface_context": 0.10,
    "court_context": 0.10,
    "team1_wins": 0.0,
}

_LEAKAGE_PATTERN = re.compile(r"(winner|loser|result|score|outcome|post_|_post|after)", re.IGNORECASE)
_ALLOWED_FINAL_COLUMNS = {
    "team1_wins",
    "elo_prob_team1_pre",
}


@dataclass(frozen=True)
class StageValidationSpec:
    """Validation policy for one pipeline stage."""

    contract: TableContract | None = None
    required_columns: tuple[str, ...] = ()
    dtype_expectations: dict[str, str] | None = None
    null_rate_columns: tuple[str, ...] = ()
    duplicate_key_sets: tuple[tuple[str, ...], ...] = ()
    check_temporal_monotonicity: bool = False
    check_leakage_guard: bool = False


def _format_missing(columns: list[str]) -> str:
    return ", ".join(columns)


def check_required_columns(df: pd.DataFrame, required_columns: tuple[str, ...], *, step_name: str) -> None:
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise ValueError(f"[{step_name}] Missing required columns: {_format_missing(missing)}")


def check_dtype_expectations(df: pd.DataFrame, expectations: dict[str, str], *, step_name: str) -> None:
    violations: list[str] = []
    for col, expected in expectations.items():
        if col not in df.columns:
            continue
        series = df[col]
        ok = False
        if expected == "datetime":
            ok = is_datetime64_any_dtype(series)
        elif expected == "numeric":
            ok = is_numeric_dtype(series)
        elif expected == "string":
            ok = is_string_dtype(series) or pd.api.types.is_object_dtype(series)
        if not ok:
            violations.append(f"{col}: expected {expected}, got {series.dtype}")

    if violations:
        raise TypeError(f"[{step_name}] Dtype compliance check failed: {'; '.join(violations)}")


def check_null_rate_thresholds(
    df: pd.DataFrame,
    columns: tuple[str, ...],
    *,
    step_name: str,
    thresholds: dict[str, float] | None = None,
) -> None:
    limits = thresholds or _NULL_RATE_THRESHOLDS
    violations: list[str] = []

    for col in columns:
        if col not in df.columns:
            continue
        threshold = limits.get(col)
        if threshold is None:
            continue
        null_rate = float(df[col].isna().mean()) if len(df) else 0.0
        if null_rate > threshold:
            violations.append(f"{col} null_rate={null_rate:.3f} > threshold={threshold:.3f}")

    if violations:
        raise ValueError(f"[{step_name}] Null-rate threshold violations: {'; '.join(violations)}")


def check_duplicate_key_constraints(df: pd.DataFrame, key_sets: tuple[tuple[str, ...], ...], *, step_name: str) -> None:
    for key_cols in key_sets:
        key_list = list(key_cols)
        missing = [col for col in key_list if col not in df.columns]
        if missing:
            raise ValueError(
                f"[{step_name}] Duplicate-key check could not run; missing key columns: {_format_missing(missing)}"
            )

        duplicate_mask = df.duplicated(subset=key_list, keep=False)
        if duplicate_mask.any():
            dup_count = int(duplicate_mask.sum())
            sample = df.loc[duplicate_mask, key_list].head(3).to_dict("records")
            raise ValueError(
                f"[{step_name}] Duplicate key constraint violated for {tuple(key_list)}; "
                f"rows={dup_count}; sample={sample}"
            )


def check_temporal_monotonicity_for_elo(df: pd.DataFrame, *, step_name: str) -> None:
    required = ("match_date", "team1_player_id", "team2_player_id")
    check_required_columns(df, required, step_name=step_name)

    if not is_datetime64_any_dtype(df["match_date"]):
        raise TypeError(f"[{step_name}] Temporal monotonicity check requires datetime 'match_date'")

    if len(df) <= 1:
        return

    sort_cols = ["match_date"]
    if "match_seq" in df.columns:
        sort_cols.append("match_seq")
    elif "match_id" in df.columns:
        sort_cols.append("match_id")

    sorted_df = df.sort_values(sort_cols, kind="mergesort")
    if not sorted_df.index.equals(df.index):
        raise ValueError(
            f"[{step_name}] Temporal monotonicity violated; expected non-decreasing order by {sort_cols}"
        )


def check_leakage_guard_before_output(df: pd.DataFrame, *, step_name: str) -> None:
    leaking_columns = [
        col for col in df.columns if col not in _ALLOWED_FINAL_COLUMNS and bool(_LEAKAGE_PATTERN.search(col))
    ]
    if leaking_columns:
        leaking_columns = sorted(leaking_columns)
        raise ValueError(
            f"[{step_name}] Leakage guard failed; final output still contains leakage-prone columns: "
            f"{', '.join(leaking_columns)}"
        )


def validate_contract(df: pd.DataFrame, contract: TableContract, *, step_name: str) -> None:
    try:
        validate_required_columns(df, contract)
        validate_dtypes(df, contract)
        validate_key_constraints(df, contract)
    except (TypeError, ValueError) as exc:
        raise type(exc)(f"[{step_name}] Contract validation failed: {exc}") from exc


def run_stage_checks(df: pd.DataFrame, step_name: str) -> None:
    """Fail-fast stage validations with explicit error messages."""

    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"[{step_name}] Step output must be a pandas DataFrame")
    if df.columns.duplicated().any():
        dupes = df.columns[df.columns.duplicated()].tolist()
        raise ValueError(f"[{step_name}] Output has duplicate columns: {dupes}")

    specs: dict[str, StageValidationSpec] = {
        "01_load_raw": StageValidationSpec(
            required_columns=RAW_INPUT_CONTRACT.required_columns,
            null_rate_columns=(
                "EventId",
                "StartDate",
                "PlayerTeam1.PlayerId",
                "PlayerTeam2.PlayerId",
                "WinningPlayerId",
                "PlayerTeam1.SglRollRank",
                "PlayerTeam2.SglRollRank",
            ),
        ),
        "02_clean_schema": StageValidationSpec(
            required_columns=(
                "match_id",
                "match_date",
                "winner_player_id",
                "team1_player_id",
                "team2_player_id",
                "team1_sgl_roll_rank",
                "team2_sgl_roll_rank",
            ),
            null_rate_columns=("match_id", "match_date", "team1_player_id", "team2_player_id"),
        ),
        "03_clean_values": StageValidationSpec(
            required_columns=(
                "match_id",
                "match_date",
                "winner_player_id",
                "team1_player_id",
                "team2_player_id",
                "team1_sgl_roll_rank",
                "team2_sgl_roll_rank",
            ),
            dtype_expectations={
                "match_date": "datetime",
                "team1_sgl_roll_rank": "numeric",
                "team2_sgl_roll_rank": "numeric",
            },
            null_rate_columns=(
                "match_id",
                "match_date",
                "winner_player_id",
                "team1_player_id",
                "team2_player_id",
                "team1_sgl_roll_rank",
                "team2_sgl_roll_rank",
            ),
            duplicate_key_sets=(("match_id", "team1_player_id", "team2_player_id"),),
        ),
        "04_split_roles": StageValidationSpec(
            required_columns=(
                "match_id",
                "match_date",
                "winner_player_id",
                "team1_player_id",
                "team2_player_id",
                "team1_sgl_roll_rank",
                "team2_sgl_roll_rank",
                "team1_wins",
            ),
            dtype_expectations={
                "match_date": "datetime",
                "team1_sgl_roll_rank": "numeric",
                "team2_sgl_roll_rank": "numeric",
                "team1_wins": "numeric",
            },
            null_rate_columns=(
                "match_id",
                "match_date",
                "team1_player_id",
                "team2_player_id",
                "team1_wins",
            ),
            duplicate_key_sets=(("match_id", "team1_player_id", "team2_player_id"),),
        ),
        "05_build_features_static": StageValidationSpec(
            contract=CLEANED_INTERIM_CONTRACT,
            null_rate_columns=(
                "rank_diff",
                "abs_rank_diff",
                "surface_context",
                "court_context",
            ),
            duplicate_key_sets=(("match_id", "team1_player_id", "team2_player_id"),),
        ),
        "06_build_features_temporal_elo": StageValidationSpec(
            contract=CLEANED_INTERIM_CONTRACT,
            null_rate_columns=("elo_diff_team1",),
            duplicate_key_sets=(("match_id", "team1_player_id", "team2_player_id"),),
            check_temporal_monotonicity=True,
        ),
        "07_finalize_model_table": StageValidationSpec(
            contract=FINAL_MODEL_CONTRACT,
            null_rate_columns=(
                "match_id",
                "match_date",
                "match_seq",
                "team1_player_id",
                "team2_player_id",
                "rank_diff",
                "abs_rank_diff",
                "elo_diff_team1",
                "team1_wins",
            ),
            duplicate_key_sets=(("match_id", "match_seq"),),
            check_temporal_monotonicity=True,
            check_leakage_guard=True,
        ),
    }

    spec = specs.get(step_name)
    if spec is None:
        raise ValueError(f"[{step_name}] No validation policy defined for this step")

    if spec.contract is not None:
        validate_contract(df, spec.contract, step_name=step_name)
    if spec.required_columns:
        check_required_columns(df, spec.required_columns, step_name=step_name)
    if spec.dtype_expectations:
        check_dtype_expectations(df, spec.dtype_expectations, step_name=step_name)

    if spec.null_rate_columns:
        check_null_rate_thresholds(df, spec.null_rate_columns, step_name=step_name)
    if spec.duplicate_key_sets:
        check_duplicate_key_constraints(df, spec.duplicate_key_sets, step_name=step_name)

    if spec.check_temporal_monotonicity:
        check_temporal_monotonicity_for_elo(df, step_name=step_name)
    if spec.check_leakage_guard:
        check_leakage_guard_before_output(df, step_name=step_name)
