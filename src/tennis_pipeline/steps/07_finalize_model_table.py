"""Step 07: select leakage-safe final columns and standardize model table order."""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any

import pandas as pd

_DEFAULT_CONFIG: dict[str, Any] = {
    "target_column": "team1_wins",
    "id_columns": ["match_id", "match_date", "match_seq", "team1_player_id", "team2_player_id"],
    "preferred_feature_order": [
        "rank_diff",
        "abs_rank_diff",
        "race_rank_diff",
        "abs_race_rank_diff",
        "elo_team1_pre",
        "elo_team2_pre",
        "elo_diff_team1",
        "elo_prob_team1_pre",
        "surface_context",
        "court_context",
    ],
    "feature_prefixes": ["diff_", "abs_diff_", "same_", "elo_"],
    "drop_columns": [
        "winner_player_id",
        "loser_player_id",
        "winner_id",
        "loser_id",
        "winning_player_id",
        "result",
        "elo_diff_pre",
    ],
}

# Post-match or target-adjacent columns that can create leakage in training.
_LEAKAGE_PATTERN = re.compile(r"(winner|loser|result|score|outcome|post_|_post|after)", flags=re.IGNORECASE)


def _normalize_config(config: Mapping[str, Any] | None) -> dict[str, Any]:
    normalized = dict(_DEFAULT_CONFIG)
    if config:
        normalized.update(dict(config))

    for key in ("target_column",):
        if not isinstance(normalized.get(key), str) or not normalized[key].strip():
            raise TypeError(f"config['{key}'] must be a non-empty string")

    for key in ("id_columns", "preferred_feature_order", "feature_prefixes", "drop_columns"):
        value = normalized.get(key)
        if not isinstance(value, list) or not all(isinstance(v, str) and v.strip() for v in value):
            raise TypeError(f"config['{key}'] must be a list[str]")

    return normalized


def _ensure_temporal_sequence(df: pd.DataFrame) -> pd.DataFrame:
    """Create a stable numeric match sequence if absent."""

    if "match_seq" in df.columns:
        return df

    out = df.copy(deep=True)
    out["_row_order"] = range(len(out))

    sort_cols = [c for c in ("match_date", "match_id", "_row_order") if c in out.columns]
    if sort_cols:
        sorted_idx = out.sort_values(sort_cols, kind="mergesort").index
        seq = pd.Series(range(1, len(out) + 1), index=sorted_idx)
        out["match_seq"] = seq.sort_index().astype("int64")
    else:
        out["match_seq"] = pd.Series(range(1, len(out) + 1), index=out.index, dtype="int64")

    return out.drop(columns=["_row_order"]) if "_row_order" in out.columns else out


def _resolve_elo_aliases(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy(deep=True)
    if "elo_diff_team1" not in out.columns and "elo_diff_pre" in out.columns:
        out["elo_diff_team1"] = out["elo_diff_pre"]
    return out


def _is_leakage_column(col: str, target_column: str) -> bool:
    if col == target_column:
        return False
    return bool(_LEAKAGE_PATTERN.search(col))


def _select_feature_columns(df: pd.DataFrame, cfg: dict[str, Any]) -> list[str]:
    target_column = cfg["target_column"]
    id_columns = set(cfg["id_columns"])
    preferred = cfg["preferred_feature_order"]
    feature_prefixes = tuple(cfg["feature_prefixes"])

    candidates = [
        c
        for c in df.columns
        if c not in id_columns
        and c != target_column
        and not _is_leakage_column(c, target_column)
        and c not in set(cfg["drop_columns"])
    ]

    patterned = [c for c in candidates if c.startswith(feature_prefixes) or c in preferred]

    preferred_present = [c for c in preferred if c in patterned]
    remaining = sorted([c for c in patterned if c not in set(preferred_present)])
    return [*preferred_present, *remaining]


def run(df_or_path: pd.DataFrame, config: Mapping[str, Any] | None = None) -> pd.DataFrame:
    """Return final model-ready dataframe with ordered features and target last."""

    if not isinstance(df_or_path, pd.DataFrame):
        raise TypeError("07_finalize_model_table.run expects a pandas DataFrame input")

    cfg = _normalize_config(config)
    target_column = cfg["target_column"]

    if target_column not in df_or_path.columns:
        raise ValueError(f"[finalize_model_table] Missing target column: {target_column}")

    out = _resolve_elo_aliases(df_or_path)
    out = _ensure_temporal_sequence(out)

    id_columns = [c for c in cfg["id_columns"] if c in out.columns]
    feature_columns = _select_feature_columns(out, cfg)

    final_columns = [*id_columns, *feature_columns, target_column]
    final_columns = [c for c in final_columns if c in out.columns]

    finalized = out.loc[:, final_columns].copy(deep=True)
    finalized[target_column] = pd.to_numeric(finalized[target_column], errors="coerce").astype("Int64")

    return finalized
