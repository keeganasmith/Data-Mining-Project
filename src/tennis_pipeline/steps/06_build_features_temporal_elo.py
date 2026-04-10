"""Step 06: build leakage-safe temporal Elo features.

Generated column naming
-----------------------
Given ``feature_prefix`` (default: ``"elo"``), this step appends:

- ``{prefix}_team1_pre``: Team1 player's Elo immediately before the match.
- ``{prefix}_team2_pre``: Team2 player's Elo immediately before the match.
- ``{prefix}_diff_pre``: pre-match Elo differential (Team1 - Team2).
- ``{prefix}_prob_team1_pre``: Team1 expected win probability from pre-match Elo.

Temporal safeguard
------------------
Rows are processed in strict chronological order (``match_date`` then
``match_seq`` when available else ``match_id`` then original row position for
deterministic ties). Elo features for a row are computed *before* applying that
row's outcome update.
Therefore each row can only depend on prior rows in this temporal order,
preventing look-ahead leakage.

Canonical ordering invariant
----------------------------
This stage is the canonical ordering point for downstream steps. From this
point onward, output rows are globally ordered by temporal keys:
``match_date`` then (``match_seq`` if present, otherwise ``match_id``), with a
stable row-position tie-breaker.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pandas as pd

from tennis_pipeline.config import ELO_DEFAULTS


def _normalize_config(config: Mapping[str, Any] | None) -> dict[str, Any]:
    normalized = dict(ELO_DEFAULTS)
    if config:
        normalized.update(dict(config))

    for key in ("initial_rating", "k_factor", "rating_scale"):
        value = normalized.get(key)
        if not isinstance(value, (int, float)):
            raise TypeError(f"config['{key}'] must be numeric")

    if normalized["rating_scale"] <= 0:
        raise ValueError("config['rating_scale'] must be > 0")
    if normalized["k_factor"] < 0:
        raise ValueError("config['k_factor'] must be >= 0")

    prefix = normalized.get("feature_prefix")
    if not isinstance(prefix, str) or not prefix.strip():
        raise TypeError("config['feature_prefix'] must be a non-empty string")
    normalized["feature_prefix"] = prefix.strip()

    strict_validation = normalized.get("strict_validation")
    if not isinstance(strict_validation, bool):
        raise TypeError("config['strict_validation'] must be a bool")

    return normalized


def _resolve_result(df: pd.DataFrame) -> pd.Series:
    """Return Team1 score in {0.0, 1.0} for each row."""

    if "team1_wins" in df.columns:
        team1_wins = pd.to_numeric(df["team1_wins"], errors="coerce")
        valid = team1_wins.isin([0, 1])
        if valid.all():
            return team1_wins.astype(float)

    required = {"winner_player_id", "team1_player_id", "team2_player_id"}
    if required.issubset(df.columns):
        winner_is_team1 = (df["winner_player_id"] == df["team1_player_id"]).astype(float)
        winner_is_team2 = (df["winner_player_id"] == df["team2_player_id"]).astype(float)
        valid = (winner_is_team1 + winner_is_team2) == 1.0
        if valid.all():
            return winner_is_team1

    raise ValueError(
        "Unable to infer row outcomes. Provide a valid binary 'team1_wins' "
        "column or winner/team id columns."
    )


def _validate_required_columns(df: pd.DataFrame) -> None:
    required = ["match_date", "team1_player_id", "team2_player_id"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"[build_features_temporal_elo] Missing required columns: {missing}")


def _temporal_sort_columns(df: pd.DataFrame, *, stable_tie_breaker: str) -> list[str]:
    """Resolve canonical temporal sort keys used by validation and downstream stages."""

    sort_cols = ["match_date"]
    if "match_seq" in df.columns:
        sort_cols.append("match_seq")
    elif "match_id" in df.columns:
        sort_cols.append("match_id")
    sort_cols.append(stable_tie_breaker)
    return sort_cols


def run(df_or_path: pd.DataFrame, config: Mapping[str, Any] | None = None) -> pd.DataFrame:
    """Add strict temporal Elo features and merge them back onto input rows."""

    if not isinstance(df_or_path, pd.DataFrame):
        raise TypeError("06_build_features_temporal_elo.run expects a pandas DataFrame input")

    cfg = _normalize_config(config)
    out = df_or_path.copy(deep=True)
    out["row_id"] = range(len(out))

    if cfg["strict_validation"]:
        _validate_required_columns(out)

    time_col = "match_date"
    team1_col = "team1_player_id"
    team2_col = "team2_player_id"

    working = out.copy(deep=True)
    working[time_col] = pd.to_datetime(working[time_col], errors="coerce", utc=False)
    if cfg["strict_validation"] and working[time_col].isna().any():
        bad_count = int(working[time_col].isna().sum())
        raise ValueError(f"[build_features_temporal_elo] Found {bad_count} rows with invalid match_date")

    sort_cols = _temporal_sort_columns(working, stable_tie_breaker="row_id")

    working = working.sort_values(sort_cols, ascending=True, kind="mergesort")

    team1_score = _resolve_result(working)
    team1_score.index = working["row_id"].to_numpy()

    ratings: dict[Any, float] = {}
    initial = float(cfg["initial_rating"])
    k_factor = float(cfg["k_factor"])
    scale = float(cfg["rating_scale"])

    team1_pre_vals: list[float] = []
    team2_pre_vals: list[float] = []
    diff_pre_vals: list[float] = []
    prob_pre_vals: list[float] = []

    for row in working.itertuples(index=False):
        idx = getattr(row, "row_id")
        p1 = getattr(row, team1_col)
        p2 = getattr(row, team2_col)

        if pd.isna(p1) or pd.isna(p2):
            raise ValueError("Encountered null team player id while computing Elo features")

        r1_pre = ratings.get(p1, initial)
        r2_pre = ratings.get(p2, initial)

        expected_1 = 1.0 / (1.0 + 10 ** ((r2_pre - r1_pre) / scale))
        s1 = float(team1_score.loc[idx])

        r1_post = r1_pre + k_factor * (s1 - expected_1)
        r2_post = r2_pre + k_factor * ((1.0 - s1) - (1.0 - expected_1))

        ratings[p1] = r1_post
        ratings[p2] = r2_post

        team1_pre_vals.append(r1_pre)
        team2_pre_vals.append(r2_pre)
        diff_pre_vals.append(r1_pre - r2_pre)
        prob_pre_vals.append(expected_1)

    prefix = cfg["feature_prefix"]
    working[f"{prefix}_team1_pre"] = team1_pre_vals
    working[f"{prefix}_team2_pre"] = team2_pre_vals
    working[f"{prefix}_diff_pre"] = diff_pre_vals
    working[f"{prefix}_prob_team1_pre"] = prob_pre_vals

    elo_cols = [
        f"{prefix}_team1_pre",
        f"{prefix}_team2_pre",
        f"{prefix}_diff_pre",
        f"{prefix}_prob_team1_pre",
    ]

    features = working[["row_id", *elo_cols]].set_index("row_id")
    out = out.set_index("row_id").join(features, how="left").reset_index()

    # Canonical ordering point: from here onward, rows are globally sorted by
    # match_date and validation-consistent temporal keys, with stable tie breaks.
    final_sort_cols = _temporal_sort_columns(out, stable_tie_breaker="row_id")
    out = out.sort_values(final_sort_cols, ascending=True, kind="mergesort").reset_index(drop=True)
    out = out.drop(columns=["row_id"])

    return out
