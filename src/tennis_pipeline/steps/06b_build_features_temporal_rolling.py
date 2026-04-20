"""Step 06b: build leakage-safe temporal rolling player features.

Generated column naming
-----------------------
Given ``feature_prefix`` (default: ``"temporal"``), this step appends:

- ``{prefix}_team1_rolling_win_pct``
- ``{prefix}_team2_rolling_win_pct``
- ``{prefix}_team1_rolling_avg_elo``
- ``{prefix}_team2_rolling_avg_elo``
- ``{prefix}_team1_rolling_avg_<stat>``
- ``{prefix}_team2_rolling_avg_<stat>``

Leakage safeguard
-----------------
For each match row, features are read from per-player state *before* applying
that row's outcomes/statistics; state updates happen strictly after capture.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from tennis_pipeline.config import TEMPORAL_ROLLING_DEFAULTS
from tennis_pipeline.temporal_ordering import prepare_temporal_ordering


@dataclass
class _RollingMeanState:
    window: int | None
    total: float = 0.0
    count: int = 0
    values: deque[float] = field(default_factory=deque)

    def mean(self, *, default: float = 0.0) -> float:
        if self.count <= 0:
            return default
        return self.total / self.count

    def update(self, value: float) -> None:
        self.values.append(value)
        self.total += value
        self.count += 1
        if self.window is not None and self.count > self.window:
            self.total -= self.values.popleft()
            self.count -= 1


def _normalize_config(config: Mapping[str, Any] | None) -> dict[str, Any]:
    normalized = dict(TEMPORAL_ROLLING_DEFAULTS)
    if config:
        normalized.update(dict(config))

    prefix = normalized.get("feature_prefix")
    if not isinstance(prefix, str) or not prefix.strip():
        raise TypeError("config['feature_prefix'] must be a non-empty string")
    normalized["feature_prefix"] = prefix.strip()

    strict_validation = normalized.get("strict_validation")
    if not isinstance(strict_validation, bool):
        raise TypeError("config['strict_validation'] must be a bool")

    rolling_window = normalized.get("rolling_window_matches")
    if rolling_window is not None:
        if not isinstance(rolling_window, int):
            raise TypeError("config['rolling_window_matches'] must be an int or None")
        if rolling_window <= 0:
            raise ValueError("config['rolling_window_matches'] must be > 0 when provided")

    include_elo_avg = normalized.get("include_elo_average")
    if not isinstance(include_elo_avg, bool):
        raise TypeError("config['include_elo_average'] must be a bool")

    min_coverage = normalized.get("paired_stats_min_numeric_coverage")
    if not isinstance(min_coverage, (int, float)):
        raise TypeError("config['paired_stats_min_numeric_coverage'] must be numeric")
    if min_coverage < 0 or min_coverage > 1:
        raise ValueError("config['paired_stats_min_numeric_coverage'] must be in [0, 1]")

    return normalized


def _validate_required_columns(df: pd.DataFrame) -> None:
    required = ["match_date", "team1_player_id", "team2_player_id"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"[build_features_temporal_rolling] Missing required columns: {missing}")


def _resolve_result(df: pd.DataFrame) -> pd.Series:
    if "team1_wins" in df.columns:
        team1_wins = pd.to_numeric(df["team1_wins"], errors="coerce")
        if team1_wins.isin([0, 1]).all():
            return team1_wins.astype(float)

    required = {"winner_player_id", "team1_player_id", "team2_player_id"}
    if required.issubset(df.columns):
        winner_is_team1 = (df["winner_player_id"] == df["team1_player_id"]).astype(float)
        winner_is_team2 = (df["winner_player_id"] == df["team2_player_id"]).astype(float)
        if ((winner_is_team1 + winner_is_team2) == 1.0).all():
            return winner_is_team1

    raise ValueError("Unable to infer row outcomes from team1_wins or winner/team ids.")


def _paired_numeric_stats(df: pd.DataFrame, *, min_coverage: float) -> list[tuple[str, str, str]]:
    pairs: list[tuple[str, str, str]] = []
    blocked_suffixes = {"player_id", "wins"}

    for col in df.columns:
        if not col.startswith("team1_"):
            continue
        suffix = col[len("team1_") :]
        if suffix in blocked_suffixes:
            continue

        mate = f"team2_{suffix}"
        if mate not in df.columns:
            continue

        s1 = pd.to_numeric(df[col], errors="coerce")
        s2 = pd.to_numeric(df[mate], errors="coerce")
        coverage = float((s1.notna() | s2.notna()).mean()) if len(df) else 0.0
        if coverage >= min_coverage:
            pairs.append((col, mate, suffix))

    return pairs


def run(df_or_path: pd.DataFrame, config: Mapping[str, Any] | None = None) -> pd.DataFrame:
    if not isinstance(df_or_path, pd.DataFrame):
        raise TypeError("06b_build_features_temporal_rolling.run expects a pandas DataFrame input")

    cfg = _normalize_config(config)
    out = df_or_path.copy(deep=True)
    out["row_id"] = range(len(out))

    if cfg["strict_validation"]:
        _validate_required_columns(out)

    working = out.copy(deep=True)
    working["match_date"] = pd.to_datetime(working["match_date"], errors="coerce", utc=False)

    working, sort_cols, _tie_breaker_text, temp_sort_cols = prepare_temporal_ordering(
        working, stable_tie_breaker="row_id"
    )
    working = working.sort_values(sort_cols, ascending=True, kind="mergesort")

    team1_score = _resolve_result(working)
    team1_score.index = working["row_id"].to_numpy()

    window = cfg["rolling_window_matches"]
    prefix = cfg["feature_prefix"]

    rolling_wins: dict[Any, _RollingMeanState] = {}
    rolling_elo: dict[Any, _RollingMeanState] = {}

    stat_pairs = _paired_numeric_stats(working, min_coverage=float(cfg["paired_stats_min_numeric_coverage"]))
    stat_states: dict[str, dict[Any, _RollingMeanState]] = {suffix: {} for _l, _r, suffix in stat_pairs}
    stat_numeric: dict[str, pd.Series] = {}
    for left, right, _suffix in stat_pairs:
        stat_numeric[left] = pd.to_numeric(working[left], errors="coerce")
        stat_numeric[right] = pd.to_numeric(working[right], errors="coerce")

    t1_roll_win_vals: list[float] = []
    t2_roll_win_vals: list[float] = []
    t1_roll_elo_vals: list[float] = []
    t2_roll_elo_vals: list[float] = []
    t1_roll_stat_vals: dict[str, list[float]] = {suffix: [] for _l, _r, suffix in stat_pairs}
    t2_roll_stat_vals: dict[str, list[float]] = {suffix: [] for _l, _r, suffix in stat_pairs}

    elo_team1_col = cfg["elo_team1_pre_column"]
    elo_team2_col = cfg["elo_team2_pre_column"]
    default_elo = float(cfg["default_elo"])

    has_elo_cols = cfg["include_elo_average"] and elo_team1_col in working.columns and elo_team2_col in working.columns
    if has_elo_cols:
        elo_team1_vals = pd.to_numeric(working[elo_team1_col], errors="coerce")
        elo_team2_vals = pd.to_numeric(working[elo_team2_col], errors="coerce")
    else:
        elo_team1_vals = pd.Series(default_elo, index=working.index)
        elo_team2_vals = pd.Series(default_elo, index=working.index)

    for row in working.itertuples(index=False):
        idx = getattr(row, "row_id")
        p1 = getattr(row, "team1_player_id")
        p2 = getattr(row, "team2_player_id")

        if pd.isna(p1) or pd.isna(p2):
            raise ValueError("Encountered null team player id while computing temporal rolling features")

        s1 = float(team1_score.loc[idx])

        p1_wins_state = rolling_wins.setdefault(p1, _RollingMeanState(window))
        p2_wins_state = rolling_wins.setdefault(p2, _RollingMeanState(window))
        t1_roll_win_vals.append(p1_wins_state.mean(default=0.5))
        t2_roll_win_vals.append(p2_wins_state.mean(default=0.5))

        p1_elo_state = rolling_elo.setdefault(p1, _RollingMeanState(window))
        p2_elo_state = rolling_elo.setdefault(p2, _RollingMeanState(window))
        t1_roll_elo_vals.append(p1_elo_state.mean(default=default_elo))
        t2_roll_elo_vals.append(p2_elo_state.mean(default=default_elo))

        for left, right, suffix in stat_pairs:
            metric_state = stat_states[suffix]
            p1_metric_state = metric_state.setdefault(p1, _RollingMeanState(window))
            p2_metric_state = metric_state.setdefault(p2, _RollingMeanState(window))
            t1_roll_stat_vals[suffix].append(p1_metric_state.mean(default=0.0))
            t2_roll_stat_vals[suffix].append(p2_metric_state.mean(default=0.0))

        p1_wins_state.update(s1)
        p2_wins_state.update(1.0 - s1)

        p1_elo_val = elo_team1_vals.loc[idx]
        p2_elo_val = elo_team2_vals.loc[idx]
        if pd.notna(p1_elo_val):
            p1_elo_state.update(float(p1_elo_val))
        if pd.notna(p2_elo_val):
            p2_elo_state.update(float(p2_elo_val))

        for left, right, suffix in stat_pairs:
            p1_val = stat_numeric[left].loc[idx]
            p2_val = stat_numeric[right].loc[idx]
            if pd.notna(p1_val):
                stat_states[suffix][p1].update(float(p1_val))
            if pd.notna(p2_val):
                stat_states[suffix][p2].update(float(p2_val))

    working[f"{prefix}_team1_rolling_win_pct"] = t1_roll_win_vals
    working[f"{prefix}_team2_rolling_win_pct"] = t2_roll_win_vals
    working[f"{prefix}_team1_rolling_avg_elo"] = t1_roll_elo_vals
    working[f"{prefix}_team2_rolling_avg_elo"] = t2_roll_elo_vals
    for _left, _right, suffix in stat_pairs:
        working[f"{prefix}_team1_rolling_avg_{suffix}"] = t1_roll_stat_vals[suffix]
        working[f"{prefix}_team2_rolling_avg_{suffix}"] = t2_roll_stat_vals[suffix]

    roll_cols = [
        f"{prefix}_team1_rolling_win_pct",
        f"{prefix}_team2_rolling_win_pct",
        f"{prefix}_team1_rolling_avg_elo",
        f"{prefix}_team2_rolling_avg_elo",
        *[f"{prefix}_team1_rolling_avg_{suffix}" for _l, _r, suffix in stat_pairs],
        *[f"{prefix}_team2_rolling_avg_{suffix}" for _l, _r, suffix in stat_pairs],
    ]

    features = working[["row_id", *roll_cols]].set_index("row_id")
    out = out.set_index("row_id").join(features, how="left").reset_index()

    out, final_sort_cols, _tie_breaker_text, final_temp_sort_cols = prepare_temporal_ordering(
        out, stable_tie_breaker="row_id"
    )
    out = out.sort_values(final_sort_cols, ascending=True, kind="mergesort").reset_index(drop=True)
    out = out.drop(columns=["row_id", *temp_sort_cols, *final_temp_sort_cols], errors="ignore")
    return out
