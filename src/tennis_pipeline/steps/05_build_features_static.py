"""Step 05: build deterministic static (pre-match) features."""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any
from tennis_pipeline.temporal_ordering import prepare_temporal_ordering

import pandas as pd

_DEFAULT_CONFIG: dict[str, Any] = {
    "drop_missing_rank_diff": True,
}


def _normalize_config(config: Mapping[str, Any] | None) -> dict[str, Any]:
    normalized = dict(_DEFAULT_CONFIG)
    if config:
        normalized.update(dict(config))
    return normalized


def _normalize_suffix(value: str) -> str:
    value = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", value)
    value = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1_\2", value)
    clean = re.sub(r"[^0-9a-zA-Z]+", "_", value).strip("_").lower()
    return clean


def _candidate_pairs(df: pd.DataFrame) -> list[tuple[str, str, str]]:
    pairs: list[tuple[str, str, str]] = []

    for col in df.columns:
        if col.startswith("team1_"):
            suffix = col[len("team1_") :]
            mate = f"team2_{suffix}"
            feature_name = suffix
        elif col.startswith("PlayerTeam1."):
            suffix = col[len("PlayerTeam1.") :]
            mate = f"PlayerTeam2.{suffix}"
            feature_name = _normalize_suffix(suffix)
        else:
            continue

        if mate in df.columns:
            pairs.append((col, mate, feature_name))

    deduped: dict[str, tuple[str, str, str]] = {}
    for team1_col, team2_col, feat_name in pairs:
        key = _normalize_suffix(feat_name)
        deduped[key] = (team1_col, team2_col, key)

    return list(deduped.values())


def _as_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def build_paired_player_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy(deep=True)

    new_cols = {}

    for team1_col, team2_col, feat in _candidate_pairs(out):
        s1 = out[team1_col]
        s2 = out[team2_col]

        n1 = _as_numeric(s1)
        n2 = _as_numeric(s2)
        numeric_coverage = (n1.notna() | n2.notna()).mean() if len(out) else 0.0

        if numeric_coverage >= 0.5:
            new_cols[f"diff_{feat}"] = n1 - n2
            new_cols[f"abs_diff_{feat}"] = (n1 - n2).abs()
        else:
            left = s1.astype("string").str.strip().str.lower()
            right = s2.astype("string").str.strip().str.lower()
            new_cols[f"same_{feat}"] = (left == right).fillna(False).astype(int)

    # Add all columns at once (key fix)
    out = pd.concat([out, pd.DataFrame(new_cols)], axis=1)

    return out


def add_rank_and_race_diff_features(df: pd.DataFrame, *, drop_missing_rank_diff: bool = True) -> pd.DataFrame:
    """Add canonical rank/race rank difference features used by models."""

    out = df.copy(deep=True)

    rank_sources: list[tuple[str, str]] = [
        ("team1_sgl_roll_rank", "team2_sgl_roll_rank"),
        ("PlayerTeam1.SglRollRank", "PlayerTeam2.SglRollRank"),
    ]
    race_sources: list[tuple[str, str]] = [
        ("team1_sgl_race_rank", "team2_sgl_race_rank"),
        ("PlayerTeam1.SglRaceRank", "PlayerTeam2.SglRaceRank"),
    ]

    if "diff_sglrollrank" in out.columns:
        out["rank_diff"] = _as_numeric(out["diff_sglrollrank"])
    else:
        for left_col, right_col in rank_sources:
            if left_col in out.columns and right_col in out.columns:
                out["rank_diff"] = _as_numeric(out[left_col]) - _as_numeric(out[right_col])
                break

    if "rank_diff" in out.columns:
        out["abs_rank_diff"] = _as_numeric(out["rank_diff"]).abs()

    if "diff_sglracerank" in out.columns:
        out["race_rank_diff"] = _as_numeric(out["diff_sglracerank"])
    else:
        for left_col, right_col in race_sources:
            if left_col in out.columns and right_col in out.columns:
                out["race_rank_diff"] = _as_numeric(out[left_col]) - _as_numeric(out[right_col])
                break

    if "race_rank_diff" in out.columns:
        out["abs_race_rank_diff"] = _as_numeric(out["race_rank_diff"]).abs()

    if drop_missing_rank_diff and "rank_diff" in out.columns:
        out = out[out["rank_diff"].notna()].copy(deep=True)

    return out


def normalize_surface_context(df: pd.DataFrame) -> pd.DataFrame:
    """Resolve canonical surface context without introducing temporal leakage."""

    out = df.copy(deep=True)

    surface_candidates = ["surface_context", "CourtSurface", "Court.Surface", "Court"]
    surface_source = next((c for c in surface_candidates if c in out.columns), None)
    if surface_source is not None:
        out["surface_context"] = out[surface_source].astype("string").fillna("Unknown")
        out["surface_context"] = out["surface_context"].replace({"": "Unknown", "<NA>": "Unknown"})
    else:
        out["surface_context"] = "Unknown"

    if "court_context" in out.columns:
        out["court_context"] = out["court_context"].astype("string").fillna("Unknown")
    else:
        out["court_context"] = "Unknown"

    return out


def run(df_or_path: pd.DataFrame, config: Mapping[str, Any] | None = None) -> pd.DataFrame:
    """Build leakage-safe deterministic static features."""

    if not isinstance(df_or_path, pd.DataFrame):
        raise TypeError("05_build_features_static.run expects a pandas DataFrame input")

    cfg = _normalize_config(config)

    out = build_paired_player_features(df_or_path)
    out = add_rank_and_race_diff_features(out, drop_missing_rank_diff=cfg.get("drop_missing_rank_diff", True))
    out = normalize_surface_context(out)

    out["__order_row_pos"] = range(len(out))
    out, sort_cols, tie_breaker_text, _temp_cols = prepare_temporal_ordering(
        out, stable_tie_breaker="__order_row_pos"
    )
    out = out.sort_values(sort_cols, kind="mergesort")
    return out
