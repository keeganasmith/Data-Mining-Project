"""Step 06b: build surface-aware anomaly features from pre-match-safe inputs."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd

_DEFAULT_CONFIG: dict[str, Any] = {
    "feature_columns": [
        "rank_diff",
        "abs_rank_diff",
        "race_rank_diff",
        "abs_race_rank_diff",
        "elo_diff_team1",
        "elo_prob_team1_pre",
        "team1_sgl_roll_rank",
        "team2_sgl_roll_rank",
    ],
    "surface_column": "surface_context",
    "anomaly_threshold": 2.5,
    "surface_z_threshold": 2.0,
    "emit_surface_anomaly_z": True,
    "drop_rows_all_missing": False,
    "random_state": 42,
}


def _normalize_config(config: Mapping[str, Any] | None) -> dict[str, Any]:
    normalized = dict(_DEFAULT_CONFIG)
    if config:
        normalized.update(dict(config))

    feature_columns = normalized.get("feature_columns")
    if not isinstance(feature_columns, list) or not all(isinstance(c, str) and c.strip() for c in feature_columns):
        raise TypeError("config['feature_columns'] must be a list[str]")

    surface_column = normalized.get("surface_column")
    if not isinstance(surface_column, str) or not surface_column.strip():
        raise TypeError("config['surface_column'] must be a non-empty string")

    for key in ("anomaly_threshold", "surface_z_threshold"):
        value = normalized.get(key)
        if not isinstance(value, (int, float)):
            raise TypeError(f"config['{key}'] must be numeric")

    for key in ("emit_surface_anomaly_z", "drop_rows_all_missing"):
        if not isinstance(normalized.get(key), bool):
            raise TypeError(f"config['{key}'] must be a bool")

    random_state = normalized.get("random_state")
    if random_state is not None and not isinstance(random_state, int):
        raise TypeError("config['random_state'] must be an int or None")

    return normalized


def _safe_scale(series: pd.Series) -> float:
    mad = float((series - series.median()).abs().median())
    if mad > 0:
        return 1.4826 * mad

    std = float(series.std(ddof=0))
    if std > 0:
        return std

    return 1.0


def _surface_impute(df: pd.DataFrame, feature_cols: list[str], surface_col: str) -> pd.DataFrame:
    out = df.copy(deep=True)

    for col in feature_cols:
        # Prefer within-surface medians first.
        out[col] = out[col].fillna(out.groupby(surface_col, dropna=False)[col].transform("median"))
        # Fall back to global median, then 0.0 if column is entirely missing.
        global_median = out[col].median()
        fill_value = float(global_median) if pd.notna(global_median) else 0.0
        out[col] = out[col].fillna(fill_value)

    return out


def run(df_or_path: pd.DataFrame, config: Mapping[str, Any] | None = None) -> pd.DataFrame:
    """Add row-aligned anomaly features using only pre-match-safe columns."""

    if not isinstance(df_or_path, pd.DataFrame):
        raise TypeError("06b_build_features_anomaly_surface.run expects a pandas DataFrame input")

    cfg = _normalize_config(config)

    out = df_or_path.copy(deep=True)
    out["__anomaly_row_id"] = range(len(out))

    surface_col = cfg["surface_column"]
    if surface_col not in out.columns:
        out[surface_col] = "Unknown"

    out[surface_col] = out[surface_col].astype("string").fillna("Unknown")
    out[surface_col] = out[surface_col].replace({"": "Unknown", "<NA>": "Unknown"})

    selected_cols = [c for c in cfg["feature_columns"] if c in out.columns]
    if not selected_cols:
        out["anomaly_score"] = 0.0
        out["anomaly_flag"] = 0
        if cfg["emit_surface_anomaly_z"]:
            out["surface_anomaly_z"] = 0.0
        return out.drop(columns=["__anomaly_row_id"], errors="ignore")

    numeric_block = out[selected_cols].apply(pd.to_numeric, errors="coerce")
    all_missing_mask = numeric_block.isna().all(axis=1)

    working = pd.concat([out[["__anomaly_row_id", surface_col]], numeric_block], axis=1)
    working = _surface_impute(working, selected_cols, surface_col)

    z_by_feature: dict[str, np.ndarray] = {}
    for feature_col in selected_cols:
        z_values = pd.Series(np.zeros(len(working), dtype=float), index=working.index, dtype=float)
        for _surface, idx in working.groupby(surface_col, dropna=False).groups.items():
            values = working.loc[idx, feature_col]
            center = float(values.median())
            scale = _safe_scale(values)
            z_values.loc[idx] = (values.to_numpy(dtype=float) - center) / scale
        z_by_feature[feature_col] = z_values.to_numpy(dtype=float)

    z_matrix = np.column_stack([np.abs(z_by_feature[c]) for c in selected_cols])
    anomaly_score = z_matrix.mean(axis=1)

    working["anomaly_score"] = anomaly_score

    if cfg["emit_surface_anomaly_z"]:
        surface_mean = working.groupby(surface_col, dropna=False)["anomaly_score"].transform("mean")
        surface_std = working.groupby(surface_col, dropna=False)["anomaly_score"].transform("std").replace(0, np.nan)
        working["surface_anomaly_z"] = ((working["anomaly_score"] - surface_mean) / surface_std).fillna(0.0)
        working["anomaly_flag"] = (working["surface_anomaly_z"] >= float(cfg["surface_z_threshold"]))
    else:
        working["anomaly_flag"] = (working["anomaly_score"] >= float(cfg["anomaly_threshold"]))

    working["anomaly_flag"] = working["anomaly_flag"].astype(int)

    features = ["anomaly_score", "anomaly_flag"]
    if cfg["emit_surface_anomaly_z"]:
        features.append("surface_anomaly_z")

    merged = out.set_index("__anomaly_row_id").join(
        working[["__anomaly_row_id", *features]].set_index("__anomaly_row_id"), how="left"
    )

    if cfg["drop_rows_all_missing"]:
        merged = merged.loc[~all_missing_mask.to_numpy()].copy(deep=True)

    return merged.reset_index(drop=True).drop(columns=["__anomaly_row_id"], errors="ignore")
