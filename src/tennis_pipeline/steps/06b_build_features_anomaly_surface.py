"""Step 06b: build surface-aware anomaly features from pre-match-safe inputs."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
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
    "knn_neighbors": 10,
    "knn_reference_size": 5000,
    "knn_chunk_size": 2048,
    "random_state": 42,
    "artifact_output_dir": None,
    "artifact_top_n": 25,
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

    knn_neighbors = normalized.get("knn_neighbors")
    if not isinstance(knn_neighbors, int) or knn_neighbors < 1:
        raise TypeError("config['knn_neighbors'] must be an int >= 1")

    knn_reference_size = normalized.get("knn_reference_size")
    if not isinstance(knn_reference_size, int) or knn_reference_size < 2:
        raise TypeError("config['knn_reference_size'] must be an int >= 2")

    knn_chunk_size = normalized.get("knn_chunk_size")
    if not isinstance(knn_chunk_size, int) or knn_chunk_size < 1:
        raise TypeError("config['knn_chunk_size'] must be an int >= 1")

    for key in ("emit_surface_anomaly_z", "drop_rows_all_missing"):
        if not isinstance(normalized.get(key), bool):
            raise TypeError(f"config['{key}'] must be a bool")

    artifact_output_dir = normalized.get("artifact_output_dir")
    if artifact_output_dir is not None and not isinstance(artifact_output_dir, (str, Path)):
        raise TypeError("config['artifact_output_dir'] must be None, str, or Path")

    artifact_top_n = normalized.get("artifact_top_n")
    if not isinstance(artifact_top_n, int) or artifact_top_n < 1:
        raise TypeError("config['artifact_top_n'] must be an int >= 1")

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


def _minmax(values: np.ndarray) -> np.ndarray:
    min_value = float(np.nanmin(values))
    max_value = float(np.nanmax(values))
    if max_value - min_value <= 0:
        return np.zeros_like(values, dtype=float)
    return (values - min_value) / (max_value - min_value)


def _knn_distance_score(
    matrix: np.ndarray,
    k_neighbors: int,
    *,
    random_state: int | None,
    reference_size: int,
    chunk_size: int,
) -> np.ndarray:
    if len(matrix) <= 1:
        return np.zeros(len(matrix), dtype=float)

    n_rows = len(matrix)
    use_full_reference = n_rows <= reference_size
    if use_full_reference:
        ref_indices = np.arange(n_rows, dtype=int)
    else:
        rng = np.random.default_rng(random_state)
        ref_indices = np.sort(rng.choice(n_rows, size=reference_size, replace=False))

    ref_matrix = matrix[ref_indices]
    k = min(max(1, int(k_neighbors)), len(ref_matrix) - 1)
    kth_distances = np.zeros(n_rows, dtype=float)

    for start in range(0, n_rows, chunk_size):
        stop = min(start + chunk_size, n_rows)
        block = matrix[start:stop]
        diffs = block[:, None, :] - ref_matrix[None, :, :]
        squared_distances = np.sum(diffs * diffs, axis=2, dtype=np.float64)
        block_distances = np.sqrt(squared_distances).astype(float, copy=False)

        if use_full_reference:
            block_indices = np.arange(start, stop, dtype=int)
            local_ref_pos = block_indices - start
            block_distances[local_ref_pos, block_indices] = np.inf

        kth_distances[start:stop] = np.partition(block_distances, kth=k - 1, axis=1)[:, k - 1]

    return _minmax(kth_distances)


def _avg_path_length(sample_size: int) -> float:
    if sample_size <= 1:
        return 0.0
    harmonic = np.log(sample_size - 1) + 0.5772156649
    return float(2.0 * harmonic - (2.0 * (sample_size - 1) / sample_size))


def _build_isolation_tree(
    matrix: np.ndarray,
    indices: np.ndarray,
    rng: np.random.Generator,
    depth: int,
    max_depth: int,
) -> dict[str, Any]:
    if len(indices) <= 1 or depth >= max_depth:
        return {"leaf": True, "size": int(len(indices))}

    feature = int(rng.integers(0, matrix.shape[1]))
    feature_values = matrix[indices, feature]
    low = float(np.min(feature_values))
    high = float(np.max(feature_values))
    if not np.isfinite(low) or not np.isfinite(high) or low == high:
        return {"leaf": True, "size": int(len(indices))}

    split = float(rng.uniform(low, high))
    left_mask = feature_values < split
    left_indices = indices[left_mask]
    right_indices = indices[~left_mask]
    if len(left_indices) == 0 or len(right_indices) == 0:
        return {"leaf": True, "size": int(len(indices))}

    return {
        "leaf": False,
        "feature": feature,
        "split": split,
        "size": int(len(indices)),
        "left": _build_isolation_tree(matrix, left_indices, rng, depth + 1, max_depth),
        "right": _build_isolation_tree(matrix, right_indices, rng, depth + 1, max_depth),
    }


def _path_length_for_row(row: np.ndarray, node: Mapping[str, Any], depth: int = 0) -> float:
    if bool(node.get("leaf", False)):
        return float(depth + _avg_path_length(int(node.get("size", 1))))
    feature = int(node["feature"])
    next_node = node["left"] if row[feature] < float(node["split"]) else node["right"]
    return _path_length_for_row(row, next_node, depth + 1)


def _iforest_score(matrix: np.ndarray, random_state: int | None, n_trees: int = 100, max_samples: int = 256) -> np.ndarray:
    if len(matrix) <= 1:
        return np.zeros(len(matrix), dtype=float)

    sample_size = min(max_samples, len(matrix))
    max_depth = int(np.ceil(np.log2(sample_size)))
    rng = np.random.default_rng(random_state)
    path_sum = np.zeros(len(matrix), dtype=float)

    for _ in range(n_trees):
        sampled_indices = rng.choice(len(matrix), size=sample_size, replace=False)
        tree = _build_isolation_tree(matrix, sampled_indices, rng, depth=0, max_depth=max_depth)
        path_sum += np.array([_path_length_for_row(row, tree) for row in matrix], dtype=float)

    avg_paths = path_sum / float(n_trees)
    cn = _avg_path_length(sample_size)
    if cn <= 0:
        return np.zeros(len(matrix), dtype=float)
    raw_scores = np.power(2.0, -(avg_paths / cn))
    return _minmax(raw_scores)


def _emit_artifacts(
    base_df: pd.DataFrame,
    *,
    output_dir: Path,
    surface_col: str,
    top_n: int,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    key_columns = [c for c in ("event_id", "match_id", "match_date", "team1_player_id", "team2_player_id") if c in base_df.columns]
    score_columns = [
        c
        for c in ("anomaly_score", "robust_z_anomaly_score", "knn_anomaly_score", "iforest_anomaly_score", "anomaly_flag")
        if c in base_df.columns
    ]

    surface_summary = (
        base_df.groupby(surface_col, dropna=False)
        .agg(
            match_count=("anomaly_score", "size"),
            anomaly_score_mean=("anomaly_score", "mean"),
            robust_z_score_mean=("robust_z_anomaly_score", "mean"),
            knn_score_mean=("knn_anomaly_score", "mean"),
            iforest_score_mean=("iforest_anomaly_score", "mean"),
            anomaly_flag_rate=("anomaly_flag", "mean"),
        )
        .reset_index()
        .sort_values("anomaly_score_mean", ascending=False)
    )
    surface_summary.to_csv(output_dir / "anomaly_summary_by_surface.csv", index=False)

    top_rows = base_df.sort_values("anomaly_score", ascending=False).head(top_n)
    top_rows.loc[:, [*key_columns, surface_col, *score_columns]].to_csv(output_dir / "anomaly_top_rows.csv", index=False)

    summary_payload = {
        "row_count": int(len(base_df)),
        "anomaly_flag_count": int(base_df["anomaly_flag"].sum()),
        "anomaly_flag_rate": float(base_df["anomaly_flag"].mean()),
        "score_columns": score_columns,
    }
    (output_dir / "anomaly_summary.json").write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")


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
    robust_z_score = z_matrix.mean(axis=1)
    robust_z_score_norm = _minmax(robust_z_score)

    feature_matrix = working[selected_cols].to_numpy(dtype=float)
    knn_score_norm = _knn_distance_score(
        feature_matrix,
        int(cfg["knn_neighbors"]),
        random_state=cfg["random_state"],
        reference_size=int(cfg["knn_reference_size"]),
        chunk_size=int(cfg["knn_chunk_size"]),
    )
    iforest_score_norm = _iforest_score(feature_matrix, random_state=cfg["random_state"], n_trees=100)

    anomaly_score = np.mean(np.column_stack([robust_z_score_norm, knn_score_norm, iforest_score_norm]), axis=1)

    working["robust_z_anomaly_score"] = robust_z_score
    working["knn_anomaly_score"] = knn_score_norm
    working["iforest_anomaly_score"] = iforest_score_norm
    working["anomaly_score"] = anomaly_score

    if cfg["emit_surface_anomaly_z"]:
        surface_mean = working.groupby(surface_col, dropna=False)["anomaly_score"].transform("mean")
        surface_std = working.groupby(surface_col, dropna=False)["anomaly_score"].transform("std").replace(0, np.nan)
        working["surface_anomaly_z"] = ((working["anomaly_score"] - surface_mean) / surface_std).fillna(0.0)
        working["anomaly_flag"] = (working["surface_anomaly_z"] >= float(cfg["surface_z_threshold"]))
    else:
        working["anomaly_flag"] = (working["anomaly_score"] >= float(cfg["anomaly_threshold"]))

    working["anomaly_flag"] = working["anomaly_flag"].astype(int)

    features = ["anomaly_score", "robust_z_anomaly_score", "knn_anomaly_score", "iforest_anomaly_score", "anomaly_flag"]
    if cfg["emit_surface_anomaly_z"]:
        features.append("surface_anomaly_z")

    merged = out.set_index("__anomaly_row_id").join(
        working[["__anomaly_row_id", *features]].set_index("__anomaly_row_id"), how="left"
    )

    if cfg["drop_rows_all_missing"]:
        merged = merged.loc[~all_missing_mask.to_numpy()].copy(deep=True)

    artifact_output_dir = cfg.get("artifact_output_dir")
    if artifact_output_dir is not None:
        _emit_artifacts(
            merged,
            output_dir=Path(artifact_output_dir),
            surface_col=surface_col,
            top_n=int(cfg["artifact_top_n"]),
        )

    return merged.reset_index(drop=True).drop(columns=["__anomaly_row_id"], errors="ignore")
