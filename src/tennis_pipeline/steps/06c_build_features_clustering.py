"""Step 06c: optional clustering features from leakage-safe temporal signals.

This step is designed to be leakage-safe by default:
- It only uses numeric columns whose names match configured safe prefixes
  (default: ``elo_`` and ``temporal_``).
- With ``fit_scope='train_only'`` (default), clustering models are fit on the
  temporally earliest training slice only, then labels are assigned to all rows.

Notes on leakage:
- Fitting unsupervised clustering on the full dataset can still leak future
  distribution information into earlier rows.
- Use ``fit_scope='all_data'`` only for diagnostics; it is not strict
  pre-match leakage-safe evaluation.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN, KMeans
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from tennis_pipeline.config import CLUSTERING_DEFAULTS
from tennis_pipeline.temporal_ordering import prepare_temporal_ordering

_METHODS = {"kmeans", "dbscan", "both"}
_FIT_SCOPES = {"train_only", "all_data"}


def _normalize_config(config: Mapping[str, Any] | None) -> dict[str, Any]:
    normalized = dict(CLUSTERING_DEFAULTS)
    if config:
        normalized.update(dict(config))

    method = normalized.get("method")
    if method not in _METHODS:
        raise ValueError(f"config['method'] must be one of {_METHODS}; got {method!r}")

    prefixes = normalized.get("source_feature_prefixes")
    if not isinstance(prefixes, list) or not prefixes or not all(isinstance(v, str) and v.strip() for v in prefixes):
        raise TypeError("config['source_feature_prefixes'] must be a non-empty list[str]")
    normalized["source_feature_prefixes"] = [v.strip() for v in prefixes]

    train_fraction = normalized.get("train_fraction")
    if not isinstance(train_fraction, (int, float)):
        raise TypeError("config['train_fraction'] must be numeric")
    if not (0 < float(train_fraction) <= 1):
        raise ValueError("config['train_fraction'] must be in (0, 1]")
    normalized["train_fraction"] = float(train_fraction)

    fit_scope = normalized.get("fit_scope")
    if fit_scope not in _FIT_SCOPES:
        raise ValueError(f"config['fit_scope'] must be one of {_FIT_SCOPES}; got {fit_scope!r}")

    return normalized


def _select_source_columns(df: pd.DataFrame, prefixes: list[str]) -> list[str]:
    cols: list[str] = []
    for col in df.columns:
        if not any(col.startswith(prefix) for prefix in prefixes):
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            cols.append(col)
    return sorted(cols)


def _build_temporal_train_mask(df: pd.DataFrame, *, train_fraction: float, fit_scope: str) -> pd.Series:
    if fit_scope == "all_data":
        return pd.Series(True, index=df.index)

    ordered = df.copy(deep=False)
    ordered["__row_id"] = range(len(ordered))
    ordered, sort_cols, _, temp_cols = prepare_temporal_ordering(ordered, stable_tie_breaker="__row_id")
    ordered = ordered.sort_values(sort_cols, kind="mergesort")

    cutoff = max(1, int(np.floor(len(ordered) * train_fraction)))
    train_row_ids = set(ordered["__row_id"].iloc[:cutoff].tolist())
    mask = pd.Series(df.index.to_series().map(lambda idx: idx in train_row_ids), index=df.index)
    return mask.astype(bool)


def _scaled_matrix(df: pd.DataFrame, cols: list[str]) -> tuple[np.ndarray, StandardScaler]:
    matrix = df.loc[:, cols].apply(pd.to_numeric, errors="coerce")
    matrix = matrix.fillna(matrix.median(numeric_only=True)).fillna(0.0)
    scaler = StandardScaler()
    return scaler.fit_transform(matrix.to_numpy(dtype=float)), scaler


def _assign_dbscan_labels(
    x_all: np.ndarray,
    x_train: np.ndarray,
    train_mask: np.ndarray,
    *,
    eps: float,
    min_samples: int,
) -> np.ndarray:
    model = DBSCAN(eps=eps, min_samples=min_samples)
    train_labels = model.fit_predict(x_train)

    labels = np.full(shape=(x_all.shape[0],), fill_value=-1, dtype=int)
    labels[train_mask] = train_labels

    core_idx = getattr(model, "core_sample_indices_", np.array([], dtype=int))
    if core_idx.size == 0:
        return labels

    core_points = x_train[core_idx]
    core_labels = train_labels[core_idx]
    nn = NearestNeighbors(n_neighbors=1).fit(core_points)
    distances, neighbors = nn.kneighbors(x_all)

    for i in range(x_all.shape[0]):
        if train_mask[i]:
            continue
        if distances[i, 0] <= eps:
            labels[i] = int(core_labels[neighbors[i, 0]])
    return labels


def run(df_or_path: pd.DataFrame, config: Mapping[str, Any] | None = None) -> pd.DataFrame:
    if not isinstance(df_or_path, pd.DataFrame):
        raise TypeError("06c_build_features_clustering.run expects a pandas DataFrame input")

    cfg = _normalize_config(config)
    out = df_or_path.copy(deep=True)

    source_cols = _select_source_columns(out, cfg["source_feature_prefixes"])
    if not source_cols:
        out["cluster_kmeans_id"] = -1
        out["cluster_dbscan_id"] = -1
        return out

    train_mask = _build_temporal_train_mask(out, train_fraction=cfg["train_fraction"], fit_scope=cfg["fit_scope"])

    ordered = out.copy(deep=False)
    ordered["__row_id"] = range(len(ordered))
    ordered, sort_cols, _, temp_cols = prepare_temporal_ordering(ordered, stable_tie_breaker="__row_id")
    ordered = ordered.sort_values(sort_cols, kind="mergesort")
    ordered = ordered.drop(columns=temp_cols, errors="ignore")

    x_all, _scaler = _scaled_matrix(ordered, source_cols)
    ordered_train_mask = train_mask.loc[ordered.index].to_numpy(dtype=bool)
    x_train = x_all[ordered_train_mask]
    if x_train.shape[0] == 0:
        x_train = x_all

    method = cfg["method"]
    run_kmeans = method in {"kmeans", "both"}
    run_dbscan = method in {"dbscan", "both"}

    if run_kmeans:
        kmeans = KMeans(
            n_clusters=int(cfg["kmeans_n_clusters"]),
            random_state=int(cfg["kmeans_random_state"]),
            n_init=int(cfg["kmeans_n_init"]),
        )
        kmeans.fit(x_train)
        ordered["cluster_kmeans_id"] = kmeans.predict(x_all).astype(int)

    if run_dbscan:
        ordered["cluster_dbscan_id"] = _assign_dbscan_labels(
            x_all,
            x_train,
            ordered_train_mask,
            eps=float(cfg["dbscan_eps"]),
            min_samples=int(cfg["dbscan_min_samples"]),
        )

    result = ordered.sort_values("__row_id", kind="mergesort").drop(columns=["__row_id"], errors="ignore")
    return result
