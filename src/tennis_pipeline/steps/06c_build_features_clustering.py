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

import json
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.cluster import DBSCAN, KMeans
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from tennis_pipeline.config import CLUSTERING_DEFAULTS
from tennis_pipeline.temporal_ordering import prepare_temporal_ordering

_METHODS = {"kmeans", "dbscan", "both"}
_FIT_SCOPES = {"train_only", "all_data"}
_PARALLEL_BACKENDS = {"loky", "threading"}
_TUNING_PROFILES = {"fast", "full"}


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

    auto_tune = normalized.get("auto_tune")
    if not isinstance(auto_tune, bool):
        raise TypeError("config['auto_tune'] must be a bool")

    tuning_profile = normalized.get("tuning_profile")
    if tuning_profile not in _TUNING_PROFILES:
        raise ValueError(f"config['tuning_profile'] must be one of {_TUNING_PROFILES}; got {tuning_profile!r}")

    fast_mode_row_threshold = normalized.get("fast_mode_row_threshold")
    if not isinstance(fast_mode_row_threshold, int):
        raise TypeError("config['fast_mode_row_threshold'] must be an int")
    if fast_mode_row_threshold < 1:
        raise ValueError("config['fast_mode_row_threshold'] must be >= 1")

    artifact_path = normalized.get("tuning_artifact_path")
    if not isinstance(artifact_path, str) or not artifact_path.strip():
        raise TypeError("config['tuning_artifact_path'] must be a non-empty string path")

    plot_dir = normalized.get("tuning_plot_dir")
    if not isinstance(plot_dir, str) or not plot_dir.strip():
        raise TypeError("config['tuning_plot_dir'] must be a non-empty string path")

    kmeans_grid = normalized.get("kmeans_tuning_n_clusters")
    if not isinstance(kmeans_grid, list) or not kmeans_grid:
        raise TypeError("config['kmeans_tuning_n_clusters'] must be a non-empty list[int]")
    if not all(isinstance(v, int) and v >= 2 for v in kmeans_grid):
        raise ValueError("config['kmeans_tuning_n_clusters'] values must be int >= 2")

    kmeans_grid_fast = normalized.get("kmeans_tuning_n_clusters_fast")
    if not isinstance(kmeans_grid_fast, list) or not kmeans_grid_fast:
        raise TypeError("config['kmeans_tuning_n_clusters_fast'] must be a non-empty list[int]")
    if not all(isinstance(v, int) and v >= 2 for v in kmeans_grid_fast):
        raise ValueError("config['kmeans_tuning_n_clusters_fast'] values must be int >= 2")

    dbscan_eps_grid = normalized.get("dbscan_tuning_eps")
    if not isinstance(dbscan_eps_grid, list) or not dbscan_eps_grid:
        raise TypeError("config['dbscan_tuning_eps'] must be a non-empty list[float]")
    if not all(isinstance(v, (int, float)) and float(v) > 0 for v in dbscan_eps_grid):
        raise ValueError("config['dbscan_tuning_eps'] values must be > 0")

    dbscan_eps_grid_fast = normalized.get("dbscan_tuning_eps_fast")
    if not isinstance(dbscan_eps_grid_fast, list) or not dbscan_eps_grid_fast:
        raise TypeError("config['dbscan_tuning_eps_fast'] must be a non-empty list[float]")
    if not all(isinstance(v, (int, float)) and float(v) > 0 for v in dbscan_eps_grid_fast):
        raise ValueError("config['dbscan_tuning_eps_fast'] values must be > 0")

    dbscan_min_samples_grid = normalized.get("dbscan_tuning_min_samples")
    if not isinstance(dbscan_min_samples_grid, list) or not dbscan_min_samples_grid:
        raise TypeError("config['dbscan_tuning_min_samples'] must be a non-empty list[int]")
    if not all(isinstance(v, int) and v >= 1 for v in dbscan_min_samples_grid):
        raise ValueError("config['dbscan_tuning_min_samples'] values must be int >= 1")

    dbscan_min_samples_grid_fast = normalized.get("dbscan_tuning_min_samples_fast")
    if not isinstance(dbscan_min_samples_grid_fast, list) or not dbscan_min_samples_grid_fast:
        raise TypeError("config['dbscan_tuning_min_samples_fast'] must be a non-empty list[int]")
    if not all(isinstance(v, int) and v >= 1 for v in dbscan_min_samples_grid_fast):
        raise ValueError("config['dbscan_tuning_min_samples_fast'] values must be int >= 1")

    dbscan_stage1_sample_size = normalized.get("dbscan_stage1_sample_size")
    if not isinstance(dbscan_stage1_sample_size, int):
        raise TypeError("config['dbscan_stage1_sample_size'] must be an int")
    if dbscan_stage1_sample_size < 1:
        raise ValueError("config['dbscan_stage1_sample_size'] must be >= 1")

    dbscan_stage2_sample_size = normalized.get("dbscan_stage2_sample_size")
    if not isinstance(dbscan_stage2_sample_size, int):
        raise TypeError("config['dbscan_stage2_sample_size'] must be an int")
    if dbscan_stage2_sample_size < 1:
        raise ValueError("config['dbscan_stage2_sample_size'] must be >= 1")

    dbscan_top_n = normalized.get("dbscan_top_n")
    if not isinstance(dbscan_top_n, int):
        raise TypeError("config['dbscan_top_n'] must be an int")
    if dbscan_top_n < 1:
        raise ValueError("config['dbscan_top_n'] must be >= 1")

    tuning_time_budget_seconds = normalized.get("tuning_time_budget_seconds")
    if not isinstance(tuning_time_budget_seconds, (int, float)):
        raise TypeError("config['tuning_time_budget_seconds'] must be numeric")
    if float(tuning_time_budget_seconds) <= 0:
        raise ValueError("config['tuning_time_budget_seconds'] must be > 0")
    normalized["tuning_time_budget_seconds"] = float(tuning_time_budget_seconds)

    parallel_n_jobs = normalized.get("parallel_n_jobs")
    if not isinstance(parallel_n_jobs, int):
        raise TypeError("config['parallel_n_jobs'] must be an int")
    if parallel_n_jobs == 0 or parallel_n_jobs < -1:
        raise ValueError("config['parallel_n_jobs'] must be -1 or >= 1")

    parallel_backend = normalized.get("parallel_backend")
    if parallel_backend not in _PARALLEL_BACKENDS:
        raise ValueError(f"config['parallel_backend'] must be one of {_PARALLEL_BACKENDS}; got {parallel_backend!r}")

    parallel_inner_threads = normalized.get("parallel_inner_threads")
    if not isinstance(parallel_inner_threads, int):
        raise TypeError("config['parallel_inner_threads'] must be an int")
    if parallel_inner_threads < 1:
        raise ValueError("config['parallel_inner_threads'] must be >= 1")

    return normalized


def _resolve_tuning_profile(
    cfg: Mapping[str, Any],
    *,
    row_count: int,
    user_config: Mapping[str, Any] | None,
) -> str:
    profile = str(cfg["tuning_profile"])
    explicitly_set = isinstance(user_config, Mapping) and "tuning_profile" in user_config
    if explicitly_set:
        return profile
    if row_count >= int(cfg["fast_mode_row_threshold"]):
        return "fast"
    return profile


def _apply_tuning_profile(cfg: Mapping[str, Any], profile: str) -> dict[str, Any]:
    profile_cfg = dict(cfg)
    if profile == "fast":
        profile_cfg["kmeans_tuning_n_clusters"] = list(cfg["kmeans_tuning_n_clusters_fast"])
        profile_cfg["dbscan_tuning_eps"] = list(cfg["dbscan_tuning_eps_fast"])
        profile_cfg["dbscan_tuning_min_samples"] = list(cfg["dbscan_tuning_min_samples_fast"])
    return profile_cfg


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
    n_jobs: int,
) -> np.ndarray:
    model = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=n_jobs)
    train_labels = model.fit_predict(x_train)

    labels = np.full(shape=(x_all.shape[0],), fill_value=-1, dtype=int)
    labels[train_mask] = train_labels

    core_idx = getattr(model, "core_sample_indices_", np.array([], dtype=int))
    if core_idx.size == 0:
        return labels

    core_points = x_train[core_idx]
    core_labels = train_labels[core_idx]
    nn = NearestNeighbors(n_neighbors=1, n_jobs=n_jobs).fit(core_points)
    distances, neighbors = nn.kneighbors(x_all)
    non_train_close = (~train_mask) & (distances[:, 0] <= eps)
    if np.any(non_train_close):
        labels[non_train_close] = core_labels[neighbors[non_train_close, 0]].astype(int)
    return labels


def _sample_for_silhouette(
    x: np.ndarray,
    labels: np.ndarray,
    *,
    sample_size: int,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray]:
    if sample_size <= 0 or len(x) <= sample_size:
        return x, labels

    rng = np.random.default_rng(seed=random_state)
    unique_labels, counts = np.unique(labels, return_counts=True)
    proportions = counts / counts.sum()
    target_counts = np.floor(proportions * sample_size).astype(int)
    target_counts = np.minimum(target_counts, counts)

    remainder = sample_size - int(target_counts.sum())
    if remainder > 0:
        fractional = proportions * sample_size - np.floor(proportions * sample_size)
        order = np.argsort(fractional)[::-1]
        for idx in order:
            if remainder == 0:
                break
            if target_counts[idx] < counts[idx]:
                target_counts[idx] += 1
                remainder -= 1

    selected_parts: list[np.ndarray] = []
    for label, take in zip(unique_labels, target_counts, strict=True):
        if take <= 0:
            continue
        label_indices = np.flatnonzero(labels == label)
        take_int = int(min(take, len(label_indices)))
        if take_int <= 0:
            continue
        chosen = rng.choice(label_indices, size=take_int, replace=False)
        selected_parts.append(chosen.astype(int, copy=False))

    if not selected_parts:
        sampled_idx = rng.choice(np.arange(len(x)), size=sample_size, replace=False).astype(int, copy=False)
        return x[sampled_idx], labels[sampled_idx]

    sampled_idx = np.concatenate(selected_parts)
    if len(sampled_idx) > sample_size:
        sampled_idx = rng.choice(sampled_idx, size=sample_size, replace=False)
    elif len(sampled_idx) < sample_size:
        needed = sample_size - len(sampled_idx)
        remaining = np.setdiff1d(np.arange(len(x)), sampled_idx, assume_unique=False)
        if len(remaining) > 0:
            extra_take = min(needed, len(remaining))
            extra = rng.choice(remaining, size=extra_take, replace=False)
            sampled_idx = np.concatenate([sampled_idx, extra.astype(int, copy=False)])

    rng.shuffle(sampled_idx)
    return x[sampled_idx], labels[sampled_idx]


def _safe_silhouette_score(
    x: np.ndarray,
    labels: np.ndarray,
    *,
    sample_size: int,
    random_state: int,
) -> float:
    unique = set(int(v) for v in labels.tolist())
    if len(unique) <= 1:
        return -1.0
    if unique == {-1}:
        return -1.0
    if len(labels) < 2:
        return -1.0
    x_eval, labels_eval = _sample_for_silhouette(
        x,
        labels,
        sample_size=sample_size,
        random_state=random_state,
    )
    unique_eval = set(int(v) for v in labels_eval.tolist())
    if len(unique_eval) <= 1 or unique_eval == {-1}:
        return -1.0
    try:
        return float(silhouette_score(x_eval, labels_eval))
    except Exception:
        return -1.0


def _plot_kmeans_tuning(results: list[dict[str, Any]], output_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    sorted_rows = sorted(results, key=lambda row: int(row["n_clusters"]))
    x = [int(row["n_clusters"]) for row in sorted_rows]
    y = [float(row["silhouette_score"]) for row in sorted_rows]
    fig = plt.figure(figsize=(8, 5))
    plt.plot(x, y, marker="o", alpha=0.6, label="all evaluated")
    coarse_x = [int(row["n_clusters"]) for row in sorted_rows if str(row.get("stage", "")) == "coarse"]
    coarse_y = [float(row["silhouette_score"]) for row in sorted_rows if str(row.get("stage", "")) == "coarse"]
    local_x = [int(row["n_clusters"]) for row in sorted_rows if str(row.get("stage", "")) == "local_refine"]
    local_y = [float(row["silhouette_score"]) for row in sorted_rows if str(row.get("stage", "")) == "local_refine"]
    if coarse_x:
        plt.scatter(coarse_x, coarse_y, marker="o", s=70, label="coarse")
    if local_x:
        plt.scatter(local_x, local_y, marker="s", s=70, label="local refine")
    plt.title("KMeans fine-tuning summary")
    plt.xlabel("n_clusters")
    plt.ylabel("silhouette_score")
    plt.grid(alpha=0.3)
    if coarse_x or local_x:
        plt.legend()
    plt.tight_layout()
    fig.savefig(output_dir / "clustering_tuning_kmeans.png", bbox_inches="tight")
    plt.close(fig)


def _plot_dbscan_tuning(results: list[dict[str, float]], output_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(results)
    if frame.empty:
        return
    pivot = frame.pivot(index="min_samples", columns="eps", values="silhouette_score").sort_index().sort_index(axis=1)
    fig = plt.figure(figsize=(9, 6))
    plt.imshow(pivot.to_numpy(), aspect="auto")
    plt.colorbar(label="silhouette_score")
    plt.title("DBSCAN fine-tuning summary")
    plt.xlabel("eps")
    plt.ylabel("min_samples")
    plt.xticks(range(len(pivot.columns)), [f"{float(v):.2f}" for v in pivot.columns], rotation=45, ha="right")
    plt.yticks(range(len(pivot.index)), [str(int(v)) for v in pivot.index])
    plt.tight_layout()
    fig.savefig(output_dir / "clustering_tuning_dbscan.png", bbox_inches="tight")
    plt.close(fig)


def _write_tuning_artifact(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _load_tuning_artifact(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    if not isinstance(loaded, dict):
        return None
    return loaded


def _tune_kmeans(
    x_train: np.ndarray,
    cfg: Mapping[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    start_time = time.perf_counter()
    budget_seconds = float(cfg["tuning_time_budget_seconds"])
    stopped_early = False
    candidates = [n_clusters for n_clusters in sorted({int(v) for v in cfg["kmeans_tuning_n_clusters"]}) if n_clusters < x_train.shape[0]]

    def _evaluate(n_clusters: int, stage: str) -> dict[str, Any]:
        with threadpool_limits(limits=int(cfg["parallel_inner_threads"])):
            model = KMeans(
                n_clusters=n_clusters,
                random_state=int(cfg["kmeans_random_state"]),
                n_init=int(cfg["kmeans_n_init"]),
            )
            labels = model.fit_predict(x_train)
        score = _safe_silhouette_score(
            x_train,
            labels,
            sample_size=int(cfg["tuning_score_sample_size"]),
            random_state=int(cfg["tuning_score_random_state"]),
        )
        return {"n_clusters": float(n_clusters), "silhouette_score": float(score), "stage": stage}

    if len(candidates) <= 4:
        coarse_candidates = list(candidates)
    else:
        coarse_idx = sorted({0, len(candidates) // 2, len(candidates) - 1})
        coarse_candidates = [candidates[idx] for idx in coarse_idx]

    coarse_results: list[dict[str, Any]] = []
    if coarse_candidates:
        elapsed_before_coarse = time.perf_counter() - start_time
        if elapsed_before_coarse >= budget_seconds:
            stopped_early = True
        else:
            coarse_results = Parallel(n_jobs=int(cfg["parallel_n_jobs"]), backend=str(cfg["parallel_backend"]))(
                delayed(_evaluate)(n_clusters, "coarse") for n_clusters in coarse_candidates
            )

    best_coarse: dict[str, Any] | None = None
    for row in coarse_results:
        score = float(row["silhouette_score"])
        if best_coarse is None or score > float(best_coarse["silhouette_score"]):
            best_coarse = {"n_clusters": int(row["n_clusters"]), "silhouette_score": score}

    neighborhood_radius = 2 if len(candidates) >= 7 else 1
    best_coarse_n_clusters = int((best_coarse or {}).get("n_clusters", int(cfg["kmeans_n_clusters"])))
    local_candidates = [
        n_clusters
        for n_clusters in candidates
        if abs(n_clusters - best_coarse_n_clusters) <= neighborhood_radius and n_clusters not in set(coarse_candidates)
    ]
    local_results: list[dict[str, Any]] = []
    if local_candidates:
        elapsed_before_local = time.perf_counter() - start_time
        if elapsed_before_local >= budget_seconds:
            stopped_early = True
        else:
            local_results = Parallel(n_jobs=int(cfg["parallel_n_jobs"]), backend=str(cfg["parallel_backend"]))(
                delayed(_evaluate)(n_clusters, "local_refine") for n_clusters in local_candidates
            )

    result_rows_by_cluster: dict[int, dict[str, Any]] = {}
    for row in [*coarse_results, *local_results]:
        key = int(row["n_clusters"])
        prior = result_rows_by_cluster.get(key)
        if prior is None or float(row["silhouette_score"]) >= float(prior["silhouette_score"]):
            result_rows_by_cluster[key] = dict(row)

    results = sorted(result_rows_by_cluster.values(), key=lambda row: int(row["n_clusters"]))
    best: dict[str, Any] | None = None
    for row in results:
        score = float(row["silhouette_score"])
        if best is None or score > float(best["silhouette_score"]):
            best = {"n_clusters": int(row["n_clusters"]), "silhouette_score": score}
    if best is None:
        best = {"n_clusters": int(cfg["kmeans_n_clusters"]), "silhouette_score": -1.0}
    stage_details: dict[str, Any] = {
        "coarse_candidates": coarse_candidates,
        "best_coarse_n_clusters": int(best_coarse_n_clusters),
        "local_neighborhood_radius": int(neighborhood_radius),
        "local_candidates": local_candidates,
        "stopped_early": bool(stopped_early),
        "elapsed_seconds": float(time.perf_counter() - start_time),
        "time_budget_seconds": float(budget_seconds),
    }
    return best, results, stage_details


def _tune_dbscan(
    x_all: np.ndarray,
    x_train: np.ndarray,
    ordered_train_mask: np.ndarray,
    cfg: Mapping[str, Any],
) -> tuple[dict[str, Any], list[dict[str, float]], dict[str, Any]]:
    start_time = time.perf_counter()
    budget_seconds = float(cfg["tuning_time_budget_seconds"])
    stopped_early = False
    best: dict[str, Any] | None = None
    eps_grid = sorted({float(v) for v in cfg["dbscan_tuning_eps"]})
    min_samples_grid = sorted({int(v) for v in cfg["dbscan_tuning_min_samples"]})
    candidates = [(eps, min_samples) for eps in eps_grid for min_samples in min_samples_grid]

    def _subsample(
        sample_size: int,
        random_state_offset: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if sample_size <= 0 or x_all.shape[0] <= sample_size:
            return x_all, x_train, ordered_train_mask
        rng = np.random.default_rng(seed=int(cfg["tuning_score_random_state"]) + random_state_offset)
        sampled_idx = np.sort(rng.choice(np.arange(x_all.shape[0]), size=sample_size, replace=False))
        x_sub_all = x_all[sampled_idx]
        mask_sub = ordered_train_mask[sampled_idx]
        x_sub_train = x_sub_all[mask_sub]
        if x_sub_train.shape[0] == 0:
            return x_sub_all, x_sub_all, np.ones(shape=(x_sub_all.shape[0],), dtype=bool)
        return x_sub_all, x_sub_train, mask_sub

    def _evaluate(
        eval_x_all: np.ndarray,
        eval_x_train: np.ndarray,
        eval_train_mask: np.ndarray,
        eps: float,
        min_samples: int,
    ) -> dict[str, float]:
        with threadpool_limits(limits=int(cfg["parallel_inner_threads"])):
            labels = _assign_dbscan_labels(
                eval_x_all,
                eval_x_train,
                eval_train_mask,
                eps=eps,
                min_samples=min_samples,
                n_jobs=1,
            )
        score = _safe_silhouette_score(
            eval_x_all,
            labels,
            sample_size=int(cfg["tuning_score_sample_size"]),
            random_state=int(cfg["tuning_score_random_state"]),
        )
        return {"eps": eps, "min_samples": float(min_samples), "silhouette_score": float(score)}

    stage1_results: list[dict[str, float]] = []
    survivors: list[tuple[float, int]] = []
    if candidates:
        elapsed_before_stage1 = time.perf_counter() - start_time
        if elapsed_before_stage1 >= budget_seconds:
            stopped_early = True
        else:
            stage1_x_all, stage1_x_train, stage1_train_mask = _subsample(int(cfg["dbscan_stage1_sample_size"]), 101)
            stage1_results = Parallel(n_jobs=int(cfg["parallel_n_jobs"]), backend=str(cfg["parallel_backend"]))(
                delayed(_evaluate)(stage1_x_all, stage1_x_train, stage1_train_mask, eps, min_samples) for eps, min_samples in candidates
            )

    if stage1_results:
        stage1_sorted = sorted(stage1_results, key=lambda row: float(row["silhouette_score"]), reverse=True)
        top_n = min(int(cfg["dbscan_top_n"]), len(stage1_sorted))
        survivors = [(float(row["eps"]), int(row["min_samples"])) for row in stage1_sorted[:top_n]]

    stage2_results: list[dict[str, float]] = []
    if survivors:
        elapsed_before_stage2 = time.perf_counter() - start_time
        if elapsed_before_stage2 >= budget_seconds:
            stopped_early = True
        else:
            stage2_x_all, stage2_x_train, stage2_train_mask = _subsample(int(cfg["dbscan_stage2_sample_size"]), 202)
            stage2_results = Parallel(n_jobs=int(cfg["parallel_n_jobs"]), backend=str(cfg["parallel_backend"]))(
                delayed(_evaluate)(stage2_x_all, stage2_x_train, stage2_train_mask, eps, min_samples) for eps, min_samples in survivors
            )

    final_results: list[dict[str, float]] = []
    run_final_pass = int(cfg["dbscan_stage2_sample_size"]) < int(x_all.shape[0])
    if run_final_pass and survivors:
        elapsed_before_final = time.perf_counter() - start_time
        if elapsed_before_final >= budget_seconds:
            stopped_early = True
        else:
            final_results = Parallel(n_jobs=int(cfg["parallel_n_jobs"]), backend=str(cfg["parallel_backend"]))(
                delayed(_evaluate)(x_all, x_train, ordered_train_mask, eps, min_samples) for eps, min_samples in survivors
            )

    best_rows = final_results if final_results else stage2_results
    for row in best_rows:
        score = float(row["silhouette_score"])
        if best is None or score > float(best["silhouette_score"]):
            best = {"eps": float(row["eps"]), "min_samples": int(row["min_samples"]), "silhouette_score": score}
    if best is None:
        best = {"eps": float(cfg["dbscan_eps"]), "min_samples": int(cfg["dbscan_min_samples"]), "silhouette_score": -1.0}

    aggregated: dict[tuple[float, int], dict[str, float]] = {}
    for rows in (stage1_results, stage2_results, final_results):
        for row in rows:
            key = (float(row["eps"]), int(row["min_samples"]))
            aggregated[key] = {"eps": key[0], "min_samples": float(key[1]), "silhouette_score": float(row["silhouette_score"])}
    merged_results = list(aggregated.values())

    staged_results: dict[str, Any] = {
        "stage1": stage1_results,
        "stage2": stage2_results,
        "final": final_results,
        "survivors": [{"eps": eps, "min_samples": min_samples} for eps, min_samples in survivors],
        "stopped_early": bool(stopped_early),
        "elapsed_seconds": float(time.perf_counter() - start_time),
        "time_budget_seconds": float(budget_seconds),
    }
    return best, merged_results, staged_results


def run(df_or_path: pd.DataFrame, config: Mapping[str, Any] | None = None) -> pd.DataFrame:
    if not isinstance(df_or_path, pd.DataFrame):
        raise TypeError("06c_build_features_clustering.run expects a pandas DataFrame input")

    cfg = _normalize_config(config)
    out = df_or_path.copy(deep=True)
    tuning_profile = _resolve_tuning_profile(cfg, row_count=len(out), user_config=config)
    tuned_cfg = _apply_tuning_profile(cfg, tuning_profile)

    source_cols = _select_source_columns(out, tuned_cfg["source_feature_prefixes"])
    if not source_cols:
        out["cluster_kmeans_id"] = -1
        out["cluster_dbscan_id"] = -1
        return out

    train_mask = _build_temporal_train_mask(out, train_fraction=tuned_cfg["train_fraction"], fit_scope=tuned_cfg["fit_scope"])

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

    method = tuned_cfg["method"]
    run_kmeans = method in {"kmeans", "both"}
    run_dbscan = method in {"dbscan", "both"}

    artifact_path = Path(str(tuned_cfg["tuning_artifact_path"]))
    plot_dir = Path(str(tuned_cfg["tuning_plot_dir"]))
    artifact = _load_tuning_artifact(artifact_path) if bool(tuned_cfg["auto_tune"]) else None
    if isinstance(artifact, dict) and artifact.get("tuning_profile") != tuning_profile:
        artifact = None
    artifact_kmeans = artifact.get("kmeans") if isinstance(artifact, dict) else None
    artifact_dbscan = artifact.get("dbscan") if isinstance(artifact, dict) else None
    kmeans_choice: dict[str, Any] | None = None
    dbscan_choice: dict[str, Any] | None = None
    kmeans_results: list[dict[str, Any]] = []
    kmeans_stage_details: dict[str, Any] | None = None
    dbscan_results: list[dict[str, float]] = []
    dbscan_staged_results: dict[str, Any] = {}
    kmeans_tuning_metadata: dict[str, Any] | None = None
    dbscan_tuning_metadata: dict[str, Any] | None = None

    if run_kmeans:
        if isinstance(artifact_kmeans, dict) and "n_clusters" in artifact_kmeans:
            kmeans_choice = {"n_clusters": int(artifact_kmeans["n_clusters"])}
            kmeans_results = [dict(v) for v in artifact.get("kmeans_results", [])] if isinstance(artifact, dict) else []
            if isinstance(artifact, dict) and isinstance(artifact.get("kmeans_tuning_stages"), dict):
                kmeans_stage_details = dict(artifact["kmeans_tuning_stages"])
            if isinstance(artifact, dict) and isinstance(artifact.get("kmeans_tuning_metadata"), dict):
                kmeans_tuning_metadata = dict(artifact["kmeans_tuning_metadata"])
        elif bool(tuned_cfg["auto_tune"]):
            kmeans_choice, kmeans_results, kmeans_stage_details = _tune_kmeans(x_train, tuned_cfg)
            if isinstance(kmeans_stage_details, dict):
                kmeans_tuning_metadata = {
                    "stopped_early": bool(kmeans_stage_details.get("stopped_early", False)),
                    "elapsed_seconds": float(kmeans_stage_details.get("elapsed_seconds", 0.0)),
                    "time_budget_seconds": float(kmeans_stage_details.get("time_budget_seconds", tuned_cfg["tuning_time_budget_seconds"])),
                }
        else:
            kmeans_choice = {"n_clusters": int(tuned_cfg["kmeans_n_clusters"])}

    if run_dbscan:
        if isinstance(artifact_dbscan, dict) and {"eps", "min_samples"}.issubset(artifact_dbscan.keys()):
            dbscan_choice = {"eps": float(artifact_dbscan["eps"]), "min_samples": int(artifact_dbscan["min_samples"])}
            dbscan_results = [dict(v) for v in artifact.get("dbscan_results", [])] if isinstance(artifact, dict) else []
            dbscan_staged_results = (
                dict(artifact.get("dbscan_staged_results", {})) if isinstance(artifact, dict) else {}
            )
            if isinstance(artifact, dict) and isinstance(artifact.get("dbscan_tuning_metadata"), dict):
                dbscan_tuning_metadata = dict(artifact["dbscan_tuning_metadata"])
        elif bool(tuned_cfg["auto_tune"]):
            dbscan_choice, dbscan_results, dbscan_staged_results = _tune_dbscan(x_all, x_train, ordered_train_mask, tuned_cfg)
            if isinstance(dbscan_staged_results, dict):
                dbscan_tuning_metadata = {
                    "stopped_early": bool(dbscan_staged_results.get("stopped_early", False)),
                    "elapsed_seconds": float(dbscan_staged_results.get("elapsed_seconds", 0.0)),
                    "time_budget_seconds": float(dbscan_staged_results.get("time_budget_seconds", tuned_cfg["tuning_time_budget_seconds"])),
                }
        else:
            dbscan_choice = {"eps": float(tuned_cfg["dbscan_eps"]), "min_samples": int(tuned_cfg["dbscan_min_samples"])}

    if bool(tuned_cfg["auto_tune"]) and artifact is None:
        artifact_payload: dict[str, Any] = {
            "selected_source_columns": source_cols,
            "method": method,
            "fit_scope": tuned_cfg["fit_scope"],
            "tuning_profile": tuning_profile,
        }
        if kmeans_choice is not None:
            artifact_payload["kmeans"] = {"n_clusters": int(kmeans_choice["n_clusters"])}
            artifact_payload["kmeans_results"] = kmeans_results
            if kmeans_stage_details is not None:
                artifact_payload["kmeans_tuning_stages"] = kmeans_stage_details
            if kmeans_tuning_metadata is not None:
                artifact_payload["kmeans_tuning_metadata"] = kmeans_tuning_metadata
        if dbscan_choice is not None:
            artifact_payload["dbscan"] = {
                "eps": float(dbscan_choice["eps"]),
                "min_samples": int(dbscan_choice["min_samples"]),
            }
            artifact_payload["dbscan_results"] = dbscan_results
            artifact_payload["dbscan_staged_results"] = dbscan_staged_results
            if dbscan_tuning_metadata is not None:
                artifact_payload["dbscan_tuning_metadata"] = dbscan_tuning_metadata
        _write_tuning_artifact(artifact_path, artifact_payload)

    if run_kmeans and kmeans_results:
        _plot_kmeans_tuning(kmeans_results, plot_dir)
    if run_dbscan and dbscan_results:
        _plot_dbscan_tuning(dbscan_results, plot_dir)

    if run_kmeans:
        kmeans = KMeans(
            n_clusters=int((kmeans_choice or {}).get("n_clusters", tuned_cfg["kmeans_n_clusters"])),
            random_state=int(tuned_cfg["kmeans_random_state"]),
            n_init=int(tuned_cfg["kmeans_n_init"]),
        )
        kmeans.fit(x_train)
        ordered["cluster_kmeans_id"] = kmeans.predict(x_all).astype(int)

    if run_dbscan:
        ordered["cluster_dbscan_id"] = _assign_dbscan_labels(
            x_all,
            x_train,
            ordered_train_mask,
            eps=float((dbscan_choice or {}).get("eps", tuned_cfg["dbscan_eps"])),
            min_samples=int((dbscan_choice or {}).get("min_samples", tuned_cfg["dbscan_min_samples"])),
            n_jobs=int(tuned_cfg["parallel_n_jobs"]),
        )

    result = ordered.sort_values("__row_id", kind="mergesort").drop(columns=["__row_id"], errors="ignore")
    return result
