"""Pipeline configuration defaults.

This module centralizes default knobs used by step modules so they can be
overridden by callers while keeping a single documented source of truth.
"""

from __future__ import annotations

from typing import Any

from tennis_pipeline.experiments.feature_sets import DEFAULT_EXPERIMENT_CONFIG
from tennis_pipeline.experiments.model_training import DEFAULT_MODEL_TRAINING_CONFIG

# Defaults for Step 06 (temporal Elo features)
ELO_DEFAULTS: dict[str, Any] = {
    # Base rating assigned to players with no prior match history.
    "initial_rating": 1500.0,
    # Constant K-factor used in the Elo update equation.
    "k_factor": 32.0,
    # Logistic scale denominator used in expected-score calculation.
    # Standard Elo uses 400.
    "rating_scale": 400.0,
    # Prefix used for generated Elo columns.
    "feature_prefix": "elo",
    # Whether to enforce required columns before computing Elo features.
    "strict_validation": True,
}

# Defaults for Step 06b (temporal rolling/player-form features)
TEMPORAL_ROLLING_DEFAULTS: dict[str, Any] = {
    # Prefix used for generated temporal rolling columns.
    "feature_prefix": "temporal",
    # Whether to enforce required columns before computing rolling features.
    "strict_validation": True,
    # Optional trailing match window for rolling means; None uses all prior matches.
    "rolling_window_matches": None,
    # Try to include Elo averages if Elo pre-match columns are present.
    "include_elo_average": True,
    # Elo columns to read from when include_elo_average=True.
    "elo_team1_pre_column": "elo_team1_pre",
    "elo_team2_pre_column": "elo_team2_pre",
    # Fallback Elo used when Elo columns are absent.
    "default_elo": 1500.0,
    # Minimum numeric coverage threshold for auto-discovered team1_/team2_ paired stats.
    "paired_stats_min_numeric_coverage": 0.8,
}

# Defaults for Step 06c (optional leakage-safe clustering features)
CLUSTERING_DEFAULTS: dict[str, Any] = {
    # Which clustering algorithm(s) to run: "kmeans", "dbscan", or "both".
    "method": "kmeans",
    # Prefixes that define eligible leakage-safe source columns.
    "source_feature_prefixes": ["elo_", "temporal_"],
    # Fraction of temporally earliest rows used to fit clustering models when fit_scope="train_only".
    "train_fraction": 0.8,
    # Leakage policy: "train_only" fits on earliest training slice only; "all_data" fits on full table (not leakage-safe).
    "fit_scope": "train_only",
    # Auto-tune clustering hyperparameters to maximize silhouette score.
    "auto_tune": True,
    # Tuning profile selector: "full" for exhaustive-ish search, "fast" for compact search grids.
    "tuning_profile": "full",
    # Auto-switch to "fast" profile when row count is high, unless tuning_profile is explicitly set by caller.
    "fast_mode_row_threshold": 50_000,
    # Optional artifact path used to cache best hyperparameters and tuning results.
    "tuning_artifact_path": "data/processed/clustering_tuning_artifact.json",
    # Optional output directory for tuning summary plots.
    "tuning_plot_dir": "data/processed",
    # Parallel worker count used by clustering tuning/search helpers.
    # -1 means use all available cores.
    "parallel_n_jobs": -1,
    # joblib backend for tuning parallelism ("loky" = processes, "threading" = threads).
    "parallel_backend": "loky",
    # Limit native BLAS/OpenMP threads inside each parallel worker to avoid oversubscription.
    "parallel_inner_threads": 1,
    # KMeans knobs.
    "kmeans_n_clusters": 8,
    "kmeans_random_state": 42,
    "kmeans_n_init": 10,
    # KMeans tuning search space.
    "kmeans_tuning_n_clusters": [2, 3, 4, 5, 6, 7, 8, 9, 10],
    # KMeans compact tuning search space for fast profile.
    "kmeans_tuning_n_clusters_fast": [4, 6, 8, 10],
    # DBSCAN knobs.
    "dbscan_eps": 0.9,
    "dbscan_min_samples": 25,
    # DBSCAN tuning search space.
    "dbscan_tuning_eps": [0.3, 0.5, 0.7, 0.9, 1.1, 1.3, 1.5],
    "dbscan_tuning_min_samples": [5, 10, 15, 20, 25, 30],
    # DBSCAN compact tuning search space for fast profile (~9 combos total).
    "dbscan_tuning_eps_fast": [0.6, 0.9, 1.2],
    "dbscan_tuning_min_samples_fast": [10, 20, 30],
}


PIPELINE_DEFAULTS: dict[str, dict[str, Any]] = {
    "06_build_features_temporal_elo": ELO_DEFAULTS,
    "06b_build_features_temporal_rolling": TEMPORAL_ROLLING_DEFAULTS,
    "06c_build_features_clustering": CLUSTERING_DEFAULTS,
    "experiments": DEFAULT_EXPERIMENT_CONFIG,
    # Includes optional `profile` (e.g. "fast") for low-cost training diagnostics.
    "model_training": DEFAULT_MODEL_TRAINING_CONFIG,
}
