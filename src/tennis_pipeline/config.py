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


PIPELINE_DEFAULTS: dict[str, dict[str, Any]] = {
    "06_build_features_temporal_elo": ELO_DEFAULTS,
    "experiments": DEFAULT_EXPERIMENT_CONFIG,
    "model_training": DEFAULT_MODEL_TRAINING_CONFIG,
    "06b_build_features_anomaly_surface": {
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
        "contamination": 0.05,
        "random_state": 42,
        # When True, anomaly z-scores are normalized within each surface.
        "emit_surface_anomaly_z": True,
        "knn_neighbors": 10,
        "knn_reference_size": 5000,
        "knn_chunk_size": 2048,
        "artifact_output_dir": "./outputs/anomaly.txt",
        "artifact_top_n": 25,
    },
}
