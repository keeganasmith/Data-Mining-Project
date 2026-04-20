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


PIPELINE_DEFAULTS: dict[str, dict[str, Any]] = {
    "06_build_features_temporal_elo": ELO_DEFAULTS,
    "06b_build_features_temporal_rolling": TEMPORAL_ROLLING_DEFAULTS,
    "experiments": DEFAULT_EXPERIMENT_CONFIG,
    # Includes optional `profile` (e.g. "fast") for low-cost training diagnostics.
    "model_training": DEFAULT_MODEL_TRAINING_CONFIG,
}
