"""Config-driven feature-set experiments for model-table materialization.

This module produces stable feature subset artifacts for common experiment
variants so downstream evaluation (ROC-AUC, upset metrics) can key off a fixed
feature-set identifier.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pandas as pd

DEFAULT_EXPERIMENT_CONFIG: dict[str, Any] = {
    "enabled": True,
    "output_subdir": "experiments",
    "manifest_filename": "feature_set_manifest.json",
    "metadata_column": "feature_set_name",
    "target_column": "team1_wins",
    "id_columns": ["event_id", "match_id", "match_date", "match_seq", "team1_player_id", "team2_player_id"],
    "elo_feature_prefixes": ["elo_"],
    "elo_feature_columns": [],
    "temporal_feature_prefixes": ["temporal_"],
    "temporal_feature_columns": [],
    "clustering_feature_prefixes": ["cluster_"],
    "clustering_feature_columns": [],
    "feature_sets": [
        {
            "name": "data_only",
            "include_elo": False,
            "include_temporal": False,
            "include_clustering": False,
        },
        {
            "name": "data_plus_temporal_elo_clustering",
            "include_elo": True,
            "include_temporal": True,
            "include_clustering": True,
        },
    ],
}


def _normalize_config(config: Mapping[str, Any] | None) -> dict[str, Any]:
    merged = dict(DEFAULT_EXPERIMENT_CONFIG)
    if config:
        merged.update(dict(config))

    string_keys = ("output_subdir", "manifest_filename", "metadata_column", "target_column")
    for key in string_keys:
        value = merged.get(key)
        if not isinstance(value, str) or not value.strip():
            raise TypeError(f"experiments config['{key}'] must be a non-empty string")

    list_keys = (
        "id_columns",
        "elo_feature_prefixes",
        "elo_feature_columns",
        "temporal_feature_prefixes",
        "temporal_feature_columns",
        "clustering_feature_prefixes",
        "clustering_feature_columns",
        "feature_sets",
    )
    for key in list_keys:
        value = merged.get(key)
        if not isinstance(value, list):
            raise TypeError(f"experiments config['{key}'] must be a list")

    if not isinstance(merged.get("enabled"), bool):
        raise TypeError("experiments config['enabled'] must be a bool")

    validated_sets: list[dict[str, Any]] = []
    for entry in merged["feature_sets"]:
        if not isinstance(entry, Mapping):
            raise TypeError("Each experiments config feature set must be a mapping")
        name = entry.get("name")
        include_elo = entry.get("include_elo")
        include_temporal = entry.get("include_temporal", False)
        include_clustering = entry.get("include_clustering", False)
        if not isinstance(name, str) or not name.strip():
            raise TypeError("Each feature set requires non-empty 'name'")
        if not isinstance(include_elo, bool):
            raise TypeError("Each feature set requires bool 'include_elo'")
        if not isinstance(include_temporal, bool):
            raise TypeError("Each feature set requires bool 'include_temporal'")
        if not isinstance(include_clustering, bool):
            raise TypeError("Each feature set requires bool 'include_clustering'")
        validated_sets.append(
            {
                "name": name.strip(),
                "include_elo": include_elo,
                "include_temporal": include_temporal,
                "include_clustering": include_clustering,
            }
        )

    merged["feature_sets"] = validated_sets
    return merged


def _collect_feature_columns(df: pd.DataFrame, cfg: Mapping[str, Any]) -> tuple[list[str], list[str], list[str], list[str]]:
    target_column = cfg["target_column"]
    id_columns = set(cfg["id_columns"])

    base_features = [c for c in df.columns if c not in id_columns and c != target_column]

    elo_prefixes = tuple(cfg["elo_feature_prefixes"])
    elo_exact = set(cfg["elo_feature_columns"])
    temporal_prefixes = tuple(cfg["temporal_feature_prefixes"])
    temporal_exact = set(cfg["temporal_feature_columns"])
    clustering_prefixes = tuple(cfg["clustering_feature_prefixes"])
    clustering_exact = set(cfg["clustering_feature_columns"])

    elo_features = [c for c in base_features if c in elo_exact or c.startswith(elo_prefixes)]
    temporal_features = [c for c in base_features if c in temporal_exact or c.startswith(temporal_prefixes)]
    clustering_features = [c for c in base_features if c in clustering_exact or c.startswith(clustering_prefixes)]
    engineered = set(elo_features) | set(temporal_features) | set(clustering_features)
    structured_features = [c for c in base_features if c not in engineered]
    return structured_features, elo_features, temporal_features, clustering_features


def materialize_feature_sets(
    df: pd.DataFrame,
    *,
    output_dir: str | Path,
    config: Mapping[str, Any] | None = None,
) -> dict[str, pd.DataFrame]:
    """Write experiment-specific model tables and a manifest.

    Returns a mapping of ``feature_set_name -> dataframe`` for in-process reuse.
    """

    if not isinstance(df, pd.DataFrame):
        raise TypeError("materialize_feature_sets expects a pandas DataFrame")

    cfg = _normalize_config(config)
    if not cfg["enabled"]:
        return {}

    target_column = cfg["target_column"]
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' is required for feature-set materialization")

    id_cols = [c for c in cfg["id_columns"] if c in df.columns]
    structured, elo, temporal, clustering = _collect_feature_columns(df, cfg)

    experiment_dir = Path(output_dir) / cfg["output_subdir"]
    experiment_dir.mkdir(parents=True, exist_ok=True)

    metadata_column = cfg["metadata_column"]
    artifacts: dict[str, pd.DataFrame] = {}
    manifest_sets: list[dict[str, Any]] = []

    for feature_set in cfg["feature_sets"]:
        name = feature_set["name"]

        selected = list(structured)
        if feature_set["include_elo"]:
            selected.extend(elo)
        if feature_set["include_temporal"]:
            selected.extend(temporal)
        if feature_set["include_clustering"]:
            selected.extend(clustering)

        ordered_features = [c for c in df.columns if c in set(selected)]
        out_columns = [*id_cols, *ordered_features, target_column]

        subset_df = df.loc[:, out_columns].copy(deep=True)
        subset_df.insert(0, metadata_column, name)

        output_path = experiment_dir / f"model_table__{name}.parquet"
        subset_df.to_parquet(output_path, index=False)

        artifacts[name] = subset_df
        manifest_sets.append(
            {
                "name": name,
                "output_path": str(output_path),
                "row_count": int(len(subset_df)),
                "feature_count": int(len(ordered_features)),
                "feature_columns": ordered_features,
                "include_elo": feature_set["include_elo"],
                "include_temporal": feature_set["include_temporal"],
                "include_clustering": feature_set["include_clustering"],
            }
        )

    manifest_payload = {
        "metadata_column": metadata_column,
        "target_column": target_column,
        "id_columns": id_cols,
        "structured_feature_columns": structured,
        "elo_feature_columns": elo,
        "temporal_feature_columns": temporal,
        "clustering_feature_columns": clustering,
        "feature_sets": manifest_sets,
    }
    manifest_path = experiment_dir / cfg["manifest_filename"]
    manifest_path.write_text(json.dumps(manifest_payload, indent=2), encoding="utf-8")

    return artifacts
