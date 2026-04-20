"""CLI for executing the modular tennis preprocessing pipeline end-to-end."""

from __future__ import annotations

import argparse
import importlib
import json
import time
from collections.abc import Mapping
from datetime import timezone, datetime
from pathlib import Path
from typing import Any

import pandas as pd
import sys
import os

project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from tennis_pipeline.config import PIPELINE_DEFAULTS
from tennis_pipeline.experiments.feature_sets import materialize_feature_sets
from tennis_pipeline.experiments.model_training import (
    MODEL_TRAINING_PROFILES,
    run_feature_set_training_experiment,
    run_model_training_experiments,
)
from tennis_pipeline.validation.checks import run_stage_checks

STEP_MODULES: tuple[str, ...] = (
    "01_load_raw",
    "02_clean_schema",
    "03_clean_values",
    "04_split_roles",
    "05_build_features_static",
    "06_build_features_temporal_elo",
    "06b_build_features_temporal_rolling",
    "07_finalize_model_table",
)


def _load_config(config_path: str | Path | None) -> dict[str, dict[str, Any]]:
    """Load optional pipeline config file and merge with defaults."""

    config: dict[str, dict[str, Any]] = {
        step_name: dict(values) for step_name, values in PIPELINE_DEFAULTS.items()
    }

    if config_path is None:
        return config

    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    suffix = path.suffix.lower()
    if suffix == ".json":
        with path.open("r", encoding="utf-8") as handle:
            loaded = json.load(handle)
    elif suffix in {".yaml", ".yml"}:
        try:
            import yaml  # type: ignore[import-not-found]
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("PyYAML is required to load .yaml/.yml config files") from exc
        with path.open("r", encoding="utf-8") as handle:
            loaded = yaml.safe_load(handle)
    else:
        raise ValueError("Unsupported config format. Use .json, .yaml, or .yml")

    if loaded is None:
        return config
    if not isinstance(loaded, Mapping):
        raise TypeError("Pipeline config file must be a mapping of step_name -> config")

    for step_name, step_cfg in loaded.items():
        if not isinstance(step_name, str):
            raise TypeError("Pipeline config step names must be strings")
        if step_cfg is None:
            continue
        if not isinstance(step_cfg, Mapping):
            raise TypeError(f"Config for step '{step_name}' must be a mapping")
        current = config.get(step_name, {})
        config[step_name] = {**current, **dict(step_cfg)}

    return config


def _ensure_elo_feature_when_disabled(df: pd.DataFrame) -> pd.DataFrame:
    """Provide default Elo columns when --use-elo is not enabled."""

    out = df.copy(deep=True)
    if "elo_diff_team1" not in out.columns:
        out["elo_diff_team1"] = 0.0
    if "elo_team1_pre" not in out.columns:
        out["elo_team1_pre"] = 1500.0
    if "elo_team2_pre" not in out.columns:
        out["elo_team2_pre"] = 1500.0
    if "elo_prob_team1_pre" not in out.columns:
        out["elo_prob_team1_pre"] = 0.5
    return out


def run_pipeline(
    input_path: str | Path,
    output_dir: str | Path = "data",
    *,
    use_elo: bool = False,
    use_temporal_features: bool = False,
    run_feature_set_experiment: bool = False,
    config_path: str | Path | None = None,
    training_profile: str | None = None,
) -> pd.DataFrame:
    """Execute all pipeline stages and persist stage/final artifacts."""

    print("[pipeline] starting tennis pipeline run")
    if run_feature_set_experiment:
        print("[pipeline] --run-feature-set-experiment enabled (implies Elo stage)")

    cfg = _load_config(config_path)
    out_root = Path(output_dir)
    interim_dir = out_root / "interim"
    processed_dir = out_root / "processed"
    interim_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)

    current: pd.DataFrame | str | Path = Path(input_path)

    effective_use_elo = use_elo or run_feature_set_experiment
    effective_use_temporal_features = use_temporal_features

    for step_name in STEP_MODULES:
        print(f"[pipeline] step {step_name}: start")
        if step_name == "06_build_features_temporal_elo" and not effective_use_elo:
            if not isinstance(current, pd.DataFrame):
                raise TypeError("Pipeline state must be DataFrame before Elo toggle branch")
            current = _ensure_elo_feature_when_disabled(current)
            run_stage_checks(current, step_name)
            current.to_parquet(interim_dir / f"{step_name}.parquet", index=False)
            print(f"[pipeline] step {step_name}: skipped (Elo disabled); wrote defaults")
            continue

        if step_name == "06b_build_features_temporal_rolling" and not effective_use_temporal_features:
            if not isinstance(current, pd.DataFrame):
                raise TypeError("Pipeline state must be DataFrame before temporal rolling toggle branch")
            run_stage_checks(current, step_name)
            current.to_parquet(interim_dir / f"{step_name}.parquet", index=False)
            print(f"[pipeline] step {step_name}: skipped (temporal rolling disabled)")
            continue

        module = importlib.import_module(f"tennis_pipeline.steps.{step_name}")
        step_config = dict(cfg.get(step_name, {}))
        current = module.run(current, config=step_config)

        if not isinstance(current, pd.DataFrame):
            raise TypeError(f"[{step_name}] Expected DataFrame output from step module")

        run_stage_checks(current, step_name)
        current.to_parquet(interim_dir / f"{step_name}.parquet", index=False)
        print(f"[pipeline] step {step_name}: complete ({len(current):,} rows, {len(current.columns):,} columns)")

    final_df = current
    if not isinstance(final_df, pd.DataFrame):
        raise TypeError("Final pipeline output is not a DataFrame")

    final_path = processed_dir / "model_table.parquet"
    final_df.to_parquet(final_path, index=False)
    print(f"[pipeline] final model table written: {final_path}")

    experiment_config = cfg.get("experiments")
    feature_set_tables: dict[str, pd.DataFrame] = {}
    if isinstance(experiment_config, Mapping):
        print("[pipeline] materializing experiment feature-set tables")
        feature_set_tables = materialize_feature_sets(final_df, output_dir=processed_dir, config=experiment_config)
        print(f"[pipeline] materialized {len(feature_set_tables)} feature-set tables")

    model_training_config = cfg.get("model_training")
    if isinstance(model_training_config, Mapping):
        baseline_training_config = dict(model_training_config)
        if training_profile is not None:
            baseline_training_config["profile"] = training_profile
            print(f"[pipeline] overriding model training profile via CLI: {training_profile}")
        if run_feature_set_experiment:
            baseline_training_config["debug_leakage"] = True
            print("[pipeline] enabling debug leakage logs for training (feature-set mode)")
            expected_total_training_runs = 1 + len(feature_set_tables)
            print(
                "[pipeline] expected training runs with --run-feature-set-experiment: "
                f"baseline + {len(feature_set_tables)} feature sets = {expected_total_training_runs}"
            )
        else:
            expected_total_training_runs = 1

        print(f"[pipeline] running baseline model-training experiment (run 1 / {expected_total_training_runs})")
        baseline_started_at = datetime.now(timezone.utc)
        baseline_start_perf = time.perf_counter()
        print(f"[pipeline] baseline training start: {baseline_started_at.isoformat()}")
        run_model_training_experiments(final_df, output_dir=processed_dir, config=baseline_training_config)
        baseline_ended_at = datetime.now(timezone.utc)
        baseline_elapsed_seconds = time.perf_counter() - baseline_start_perf
        print(
            "[pipeline] baseline training end: "
            f"{baseline_ended_at.isoformat()} (elapsed {baseline_elapsed_seconds:.2f}s)"
        )
        if run_feature_set_experiment:
            if not feature_set_tables and isinstance(experiment_config, Mapping):
                print("[pipeline] feature sets missing; rematerializing before training")
                feature_set_tables = materialize_feature_sets(final_df, output_dir=processed_dir, config=experiment_config)
                expected_total_training_runs = 1 + len(feature_set_tables)
                print(
                    "[pipeline] updated expected training runs: "
                    f"baseline + {len(feature_set_tables)} feature sets = {expected_total_training_runs}"
                )
            feature_set_training_config = dict(model_training_config)
            if training_profile is not None:
                feature_set_training_config["profile"] = training_profile
            feature_set_training_config["debug_leakage"] = True
            print(f"[pipeline] running model training across {len(feature_set_tables)} feature sets")
            run_feature_set_training_experiment(
                feature_set_tables,
                output_dir=processed_dir,
                config=feature_set_training_config,
                start_run_index=2,
                total_runs=expected_total_training_runs,
            )
            print("[pipeline] feature-set training experiment complete")

    print("[pipeline] pipeline run complete")

    return final_df


def build_parser() -> argparse.ArgumentParser:
    """Build command-line argument parser for pipeline execution."""

    parser = argparse.ArgumentParser(description="Run the tennis feature pipeline end-to-end")
    parser.add_argument("--input-path", required=True, help="Path to raw input file (.csv/.txt/.parquet)")
    parser.add_argument(
        "--output-dir",
        default="data",
        help="Base output directory. Stage artifacts go to <output-dir>/interim and final output to <output-dir>/processed",
    )
    parser.add_argument(
        "--config-path",
        default=None,
        help="Optional JSON/YAML config with per-step settings",
    )
    parser.add_argument(
        "--use-elo",
        action="store_true",
        help="Enable temporal Elo feature stage (06_build_features_temporal_elo)",
    )
    parser.add_argument(
        "--use-temporal-features",
        action="store_true",
        help="Enable temporal rolling feature stage (06b_build_features_temporal_rolling)",
    )
    parser.add_argument(
        "--run-feature-set-experiment",
        action="store_true",
        help=(
            "Train fixed-hyperparameter models across feature sets that retain temporal pre-match "
            "skill signals. Implies --use-elo."
        ),
    )
    parser.add_argument(
        "--training-profile",
        default=None,
        choices=sorted(MODEL_TRAINING_PROFILES.keys()),
        help=(
            "Optional model-training profile override. Use 'fast' for lower estimator counts "
            "to run quick diagnostics."
        ),
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    run_pipeline(
        input_path=args.input_path,
        output_dir=args.output_dir,
        use_elo=bool(args.use_elo),
        use_temporal_features=bool(args.use_temporal_features),
        run_feature_set_experiment=bool(args.run_feature_set_experiment),
        config_path=args.config_path,
        training_profile=args.training_profile,
    )


if __name__ == "__main__":
    main()
