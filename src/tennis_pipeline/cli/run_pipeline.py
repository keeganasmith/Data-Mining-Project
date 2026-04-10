"""CLI for executing the modular tennis preprocessing pipeline end-to-end."""

from __future__ import annotations

import argparse
import importlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pandas as pd

from tennis_pipeline.config import PIPELINE_DEFAULTS
from tennis_pipeline.data_contracts import (
    CLEANED_INTERIM_CONTRACT,
    FINAL_MODEL_CONTRACT,
    RAW_INPUT_CONTRACT,
    validate_required_columns,
    validate_table,
)

STEP_MODULES: tuple[str, ...] = (
    "01_load_raw",
    "02_clean_schema",
    "03_clean_values",
    "04_split_roles",
    "05_build_features_static",
    "06_build_features_temporal_elo",
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


def _validate_stage(df: pd.DataFrame, step_name: str) -> None:
    """Run stage-appropriate validation checks after each step."""

    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"[{step_name}] Step output must be a pandas DataFrame")
    if df.columns.duplicated().any():
        dupes = df.columns[df.columns.duplicated()].tolist()
        raise ValueError(f"[{step_name}] Output has duplicate columns: {dupes}")

    if step_name == "01_load_raw":
        validate_required_columns(df, RAW_INPUT_CONTRACT)
    elif step_name == "02_clean_schema":
        required_canonical = [
            RAW_INPUT_CONTRACT.canonical_column_names.get(col, col)
            for col in RAW_INPUT_CONTRACT.required_columns
        ]
        missing = [col for col in required_canonical if col not in df.columns]
        if missing:
            raise ValueError(f"[{step_name}] Missing canonical required columns: {missing}")
    elif step_name in {"03_clean_values", "04_split_roles"}:
        required = [
            "match_id",
            "match_date",
            "winner_player_id",
            "team1_player_id",
            "team2_player_id",
            "team1_sgl_roll_rank",
            "team2_sgl_roll_rank",
        ]
        missing = [col for col in required if col not in df.columns]
        if missing:
            raise ValueError(f"[{step_name}] Missing required cleaned columns: {missing}")
    elif step_name in {"05_build_features_static", "06_build_features_temporal_elo"}:
        validate_table(df, CLEANED_INTERIM_CONTRACT)
    elif step_name == "07_finalize_model_table":
        validate_table(df, FINAL_MODEL_CONTRACT)


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
    config_path: str | Path | None = None,
) -> pd.DataFrame:
    """Execute all pipeline stages and persist stage/final artifacts."""

    cfg = _load_config(config_path)
    out_root = Path(output_dir)
    interim_dir = out_root / "interim"
    processed_dir = out_root / "processed"
    interim_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)

    current: pd.DataFrame | str | Path = Path(input_path)

    for step_name in STEP_MODULES:
        if step_name == "06_build_features_temporal_elo" and not use_elo:
            if not isinstance(current, pd.DataFrame):
                raise TypeError("Pipeline state must be DataFrame before Elo toggle branch")
            current = _ensure_elo_feature_when_disabled(current)
            _validate_stage(current, step_name)
            current.to_parquet(interim_dir / f"{step_name}.parquet", index=False)
            continue

        module = importlib.import_module(f"tennis_pipeline.steps.{step_name}")
        step_config = cfg.get(step_name)
        current = module.run(current, config=step_config)

        if not isinstance(current, pd.DataFrame):
            raise TypeError(f"[{step_name}] Expected DataFrame output from step module")

        _validate_stage(current, step_name)
        current.to_parquet(interim_dir / f"{step_name}.parquet", index=False)

    final_df = current
    if not isinstance(final_df, pd.DataFrame):
        raise TypeError("Final pipeline output is not a DataFrame")

    final_path = processed_dir / "model_table.parquet"
    final_df.to_parquet(final_path, index=False)
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
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    run_pipeline(
        input_path=args.input_path,
        output_dir=args.output_dir,
        use_elo=bool(args.use_elo),
        config_path=args.config_path,
    )


if __name__ == "__main__":
    main()
