# Tennis Pipeline: End-to-End Usage Guide

This repository includes a modular, step-based data pipeline for creating a leakage-safe tennis modeling table and experiment artifacts.

The pipeline entrypoint is `src/tennis_pipeline/cli/run_pipeline.py`, which executes steps `01` through `07` in order (including optional temporal feature stages) and writes intermediate and final artifacts to disk.

## 1) How the pipeline works

The runner executes the following steps in a fixed order:

1. `01_load_raw`
2. `02_clean_schema`
3. `03_clean_values`
4. `04_split_roles`
5. `05_build_features_static`
6. `06_build_features_temporal_elo` (optional via `--use-elo`)
7. `06b_build_features_temporal_rolling` (optional via `--use-temporal-features`)
8. `07_finalize_model_table`

For each executed step, the pipeline:

- runs the step module,
- validates stage-level checks,
- writes an interim parquet artifact to `<output-dir>/interim/<step_name>.parquet`.

After step execution, it writes the final model table to:

- `<output-dir>/processed/model_table.parquet`

If experiments are enabled in config, it also materializes experiment-specific model-table subsets and a manifest under:

- `<output-dir>/processed/experiments/`

---

## 2) CLI arguments and what they do

Run the pipeline with:

```bash
python -m tennis_pipeline.cli.run_pipeline \
  --input-path <path-to-input-file> \
  --output-dir <output-root> \
  [--config-path <json-or-yaml-config>] \
  [--use-elo] \
  [--use-temporal-features] \
  [--run-feature-set-experiment]
```

### Required argument

- `--input-path`
  - Path to the raw input file.
  - Supported formats: `.csv`, `.txt`, `.parquet`, `.pq`, `.joblib`.

### Optional arguments

- `--output-dir` (default: `data`)
  - Base output directory.
  - The runner creates:
    - `<output-dir>/interim/` (step-level parquet snapshots)
    - `<output-dir>/processed/` (final + experiment outputs)

- `--config-path` (default: none)
  - Optional JSON/YAML config file.
  - Merged with internal defaults using `step_name -> config` keys.

- `--use-elo` (flag)
  - Enables step `06_build_features_temporal_elo`.
  - If omitted, Elo columns are still present via safe defaults:
    - `elo_diff_team1 = 0.0`
    - `elo_team1_pre = 1500.0`
    - `elo_team2_pre = 1500.0`
    - `elo_prob_team1_pre = 0.5`


- `--use-temporal-features` (flag)
  - Enables step `06b_build_features_temporal_rolling`.
  - Adds leakage-safe pre-match rolling player-form features (win pct, avg Elo, paired-stat rolling means).

- `--run-feature-set-experiment` (flag)
  - Runs fixed-hyperparameter training across two feature-set variants:
    - structured_only
    - structured_plus_elo
  - This flag does not require anomaly features.

---

## 3) Step-by-step behavior (what each stage does)

## Step 01 — `01_load_raw`

Purpose:

- Reads a raw table from a path (or accepts an already loaded DataFrame).
- Applies parse-level normalization:
  - trim column names,
  - remove `Unnamed:` columns.

Config keys:

- `read_csv_kwargs` (mapping)
- `trim_column_names` (bool)
- `drop_unnamed_columns` (bool)

## Step 02 — `02_clean_schema`

Purpose:

- Canonicalizes schema with repository data contracts.
- Applies optional alias renaming.
- Optionally enforces required columns.

Config keys:

- `column_aliases` (mapping)
- `required_columns` (list/tuple/set or null)
- `enforce_required_schema` (bool)

## Step 03 — `03_clean_values`

Purpose:

- Coerces date/numeric/string dtypes.
- Drops missing required rows.
- Removes duplicate rows, including canonical Team1/Team2 pair duplicates.
- Applies invalid-row filters (winner/team consistency and rank sanity).

Key default behaviors include:

- required non-null IDs and date checks,
- dropping rows where `team1_player_id == team2_player_id`,
- dropping rows where winner is not one of the teams,
- dropping rows with non-positive ranks when rank columns exist.

## Step 04 — `04_split_roles`

Purpose:

- Deterministically aligns Team1/Team2 role assignment.
- Creates binary supervised target (`team1_wins`) from stable role assignment.

Important properties:

- Uses deterministic hash-based swap logic (seeded by `random_seed`).
- Idempotent (re-running yields same alignment).

Primary config keys:

- `random_seed`
- `winner_column`, `team1_id_column`, `team2_id_column`
- `target_column`
- duplicate-handling controls for final role duplicates

## Step 05 — `05_build_features_static`

Purpose:

- Builds paired Team1/Team2 engineered features.
- Adds canonical rank/race diff features:
  - `rank_diff`, `abs_rank_diff`
  - `race_rank_diff`, `abs_race_rank_diff`
- Normalizes surface/court context into:
  - `surface_context`
  - `court_context`
- Applies deterministic temporal ordering.

Config keys:

- `drop_missing_rank_diff` (bool)

## Step 06 — `06_build_features_temporal_elo` (optional)

Purpose:

- Builds leakage-safe, pre-match Elo features by chronological online updates.
- Appends columns (default prefix `elo`):
  - `elo_team1_pre`
  - `elo_team2_pre`
  - `elo_diff_team1` (canonical)
  - `elo_diff_pre` (legacy alias)
  - `elo_prob_team1_pre`

Important temporal guarantee:

- Features for a match are computed before applying that match outcome update.

Primary config keys:

- `initial_rating` (default 1500.0)
- `k_factor` (default 32.0)
- `rating_scale` (default 400.0)
- `feature_prefix` (default `elo`)
- `strict_validation` (default `True`)


## Step 06b — `06b_build_features_temporal_rolling` (optional)

Purpose:

- Adds leakage-safe pre-match rolling player-form features in a separate stage.
- Includes rolling win percentage, rolling average Elo, and rolling means for paired numeric `team1_*`/`team2_*` stats.
- Uses the same temporal ordering + pre-capture/post-update pattern to avoid leakage.

Config keys:

- `feature_prefix` (default `temporal`)
- `rolling_window_matches` (default `None`, i.e., all prior matches)
- `paired_stats_min_numeric_coverage` (default `0.8`)
- `include_elo_average`, `elo_team1_pre_column`, `elo_team2_pre_column`, `default_elo`

## Step 07 — `07_finalize_model_table`

Purpose:

- Selects leakage-safe final features.
- Standardizes final model-table column ordering.
- Ensures temporal sequence (`match_seq`) if absent.

Defaults include:

- target column: `team1_wins`
- ID columns: `event_id`, `match_id`, `match_date`, `match_seq`, `team1_player_id`, `team2_player_id`
- preferred ordering for rank, Elo, and context features

---

## 4) Running experiments (feature-set materialization)

The runner can automatically generate multiple experiment tables from the final model table.

> Note: anomaly detection was evaluated and then removed from the documented experiment set due to no measurable benefit.

By default (`experiments.enabled: true`), it writes:

- `<output-dir>/processed/experiments/model_table__structured_only.parquet`
- `<output-dir>/processed/experiments/model_table__structured_plus_elo.parquet`
- `<output-dir>/processed/experiments/feature_set_manifest.json`

Each output includes:

- metadata column (default: `feature_set_name`),
- configured ID columns,
- selected feature subset,
- target (`team1_wins`).

### Example: run pipeline + experiments

```bash
PYTHONPATH=src python -m tennis_pipeline.cli.run_pipeline \
  --input-path data/csv_data/atp_2024.csv \
  --output-dir data \
  --use-elo \
  --run-feature-set-experiment
```

### Example: custom experiment config

Create `pipeline_config.json`:

```json
{
  "experiments": {
    "enabled": true,
    "output_subdir": "experiments",
    "metadata_column": "feature_set_name",
    "target_column": "team1_wins",
    "feature_sets": [
      {"name": "structured_only", "include_elo": false},
      {"name": "structured_plus_elo", "include_elo": true}
    ]
  }
}
```

Run:

```bash
PYTHONPATH=src python -m tennis_pipeline.cli.run_pipeline \
  --input-path data/csv_data/atp_2024.csv \
  --output-dir data \
  --config-path pipeline_config.json \
  --use-elo \
  --run-feature-set-experiment
```

---


## 5) Model training analytics (tree models)

By default (`model_training.enabled: true`), the pipeline now trains:

- Decision Tree
- Random Forest
- Gradient Boosted Decision Trees (GBDT)

For each model, it trains with fixed defaults (no depth/hyperparameter sweep) and writes:

- `depth_accuracy_curves.csv` (training and validation accuracy by depth)
- `depth_accuracy_curves.png` (training vs validation accuracy curves)
- `model_summary_metrics.csv` (probability-quality metrics—test log loss, test Brier score, test ECE—plus ROC-AUC and accuracy diagnostics)
- `roc_curve__decision_tree.png`
- `roc_curve__random_forest.png`
- `roc_curve__gbdt.png`
- `match_probability_predictions.csv` (per-match probabilities and outcomes)
- `model_training_manifest.json`

`match_probability_predictions.csv` probability orientation:

- `prob_team1_victory` = model probability that Team1 wins (`team1_wins = 1`).
- `prob_team2_victory` = `1 - prob_team1_victory`.
- Team mapping comes from row ID columns:
  - Team1 ID: `team1_player_id`
  - Team2 ID: `team2_player_id`

When running feature-set experiments (`--run-feature-set-experiment`), the cross-run summary now defaults to **pricing-oriented probabilistic metrics**:

- `feature_set_pricing_metric_comparison.csv`
- `feature_set_pricing_metric_comparison.png`

This default is intentional for market pricing workflows. Book/price construction depends on reliable class probabilities (well-calibrated confidence), not just hard win/loss labels. A model can post similar or even better test accuracy while still producing worse implied probabilities; log loss, Brier score, and ECE directly measure this risk.

All artifacts are written under:

- `<output-dir>/processed/model_training/`

### Example: configure fixed training hyperparameters

```json
{
  "model_training": {
    "enabled": true,
    "depth_values": [8],
    "rf_n_estimators": 300,
    "dt_min_samples_leaf": 25,
    "rf_min_samples_leaf": 15,
    "gbdt_n_estimators": 300,
    "gbdt_learning_rate": 0.05,
    "gbdt_min_samples_leaf": 20,
    "gbdt_subsample": 0.8
  }
}
```

---

## 6) How to view resulting output

## A) Inspect final model table schema + sample rows

```bash
python - <<'PY'
import pandas as pd

df = pd.read_parquet("data/processed/model_table.parquet")
print("shape:", df.shape)
print("columns:", list(df.columns))
print(df.head(10))
PY
```

## B) Inspect per-step interim snapshots

```bash
python - <<'PY'
from pathlib import Path
import pandas as pd

interim_dir = Path("data/interim")
for path in sorted(interim_dir.glob("*.parquet")):
    df = pd.read_parquet(path)
    print(f"{path.name}: rows={len(df):,}, cols={len(df.columns):,}")
PY
```

## C) Inspect experiment manifest and tables

```bash
python - <<'PY'
import json
import pandas as pd
from pathlib import Path

manifest_path = Path("data/processed/experiments/feature_set_manifest.json")
manifest = json.loads(manifest_path.read_text())
print("feature sets:", [f["name"] for f in manifest["feature_sets"]])

for entry in manifest["feature_sets"]:
    p = Path(entry["output_path"])
    df = pd.read_parquet(p)
    print(entry["name"], df.shape)
PY
```


---

## 6) Practical run patterns

## Quick baseline run (structured_only)

```bash
PYTHONPATH=src python -m tennis_pipeline.cli.run_pipeline \
  --input-path data/csv_data/atp_2024.csv \
  --output-dir data
```

## Elo-only run

```bash
PYTHONPATH=src python -m tennis_pipeline.cli.run_pipeline \
  --input-path data/csv_data/atp_2024.csv \
  --output-dir data \
  --use-elo
```


## Reproducibility tips

- Keep `random_seed` fixed (step 04 options).
- Keep `k_factor`, `initial_rating`, and `rating_scale` fixed for Elo comparability.
- Keep experiment config and feature-set names stable between runs.

---

## 7) Where to find deeper implementation details

- Step-to-notebook mapping: `docs/pipeline_mapping.md`
- Pipeline runner and CLI flags: `src/tennis_pipeline/cli/run_pipeline.py`
- Default pipeline and Elo/experiment configs: `src/tennis_pipeline/config.py`
- Experiment feature-set materialization logic: `src/tennis_pipeline/experiments/feature_sets.py`
- Stage implementations: `src/tennis_pipeline/steps/`
