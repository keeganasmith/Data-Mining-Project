#!/bin/bash
#python -m tennis_pipeline.cli.run_pipeline   --input-path data/raw_data.joblib   --output-dir data  --use-elo   --use-temporal-features   --clustering-method kmeans --config-path parallel_config.json
python -m tennis_pipeline.cli.run_pipeline \
  --input-path data/raw_data.joblib \
  --use-elo \
  --clustering-method kmeans \
  --run-feature-set-experiment \
  --config-path parallel_config.json
