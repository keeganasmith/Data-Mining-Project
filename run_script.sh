#!/bin/bash
python -m tennis_pipeline.cli.run_pipeline   --input-path data/raw_data.joblib   --output-dir data  --use-elo   --use-temporal-features   --clustering-method kmeans --config-path parallel_config.json
