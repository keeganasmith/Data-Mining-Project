import importlib.util
import io
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

import pandas as pd

from tennis_pipeline.experiments.model_training import (
    run_feature_set_training_experiment,
    run_model_training_experiments,
)


@unittest.skipUnless(
    importlib.util.find_spec("sklearn") is not None and importlib.util.find_spec("matplotlib") is not None,
    "requires optional dependencies: scikit-learn and matplotlib",
)
class ModelTrainingExperimentsTests(unittest.TestCase):
    def test_training_outputs_depth_curves_roc_and_summary_metrics(self) -> None:
        rows = []
        for i in range(80):
            rows.append(
                {
                    "event_id": f"e{i//4}",
                    "match_id": f"m{i}",
                    "match_date": f"2024-01-{(i % 28) + 1:02d}",
                    "match_seq": i,
                    "team1_player_id": f"p{i}",
                    "team2_player_id": f"q{i}",
                    "surface_context": "Clay" if i % 2 == 0 else "Hard",
                    "rank_diff": (i % 15) - 7,
                    "abs_rank_diff": abs((i % 15) - 7),
                    "elo_diff_team1": (i % 30) - 15,
                    "elo_prob_team1_pre": 0.5 + ((i % 10) - 5) / 25.0,
                    "team1_wins": 1 if i % 3 != 0 else 0,
                }
            )

        df = pd.DataFrame(rows)

        with tempfile.TemporaryDirectory() as tmpdir:
            manifest = run_model_training_experiments(
                df,
                output_dir=tmpdir,
                config={
                    "depth_values": [1, 2, 3],
                    "rf_n_estimators": 20,
                    "gbdt_n_estimators": 20,
                },
            )

            artifact_dir = Path(tmpdir) / "model_training"
            self.assertTrue((artifact_dir / "depth_accuracy_curves.csv").exists())
            self.assertTrue((artifact_dir / "model_summary_metrics.csv").exists())
            self.assertTrue((artifact_dir / "depth_accuracy_curves.png").exists())
            self.assertTrue((artifact_dir / "model_training_manifest.json").exists())

            for name in ("decision_tree", "random_forest", "gbdt"):
                self.assertTrue((artifact_dir / f"roc_curve__{name}.png").exists())

            self.assertIn("models", manifest)
            self.assertEqual(3, len(manifest["models"]))

    def test_debug_leakage_prints_suspicious_and_near_copy_features(self) -> None:
        rows = []
        for i in range(80):
            target = 1 if i % 3 != 0 else 0
            rows.append(
                {
                    "event_id": f"e{i//4}",
                    "match_id": f"m{i}",
                    "match_date": f"2024-01-{(i % 28) + 1:02d}",
                    "match_seq": i,
                    "team1_player_id": f"p{i}",
                    "team2_player_id": f"q{i}",
                    "surface_context": "Clay" if i % 2 == 0 else "Hard",
                    "rank_diff": (i % 15) - 7,
                    "winner_proxy_flag": target,
                    "team1_wins": target,
                }
            )

        df = pd.DataFrame(rows)
        stdout = io.StringIO()
        with tempfile.TemporaryDirectory() as tmpdir, redirect_stdout(stdout):
            run_model_training_experiments(
                df,
                output_dir=tmpdir,
                config={
                    "depth_values": [1],
                    "rf_n_estimators": 10,
                    "gbdt_n_estimators": 10,
                    "debug_leakage": True,
                },
            )

        output = stdout.getvalue()
        self.assertIn("[leakage-debug] suspicious feature names:", output)
        self.assertIn("winner_proxy_flag", output)
        self.assertIn("[leakage-debug] near-perfect target copies detected:", output)

    def test_feature_set_training_outputs_cross_run_pricing_metric_comparison(self) -> None:
        rows = []
        for i in range(120):
            rows.append(
                {
                    "event_id": f"e{i//4}",
                    "match_id": f"m{i}",
                    "match_date": f"2024-02-{(i % 28) + 1:02d}",
                    "match_seq": i,
                    "team1_player_id": f"p{i}",
                    "team2_player_id": f"q{i}",
                    "surface_context": "Clay" if i % 2 == 0 else "Hard",
                    "rank_diff": (i % 15) - 7,
                    "elo_diff_team1": (i % 30) - 15,
                    "anomaly_score": ((i % 11) - 5) / 10.0,
                    "team1_wins": 1 if i % 4 != 0 else 0,
                }
            )

        full_df = pd.DataFrame(rows)
        feature_set_tables = {
            "structured_only": full_df.drop(columns=["elo_diff_team1", "anomaly_score"]),
            "structured_plus_elo": full_df.drop(columns=["anomaly_score"]),
            "structured_plus_anomaly": full_df.drop(columns=["elo_diff_team1"]),
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            manifests = run_feature_set_training_experiment(
                feature_set_tables,
                output_dir=tmpdir,
                config={
                    "depth_values": [2],
                    "rf_n_estimators": 20,
                    "gbdt_n_estimators": 20,
                },
            )

            summary_dir = Path(tmpdir) / "model_training_feature_sets"
            summary_csv = summary_dir / "feature_set_pricing_metric_comparison.csv"
            self.assertTrue(summary_csv.exists())
            self.assertTrue((summary_dir / "feature_set_pricing_metric_comparison.png").exists())

            summary_df = pd.read_csv(summary_csv)
            self.assertTrue(
                {
                    "feature_set",
                    "model",
                    "test_log_loss",
                    "test_brier_score",
                    "test_ece_10_bins",
                }.issubset(set(summary_df.columns))
            )

            self.assertIn("structured_only", manifests)
            artifacts = manifests["structured_only"].get("artifacts", {})
            self.assertIn("feature_set_pricing_metric_comparison_csv", artifacts)
            self.assertIn("feature_set_pricing_metric_comparison_plot", artifacts)


if __name__ == "__main__":
    unittest.main()
