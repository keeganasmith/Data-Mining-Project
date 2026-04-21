import importlib.util
import io
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

import pandas as pd

from tennis_pipeline.experiments.model_training import (
    _compute_market_pricing_evaluation,
    _validate_side_probabilities,
    run_feature_set_training_experiment,
    run_model_training_experiments,
)


class ProbabilityValidationTests(unittest.TestCase):
    def test_validate_side_probabilities_rejects_non_numeric_values(self) -> None:
        with self.assertRaisesRegex(ValueError, "non-numeric"):
            _validate_side_probabilities(
                prob_team1=pd.Series([0.2, "not-a-number"]),
                prob_team2=pd.Series([0.8, 0.2]),
            )

    def test_validate_side_probabilities_rejects_non_complementary_values(self) -> None:
        with self.assertRaisesRegex(ValueError, "do not sum to 1"):
            _validate_side_probabilities(
                prob_team1=pd.Series([0.55, 0.61]),
                prob_team2=pd.Series([0.45, 0.38]),
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
            self.assertTrue((artifact_dir / "match_probability_predictions.csv").exists())
            self.assertTrue((artifact_dir / "depth_accuracy_curves.png").exists())
            self.assertTrue((artifact_dir / "model_training_manifest.json").exists())
            predictions_df = pd.read_csv(artifact_dir / "match_probability_predictions.csv")
            self.assertTrue(
                {
                    "event_id",
                    "match_id",
                    "match_date",
                    "match_seq",
                    "team1_player_id",
                    "team2_player_id",
                    "model_name",
                    "prob_team1_victory",
                    "prob_team2_victory",
                    "predicted_label_team1_win",
                    "actual_team1_win",
                }.issubset(set(predictions_df.columns))
            )
            self.assertTrue(pd.api.types.is_numeric_dtype(predictions_df["prob_team1_victory"]))
            self.assertTrue(pd.api.types.is_numeric_dtype(predictions_df["prob_team2_victory"]))
            complementary = predictions_df["prob_team1_victory"] + predictions_df["prob_team2_victory"]
            self.assertTrue(((complementary - 1.0).abs() <= 1e-6).all())

            for name in ("decision_tree", "random_forest", "gbdt"):
                self.assertTrue((artifact_dir / f"roc_curve__{name}.png").exists())

            self.assertIn("models", manifest)
            self.assertEqual(3, len(manifest["models"]))
            self.assertNotIn("market_pricing_evaluation_csv", manifest.get("artifacts", {}))
            self.assertIn("match_probability_predictions_csv", manifest.get("artifacts", {}))

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
            self.assertIn("full_feature_hyperparameter_tuning", manifests["structured_only"])

    def test_full_feature_hyperparameter_tuning_writes_visual_artifacts(self) -> None:
        rows = []
        for i in range(160):
            rows.append(
                {
                    "event_id": f"e{i//4}",
                    "match_id": f"m{i}",
                    "match_date": f"2024-04-{(i % 28) + 1:02d}",
                    "match_seq": i,
                    "team1_player_id": f"p{i}",
                    "team2_player_id": f"q{i}",
                    "surface_context": "Clay" if i % 2 == 0 else "Hard",
                    "rank_diff": (i % 13) - 6,
                    "elo_diff_team1": (i % 27) - 13,
                    "temporal_recent_win_rate_team1": (i % 10) / 10.0,
                    "cluster_kmeans_label": i % 5,
                    "team1_wins": 1 if i % 5 in (1, 2, 4) else 0,
                }
            )

        full_df = pd.DataFrame(rows)
        feature_set_tables = {"data_plus_temporal_elo_clustering": full_df}
        with tempfile.TemporaryDirectory() as tmpdir:
            manifests = run_feature_set_training_experiment(
                feature_set_tables,
                output_dir=tmpdir,
                config={
                    "depth_values": [2],
                    "rf_n_estimators": 20,
                    "gbdt_n_estimators": 20,
                    "hyperparameter_tuning_time_budget_seconds": 120,
                    "hyperparameter_tuning_n_estimators": [10, 20],
                    "hyperparameter_tuning_max_depth": [2, 4],
                    "hyperparameter_tuning_min_samples_leaf": [2, 5],
                    "hyperparameter_tuning_learning_rate": [0.05],
                    "hyperparameter_tuning_subsample": [0.8],
                },
            )
            tuning_dir = Path(tmpdir) / "model_training_hyperparameter_tuning"
            self.assertTrue((tuning_dir / "hyperparameter_tuning_manifest.json").exists())
            self.assertTrue((tuning_dir / "hyperparameter_tuning_results.csv").exists())
            self.assertTrue((tuning_dir / "hyperparameter_tuning_best_log_loss.png").exists())
            self.assertTrue((tuning_dir / "hyperparameter_tuning_curve__random_forest.png").exists())
            self.assertTrue((tuning_dir / "hyperparameter_tuning_curve__gbdt.png").exists())
            tuning_results_df = pd.read_csv(tuning_dir / "hyperparameter_tuning_results.csv")
            self.assertIn("training_accuracy", tuning_results_df.columns)
            run_manifest = manifests["data_plus_temporal_elo_clustering"]["full_feature_hyperparameter_tuning"]
            self.assertEqual("completed", run_manifest.get("status"))
            self.assertIn("artifacts", run_manifest)

    def test_market_pricing_evaluation_metrics_from_synthetic_inputs(self) -> None:
        y_test = pd.Series([1, 0, 1, 0], dtype=int)
        model_prob_team1 = [0.60, 0.45, 0.70, 0.30]
        market_frame = pd.DataFrame(
            {
                "t1_odds": [1.90, 2.20, 1.70, 2.80],
                "t2_odds": [2.00, 1.75, 2.25, 1.50],
            }
        )
        pricing_df, summary = _compute_market_pricing_evaluation(
            model_name="decision_tree",
            y_test=y_test,
            model_probability_team1=model_prob_team1,
            market_frame=market_frame,
            cfg={
                "market_team1_odds_column": "t1_odds",
                "market_team2_odds_column": "t2_odds",
                "market_team1_implied_prob_column": None,
                "market_team2_implied_prob_column": None,
                "market_payout_convention": "decimal",
                "market_edge_bucket_count": 4,
            },
        )

        self.assertFalse(pricing_df.empty)
        self.assertEqual(8, len(pricing_df))
        team1_row = pricing_df[(pricing_df["match_index"] == 0) & (pricing_df["side"] == "team1")].iloc[0]
        self.assertAlmostEqual(0.60 * 1.90 - 1.0, team1_row["expected_value"], places=8)
        self.assertAlmostEqual(1.90 - 1.0, team1_row["realized_return"], places=8)
        self.assertIn("edge_bucket_roi", summary)
        self.assertGreaterEqual(len(summary["edge_bucket_roi"]), 2)

    def test_training_with_market_columns_creates_artifact_and_manifest_summary(self) -> None:
        rows = []
        for i in range(90):
            rows.append(
                {
                    "event_id": f"e{i//3}",
                    "match_id": f"m{i}",
                    "match_date": f"2024-03-{(i % 28) + 1:02d}",
                    "match_seq": i,
                    "team1_player_id": f"p{i}",
                    "team2_player_id": f"q{i}",
                    "surface_context": "Clay" if i % 2 == 0 else "Hard",
                    "rank_diff": (i % 17) - 8,
                    "elo_diff_team1": (i % 31) - 15,
                    "market_team1_odds": 1.70 + (i % 5) * 0.1,
                    "market_team2_odds": 1.65 + ((i + 2) % 5) * 0.1,
                    "team1_wins": 1 if i % 4 in (1, 2) else 0,
                }
            )

        df = pd.DataFrame(rows)
        with tempfile.TemporaryDirectory() as tmpdir:
            manifest = run_model_training_experiments(
                df,
                output_dir=tmpdir,
                config={
                    "depth_values": [2],
                    "rf_n_estimators": 20,
                    "gbdt_n_estimators": 20,
                    "market_team1_odds_column": "market_team1_odds",
                    "market_team2_odds_column": "market_team2_odds",
                    "market_edge_bucket_count": 5,
                },
            )
            artifact_dir = Path(tmpdir) / "model_training"
            market_eval_path = artifact_dir / "market_pricing_evaluation.csv"

            self.assertTrue(market_eval_path.exists())
            market_df = pd.read_csv(market_eval_path)
            self.assertIn("expected_value", market_df.columns)
            self.assertIn("probability_delta_model_minus_market", market_df.columns)
            self.assertIn("edge_bucket", market_df.columns)

            market_summary = manifest.get("market_pricing_evaluation", {})
            self.assertTrue(market_summary.get("enabled"))
            self.assertEqual("decimal", market_summary.get("payout_convention"))
            self.assertEqual(3, len(market_summary.get("models", [])))
            self.assertIn("market_pricing_evaluation_csv", manifest.get("artifacts", {}))


if __name__ == "__main__":
    unittest.main()
