import importlib.util
import io
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

import pandas as pd

from tennis_pipeline.experiments.model_training import _select_training_feature_columns, run_model_training_experiments


class ModelTrainingFeatureSelectionTests(unittest.TestCase):
    def test_drops_leakage_columns_during_training_selection(self) -> None:
        df = pd.DataFrame(
            [
                {
                    "event_id": "e1",
                    "match_id": "m1",
                    "match_date": "2024-01-01",
                    "match_seq": 1,
                    "team1_player_id": "p1",
                    "team2_player_id": "p2",
                    "rank_diff": -5,
                    "diff_sglrollrank": -5,
                    "diff_sets_0_stats_pointstats_totalpointswon_dividend": 0.9,
                    "diff_sets_0_stats_returnstats_breakpointsconverted_dividend": 0.6,
                    "team1_wins": 1,
                }
            ]
        )

        cols = _select_training_feature_columns(
            df,
            {
                "id_columns": ["event_id", "match_id", "match_date", "match_seq", "team1_player_id", "team2_player_id"],
                "target_column": "team1_wins",
            },
        )

        self.assertIn("rank_diff", cols)
        self.assertIn("diff_sglrollrank", cols)
        self.assertNotIn("diff_sets_0_stats_pointstats_totalpointswon_dividend", cols)
        self.assertNotIn("diff_sets_0_stats_returnstats_breakpointsconverted_dividend", cols)


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

    def test_debug_leakage_prints_summary_after_leakage_filtering(self) -> None:
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
        self.assertIn("none", output)
        self.assertIn("[leakage-debug] near-perfect target copies detected:", output)


if __name__ == "__main__":
    unittest.main()
