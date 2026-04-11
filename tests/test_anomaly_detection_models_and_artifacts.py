import importlib
import tempfile
import unittest
from pathlib import Path

import pandas as pd


class AnomalyDetectionModelsAndArtifactsTests(unittest.TestCase):
    def test_step_emits_knn_iforest_scores_and_artifacts(self) -> None:
        fixture = pd.DataFrame(
            [
                {
                    "event_id": "e1",
                    "match_id": "m1",
                    "match_date": "2024-01-01",
                    "team1_player_id": "A",
                    "team2_player_id": "B",
                    "surface_context": "Clay",
                    "rank_diff": 10,
                    "abs_rank_diff": 10,
                    "race_rank_diff": 8,
                    "abs_race_rank_diff": 8,
                    "elo_diff_team1": 12,
                    "elo_prob_team1_pre": 0.54,
                    "team1_sgl_roll_rank": 15,
                    "team2_sgl_roll_rank": 25,
                },
                {
                    "event_id": "e1",
                    "match_id": "m2",
                    "match_date": "2024-01-02",
                    "team1_player_id": "C",
                    "team2_player_id": "D",
                    "surface_context": "Clay",
                    "rank_diff": -2,
                    "abs_rank_diff": 2,
                    "race_rank_diff": -1,
                    "abs_race_rank_diff": 1,
                    "elo_diff_team1": -4,
                    "elo_prob_team1_pre": 0.48,
                    "team1_sgl_roll_rank": 40,
                    "team2_sgl_roll_rank": 38,
                },
                {
                    "event_id": "e2",
                    "match_id": "m3",
                    "match_date": "2024-01-03",
                    "team1_player_id": "E",
                    "team2_player_id": "F",
                    "surface_context": "Grass",
                    "rank_diff": 80,
                    "abs_rank_diff": 80,
                    "race_rank_diff": 77,
                    "abs_race_rank_diff": 77,
                    "elo_diff_team1": 150,
                    "elo_prob_team1_pre": 0.88,
                    "team1_sgl_roll_rank": 2,
                    "team2_sgl_roll_rank": 130,
                },
            ]
        )

        step = importlib.import_module("tennis_pipeline.steps.06b_build_features_anomaly_surface")
        with tempfile.TemporaryDirectory() as tmpdir:
            out = step.run(
                fixture,
                config={
                    "artifact_output_dir": tmpdir,
                    "knn_neighbors": 2,
                    "artifact_top_n": 2,
                },
            )

            for col in ("knn_anomaly_score", "iforest_anomaly_score", "robust_z_anomaly_score", "anomaly_score"):
                self.assertIn(col, out.columns)

            self.assertTrue(Path(tmpdir, "anomaly_summary_by_surface.csv").exists())
            self.assertTrue(Path(tmpdir, "anomaly_top_rows.csv").exists())
            self.assertTrue(Path(tmpdir, "anomaly_summary.json").exists())


if __name__ == "__main__":
    unittest.main()
