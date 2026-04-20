import importlib
import unittest

import pandas as pd

step = importlib.import_module("tennis_pipeline.steps.06c_build_features_clustering")


class ClusteringFeatureStepTests(unittest.TestCase):
    def _sample_df(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "event_id": ["e1", "e1", "e1", "e2", "e2", "e2"],
                "match_id": ["6", "5", "4", "3", "2", "1"],
                "match_date": pd.to_datetime(
                    ["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05", "2024-01-06"]
                ),
                "team1_player_id": ["p1", "p2", "p3", "p4", "p5", "p6"],
                "team2_player_id": ["q1", "q2", "q3", "q4", "q5", "q6"],
                "team1_wins": [1, 0, 1, 0, 1, 0],
                "elo_team1_pre": [1600, 1580, 1620, 1500, 1490, 1510],
                "elo_team2_pre": [1500, 1510, 1490, 1520, 1540, 1530],
                "temporal_team1_rolling_win_pct": [0.65, 0.60, 0.72, 0.55, 0.51, 0.57],
                "temporal_team2_rolling_win_pct": [0.35, 0.40, 0.28, 0.45, 0.49, 0.43],
            }
        )

    def test_kmeans_adds_cluster_column(self) -> None:
        df = self._sample_df()
        out = step.run(df, config={"method": "kmeans", "kmeans_n_clusters": 2, "train_fraction": 0.5})
        self.assertIn("cluster_kmeans_id", out.columns)
        self.assertEqual(len(out), len(df))
        self.assertNotIn("cluster_dbscan_id", out.columns)

    def test_dbscan_adds_cluster_column(self) -> None:
        df = self._sample_df()
        out = step.run(df, config={"method": "dbscan", "dbscan_eps": 2.0, "dbscan_min_samples": 1, "train_fraction": 0.5})
        self.assertIn("cluster_dbscan_id", out.columns)
        self.assertEqual(len(out), len(df))
        self.assertNotIn("cluster_kmeans_id", out.columns)

    def test_fit_scope_all_data_runs(self) -> None:
        df = self._sample_df()
        out = step.run(df, config={"method": "both", "fit_scope": "all_data", "kmeans_n_clusters": 2, "dbscan_min_samples": 1})
        self.assertIn("cluster_kmeans_id", out.columns)
        self.assertIn("cluster_dbscan_id", out.columns)


if __name__ == "__main__":
    unittest.main()
