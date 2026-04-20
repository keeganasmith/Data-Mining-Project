import importlib
import itertools
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

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

    def test_auto_tuning_writes_artifact_and_plots(self) -> None:
        df = self._sample_df()
        with tempfile.TemporaryDirectory() as tmp:
            artifact_path = Path(tmp) / "cluster_artifact.json"
            plot_dir = Path(tmp) / "plots"
            out = step.run(
                df,
                config={
                    "method": "both",
                    "fit_scope": "all_data",
                    "train_fraction": 0.5,
                    "tuning_artifact_path": str(artifact_path),
                    "tuning_plot_dir": str(plot_dir),
                    "dbscan_tuning_min_samples": [1, 2],
                },
            )
            self.assertIn("cluster_kmeans_id", out.columns)
            self.assertIn("cluster_dbscan_id", out.columns)
            self.assertTrue(artifact_path.exists())
            artifact_payload = json.loads(artifact_path.read_text(encoding="utf-8"))
            self.assertIn("dbscan_staged_results", artifact_payload)
            self.assertIn("stage1", artifact_payload["dbscan_staged_results"])
            self.assertIn("stage2", artifact_payload["dbscan_staged_results"])
            self.assertTrue((plot_dir / "clustering_tuning_kmeans.png").exists())
            self.assertTrue((plot_dir / "clustering_tuning_dbscan.png").exists())

    def test_existing_artifact_skips_tuning(self) -> None:
        df = self._sample_df()
        with tempfile.TemporaryDirectory() as tmp:
            artifact_path = Path(tmp) / "cluster_artifact.json"
            plot_dir = Path(tmp) / "plots"
            base_config = {
                "method": "both",
                "fit_scope": "all_data",
                "train_fraction": 0.5,
                "tuning_artifact_path": str(artifact_path),
                "tuning_plot_dir": str(plot_dir),
                "dbscan_tuning_min_samples": [1, 2],
            }
            _ = step.run(df, config=base_config)
            with mock.patch.object(step, "_tune_kmeans", side_effect=AssertionError("kmeans retuned unexpectedly")):
                with mock.patch.object(step, "_tune_dbscan", side_effect=AssertionError("dbscan retuned unexpectedly")):
                    out = step.run(df, config=base_config)
            self.assertIn("cluster_kmeans_id", out.columns)
            self.assertIn("cluster_dbscan_id", out.columns)

    def test_tuning_budget_stops_early_and_falls_back_to_defaults(self) -> None:
        df = self._sample_df()
        with tempfile.TemporaryDirectory() as tmp:
            artifact_path = Path(tmp) / "cluster_artifact.json"
            plot_dir = Path(tmp) / "plots"
            tick = itertools.count()
            with mock.patch.object(step.time, "perf_counter", side_effect=lambda: 100.0 + next(tick) * 0.2):
                out = step.run(
                    df,
                    config={
                        "method": "both",
                        "fit_scope": "all_data",
                        "train_fraction": 0.5,
                        "tuning_artifact_path": str(artifact_path),
                        "tuning_plot_dir": str(plot_dir),
                        "kmeans_n_clusters": 3,
                        "dbscan_eps": 1.1,
                        "dbscan_min_samples": 2,
                        "dbscan_tuning_min_samples": [1, 2],
                        "tuning_time_budget_seconds": 0.05,
                    },
                )
            self.assertIn("cluster_kmeans_id", out.columns)
            self.assertIn("cluster_dbscan_id", out.columns)
            artifact_payload = json.loads(artifact_path.read_text(encoding="utf-8"))
            self.assertEqual(artifact_payload["kmeans"]["n_clusters"], 3)
            self.assertEqual(artifact_payload["dbscan"]["eps"], 1.1)
            self.assertEqual(artifact_payload["dbscan"]["min_samples"], 2)
            self.assertTrue(artifact_payload["kmeans_tuning_metadata"]["stopped_early"])
            self.assertTrue(artifact_payload["dbscan_tuning_metadata"]["stopped_early"])
            self.assertGreater(artifact_payload["kmeans_tuning_metadata"]["elapsed_seconds"], 0.0)
            self.assertGreater(artifact_payload["dbscan_tuning_metadata"]["elapsed_seconds"], 0.0)


if __name__ == "__main__":
    unittest.main()
