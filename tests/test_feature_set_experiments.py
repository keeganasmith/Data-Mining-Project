import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from tennis_pipeline.experiments.feature_sets import materialize_feature_sets


class FeatureSetExperimentsTests(unittest.TestCase):
    def test_materialize_feature_sets_supports_data_only_vs_engineered_bundle(self) -> None:
        df = pd.DataFrame(
            {
                "event_id": ["e1", "e2"],
                "match_id": ["m1", "m2"],
                "match_date": ["2024-01-01", "2024-01-02"],
                "match_seq": [1, 2],
                "team1_player_id": ["p1", "p2"],
                "team2_player_id": ["q1", "q2"],
                "rank_diff": [1, -2],
                "elo_diff_team1": [22.0, -15.0],
                "temporal_team1_win_rate": [0.6, 0.4],
                "cluster_kmeans_id": [2, 1],
                "team1_wins": [1, 0],
            }
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.object(pd.DataFrame, "to_parquet", return_value=None):
                artifacts = materialize_feature_sets(df, output_dir=tmpdir)

            self.assertIn("data_only", artifacts)
            self.assertIn("data_plus_temporal_elo_clustering", artifacts)

            data_only_cols = set(artifacts["data_only"].columns)
            engineered_cols = set(artifacts["data_plus_temporal_elo_clustering"].columns)

            self.assertIn("rank_diff", data_only_cols)
            self.assertNotIn("elo_diff_team1", data_only_cols)
            self.assertNotIn("temporal_team1_win_rate", data_only_cols)
            self.assertNotIn("cluster_kmeans_id", data_only_cols)

            self.assertIn("elo_diff_team1", engineered_cols)
            self.assertIn("temporal_team1_win_rate", engineered_cols)
            self.assertIn("cluster_kmeans_id", engineered_cols)

            manifest_path = Path(tmpdir) / "experiments" / "feature_set_manifest.json"
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertIn("temporal_feature_columns", payload)
            self.assertIn("clustering_feature_columns", payload)


if __name__ == "__main__":
    unittest.main()
