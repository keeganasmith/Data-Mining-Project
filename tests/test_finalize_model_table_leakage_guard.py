import importlib
import unittest

import pandas as pd


class FinalizeModelTableLeakageGuardTests(unittest.TestCase):
    def test_filters_in_match_set_stats_from_prefixed_diff_features(self) -> None:
        step = importlib.import_module("tennis_pipeline.steps.07_finalize_model_table")

        df = pd.DataFrame(
            [
                {
                    "event_id": "e1",
                    "match_id": "m1",
                    "match_date": "2024-01-01",
                    "team1_player_id": "p1",
                    "team2_player_id": "p2",
                    "rank_diff": -5,
                    "diff_sglrollrank": -5,
                    "diff_sets_0_stats_pointstats_totalpointswon_dividend": 0.9,
                    "diff_sets_0_stats_returnstats_breakpointsconverted_dividend": 0.6,
                    "elo_diff_team1": 45.0,
                    "team1_wins": 1,
                }
            ]
        )

        out = step.run(df)
        self.assertIn("rank_diff", out.columns)
        self.assertIn("diff_sglrollrank", out.columns)
        self.assertIn("elo_diff_team1", out.columns)
        self.assertNotIn("diff_sets_0_stats_pointstats_totalpointswon_dividend", out.columns)
        self.assertNotIn("diff_sets_0_stats_returnstats_breakpointsconverted_dividend", out.columns)


if __name__ == "__main__":
    unittest.main()
