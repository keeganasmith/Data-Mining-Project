import importlib
import unittest

import pandas as pd


class TemporalEloRollingFeatureTests(unittest.TestCase):
    def test_rolling_features_are_pre_match_and_leakage_safe(self) -> None:
        step = importlib.import_module("tennis_pipeline.steps.06b_build_features_temporal_rolling")

        df = pd.DataFrame(
            [
                {
                    "match_id": "m1",
                    "match_date": "2024-01-01",
                    "team1_player_id": "A",
                    "team2_player_id": "B",
                    "team1_service_points_won_pct": 0.60,
                    "team2_service_points_won_pct": 0.40,
                    "team1_wins": 1,
                },
                {
                    "match_id": "m2",
                    "match_date": "2024-01-02",
                    "team1_player_id": "A",
                    "team2_player_id": "C",
                    "team1_service_points_won_pct": 0.30,
                    "team2_service_points_won_pct": 0.70,
                    "team1_wins": 0,
                },
                {
                    "match_id": "m3",
                    "match_date": "2024-01-03",
                    "team1_player_id": "B",
                    "team2_player_id": "A",
                    "team1_service_points_won_pct": 0.50,
                    "team2_service_points_won_pct": 0.50,
                    "team1_wins": 1,
                },
            ]
        )

        out = step.run(df)

        # Row 1 has no history.
        self.assertAlmostEqual(float(out.loc[0, "temporal_team1_rolling_win_pct"]), 0.5)
        self.assertAlmostEqual(float(out.loc[0, "temporal_team2_rolling_win_pct"]), 0.5)
        self.assertAlmostEqual(float(out.loc[0, "temporal_team1_rolling_avg_service_points_won_pct"]), 0.0)

        # Row 2 (A vs C): A has one prior win + one prior service metric.
        self.assertAlmostEqual(float(out.loc[1, "temporal_team1_rolling_win_pct"]), 1.0)
        self.assertAlmostEqual(float(out.loc[1, "temporal_team2_rolling_win_pct"]), 0.5)
        self.assertAlmostEqual(float(out.loc[1, "temporal_team1_rolling_avg_service_points_won_pct"]), 0.60)

        # Row 3 (B vs A):
        # - B's pre-match history is one loss and service=0.40.
        # - A's pre-match history includes row1 win and row2 loss => 0.5, service avg=(0.60+0.30)/2.
        self.assertAlmostEqual(float(out.loc[2, "temporal_team1_rolling_win_pct"]), 0.0)
        self.assertAlmostEqual(float(out.loc[2, "temporal_team2_rolling_win_pct"]), 0.5)
        self.assertAlmostEqual(float(out.loc[2, "temporal_team1_rolling_avg_service_points_won_pct"]), 0.40)
        self.assertAlmostEqual(float(out.loc[2, "temporal_team2_rolling_avg_service_points_won_pct"]), 0.45)

    def test_rolling_window_limits_history_length(self) -> None:
        step = importlib.import_module("tennis_pipeline.steps.06b_build_features_temporal_rolling")

        df = pd.DataFrame(
            [
                {
                    "match_id": "m1",
                    "match_date": "2024-01-01",
                    "team1_player_id": "A",
                    "team2_player_id": "B",
                    "team1_wins": 1,
                },
                {
                    "match_id": "m2",
                    "match_date": "2024-01-02",
                    "team1_player_id": "A",
                    "team2_player_id": "B",
                    "team1_wins": 0,
                },
                {
                    "match_id": "m3",
                    "match_date": "2024-01-03",
                    "team1_player_id": "A",
                    "team2_player_id": "B",
                    "team1_wins": 1,
                },
            ]
        )

        out = step.run(df, config={"rolling_window_matches": 1})

        # With window=1, row3 should only see row2's outcome for both players.
        self.assertAlmostEqual(float(out.loc[2, "temporal_team1_rolling_win_pct"]), 0.0)
        self.assertAlmostEqual(float(out.loc[2, "temporal_team2_rolling_win_pct"]), 1.0)


if __name__ == "__main__":
    unittest.main()
