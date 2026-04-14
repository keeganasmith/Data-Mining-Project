import importlib
import unittest

import pandas as pd


class FeatureSuffixNormalizationTests(unittest.TestCase):
    def test_normalize_suffix_handles_camel_case_metrics(self) -> None:
        static_features = importlib.import_module("tennis_pipeline.steps.05_build_features_static")

        self.assertEqual("aces_per_service_game", static_features._normalize_suffix("AcesPerServiceGame"))
        self.assertEqual("double_faults_per_service_game", static_features._normalize_suffix("DoubleFaultsPerServiceGame"))
        self.assertEqual("break_points_saved_pct", static_features._normalize_suffix("BreakPointsSavedPct"))
        self.assertEqual("return_points_won_pct", static_features._normalize_suffix("ReturnPointsWonPct"))
        self.assertEqual("service_points_won_pct", static_features._normalize_suffix("ServicePointsWonPct"))

    def test_build_paired_player_features_uses_snake_case_suffixes(self) -> None:
        static_features = importlib.import_module("tennis_pipeline.steps.05_build_features_static")

        fixture = pd.DataFrame(
            {
                "PlayerTeam1.AcesPerServiceGame": [0.55, 0.35],
                "PlayerTeam2.AcesPerServiceGame": [0.40, 0.45],
                "PlayerTeam1.BreakPointsSavedPct": [0.70, 0.62],
                "PlayerTeam2.BreakPointsSavedPct": [0.66, 0.60],
            }
        )

        out = static_features.build_paired_player_features(fixture)

        self.assertIn("diff_aces_per_service_game", out.columns)
        self.assertIn("abs_diff_aces_per_service_game", out.columns)
        self.assertIn("diff_break_points_saved_pct", out.columns)
        self.assertIn("abs_diff_break_points_saved_pct", out.columns)


if __name__ == "__main__":
    unittest.main()
