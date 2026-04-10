import unittest
import importlib

import pandas as pd


class CanonicalPairDedupeTests(unittest.TestCase):
    def test_mirrored_pairs_are_deduped_before_stage_04(self) -> None:
        fixture = pd.DataFrame(
            [
                {
                    "match_id": "m1",
                    "match_date": "2024-01-01",
                    "winner_player_id": "A",
                    "team1_player_id": "A",
                    "team2_player_id": "B",
                    "team1_sgl_roll_rank": 5,
                    "team2_sgl_roll_rank": 11,
                },
                {
                    "match_id": "m1",
                    "match_date": "2024-01-01",
                    "winner_player_id": "A",
                    "team1_player_id": "B",
                    "team2_player_id": "A",
                    "team1_sgl_roll_rank": 11,
                    "team2_sgl_roll_rank": 5,
                },
            ]
        )

        clean_values = importlib.import_module("tennis_pipeline.steps.03_clean_values")
        split_roles = importlib.import_module("tennis_pipeline.steps.04_split_roles")

        clean_df = clean_values.run(fixture)
        self.assertEqual(1, len(clean_df))
        self.assertEqual(1, clean_df.attrs.get("canonical_pair_duplicates_dropped"))

        split_df = split_roles.run(clean_df)
        duplicate_count = split_df.duplicated(
            subset=["match_id", "team1_player_id", "team2_player_id"]
        ).sum()
        self.assertEqual(0, int(duplicate_count))


if __name__ == "__main__":
    unittest.main()
