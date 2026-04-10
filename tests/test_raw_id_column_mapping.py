import unittest
import importlib

import pandas as pd


class RawIdColumnMappingTests(unittest.TestCase):
    def test_match_and_event_ids_use_correct_raw_columns(self) -> None:
        fixture = pd.DataFrame(
            [
                {
                    "MatchId": "m-123",
                    "EventId": "e-999",
                    "StartDate": "2024-01-01",
                    "WinningPlayerId": "A",
                    "PlayerTeam1.PlayerId": "A",
                    "PlayerTeam2.PlayerId": "B",
                    "PlayerTeam1.SglRollRank": 5,
                    "PlayerTeam2.SglRollRank": 10,
                }
            ]
        )

        clean_schema = importlib.import_module("tennis_pipeline.steps.02_clean_schema")
        out = clean_schema.run(fixture)

        self.assertEqual("m-123", out.loc[0, "match_id"])
        self.assertEqual("e-999", out.loc[0, "event_id"])


if __name__ == "__main__":
    unittest.main()
