import unittest

import pandas as pd

from tennis_pipeline.temporal_ordering import prepare_temporal_ordering


class TemporalOrderingTests(unittest.TestCase):
    def test_match_id_orders_descending_within_event(self) -> None:
        fixture = pd.DataFrame(
            [
                {"match_date": "2024-01-01", "event_id": "e1", "match_id": "1"},
                {"match_date": "2024-01-01", "event_id": "e1", "match_id": "3"},
                {"match_date": "2024-01-01", "event_id": "e1", "match_id": "2"},
                {"match_date": "2024-01-01", "event_id": "e2", "match_id": "1"},
            ]
        )

        ordered, sort_cols, _tie_breaker_text, _temp_cols = prepare_temporal_ordering(
            fixture, stable_tie_breaker="__row_pos"
        )
        ordered["__row_pos"] = range(len(ordered))
        out = ordered.sort_values(sort_cols, kind="mergesort")

        self.assertEqual(["3", "2", "1"], out.loc[out["event_id"] == "e1", "match_id"].tolist())


if __name__ == "__main__":
    unittest.main()
