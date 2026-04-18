"""Shared temporal ordering helpers for feature generation and validation."""

from __future__ import annotations

import pandas as pd

_ORDER_MATCH_ID_NUM_COL = "__order_match_id_num"
_ORDER_MATCH_ID_NUM_DESC_COL = "__order_match_id_num_desc"
_ORDER_MATCH_ID_NON_NUMERIC_COL = "__order_match_id_non_numeric"
_ORDER_MATCH_ID_TEXT_COL = "__order_match_id_text"
_ORDER_MATCH_ID_TEXT_DESC_RANK_COL = "__order_match_id_text_desc_rank"


def prepare_temporal_ordering(
    df: pd.DataFrame, *, stable_tie_breaker: str
) -> tuple[pd.DataFrame, list[str], str, list[str]]:
    """Return dataframe plus canonical temporal ordering keys and explanation text."""

    working = df.copy(deep=False)
    sort_cols = ["match_date"]
    temp_cols: list[str] = []

    if "match_seq" in working.columns:
        sort_cols.append("match_seq")
        tie_breaker_text = "match_seq"
    elif "match_id" in working.columns:
        if "event_id" in working.columns:
            # MatchIds are only chronological within an event.
            sort_cols.append("event_id")

        match_id_num = pd.to_numeric(working["match_id"], errors="coerce")
        working[_ORDER_MATCH_ID_NUM_COL] = match_id_num
        working[_ORDER_MATCH_ID_NUM_DESC_COL] = -match_id_num
        working[_ORDER_MATCH_ID_NON_NUMERIC_COL] = match_id_num.isna().astype("int8")
        working[_ORDER_MATCH_ID_TEXT_COL] = (
            working["match_id"].astype("string").fillna("").str.strip()
        )
        working[_ORDER_MATCH_ID_TEXT_DESC_RANK_COL] = working[_ORDER_MATCH_ID_TEXT_COL].rank(
            method="dense", ascending=False
        )
        sort_cols.extend(
            [
                _ORDER_MATCH_ID_NON_NUMERIC_COL,
                _ORDER_MATCH_ID_NUM_DESC_COL,
                _ORDER_MATCH_ID_TEXT_DESC_RANK_COL,
            ]
        )
        temp_cols.extend(
            [
                _ORDER_MATCH_ID_NUM_COL,
                _ORDER_MATCH_ID_NUM_DESC_COL,
                _ORDER_MATCH_ID_NON_NUMERIC_COL,
                _ORDER_MATCH_ID_TEXT_COL,
                _ORDER_MATCH_ID_TEXT_DESC_RANK_COL,
            ]
        )
        tie_breaker_text = "event_id + match_id (descending within event; numeric coercion first, string fallback for non-numeric IDs)"
    else:
        tie_breaker_text = "stable row order (no match_seq/match_id present)"

    sort_cols.append(stable_tie_breaker)
    return working, sort_cols, tie_breaker_text, temp_cols
