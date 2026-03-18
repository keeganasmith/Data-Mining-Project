"""Reusable preprocessing helpers for ATP/tennis match datasets.

All functions mutate the provided DataFrame in-place and also return it for
convenience/chaining.
"""

import re

import joblib
import pandas as pd

RAW_DATA_PATH = "./data/raw_data.joblib"


# Columns/patterns that directly encode match outcomes or post-match box score
# information and should not be used for pre-match winner prediction features.
DEFAULT_LEAKAGE_TOKEN_PATTERNS = (
    r"\bwinner\b",
    r"\bwinning\b",
    r"\bloser\b",
    r"\bresult\b",
    r"\bscore\b",
    r"\bmatchtime\b",
    r"\btime\b",
    r"\bstats\b",
    r"\baces\b",
    r"\bdoublefaults\b",
    r"\bbreakpoints\b",
    r"\breturnpointswon\b",
    r"\bservicepointswon\b",
    r"\btotalpointswon\b",
    r"\bsetscore\b",
    r"\btiebreakscore\b",
    # Identifier columns should not reach train-time features: they leak identity
    # and can inflate random split metrics via memorization.
    r"\bplayerid\b",
    r"\bmatchid\b",
)


def get_raw_tennis_df(path=RAW_DATA_PATH):
    """Load the serialized raw tennis DataFrame."""
    return joblib.load(path)


def remove_unnamed_cols(df):
    unnamed_cols = [c for c in df.columns if c.startswith("Unnamed:")]
    if unnamed_cols:
        df.drop(columns=unnamed_cols, inplace=True)
    return df


def remove_duplicates(df, key_cols=None):
    """Drop duplicates using a key if present, otherwise by full-row match."""
    if key_cols is None:
        key_cols = ["EventId", "EventYear", "MatchId"]

    available_keys = [c for c in key_cols if c in df.columns]
    before = len(df)

    if available_keys:
        df.drop_duplicates(subset=available_keys, inplace=True, keep="first")
    else:
        df.drop_duplicates(inplace=True, keep="first")

    return before - len(df)


def coerce_dates(df, date_cols=None):
    if date_cols is None:
        date_cols = [
            "StartDate",
            "EndDate",
            "PlayerTeam1.RankDate",
            "PlayerTeam2.RankDate",
            "Date",
            "date",
            "tourney_date",
        ]

    available_date_cols = set(date_cols)
    available_date_cols.update([c for c in df.columns if c.lower().endswith("date")])

    for col in sorted(available_date_cols):
        if col not in df.columns:
            continue

        if col == "tourney_date":
            # ATP files commonly encode dates as YYYYMMDD integers.
            df[col] = pd.to_datetime(df[col].astype("Int64").astype(str), format="%Y%m%d", errors="coerce")
        else:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    return df


def drop_percent_cols(df):
    percent_cols = [c for c in df.columns if c.endswith(".Percent")]
    if percent_cols:
        df.drop(columns=percent_cols, inplace=True)
    return df


def find_leakage_columns(df, token_patterns=None):
    """Return sorted columns that likely leak target information.

    Detection is intentionally conservative and based on column-name patterns so
    it works across yearly ATP exports with inconsistent schemas.
    """
    if token_patterns is None:
        token_patterns = DEFAULT_LEAKAGE_TOKEN_PATTERNS

    compiled = [re.compile(pattern, flags=re.IGNORECASE) for pattern in token_patterns]
    leakage_cols = []
    for col in df.columns:
        norm_col = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", col)
        norm_col = norm_col.replace(".", " ").replace("_", " ")
        if any(pattern.search(norm_col) for pattern in compiled):
            leakage_cols.append(col)
    return sorted(set(leakage_cols))


def drop_leakage_cols(df, token_patterns=None):
    """Drop columns that would cause label leakage for pre-match modeling."""
    leakage_cols = find_leakage_columns(df, token_patterns=token_patterns)
    if leakage_cols:
        df.drop(columns=leakage_cols, inplace=True)
    return leakage_cols


def aggregate_set_stats_into_set0(df, drop_source_set_cols=False):
    """Aggregate Sets[1..n] stat columns into Sets[0] stat columns."""
    set_stat_pattern = re.compile(r"Sets\[(\d+)\]")
    source_cols_to_drop = []

    for source_col in list(df.columns):
        if ".Stats." not in source_col:
            continue

        match = set_stat_pattern.search(source_col)
        if match is None:
            continue

        set_idx = int(match.group(1))
        if set_idx == 0:
            continue

        target_col = source_col.replace(f"Sets[{set_idx}]", "Sets[0]")
        if target_col not in df.columns:
            continue

        lhs = pd.to_numeric(df[target_col], errors="coerce")
        rhs = pd.to_numeric(df[source_col], errors="coerce")

        combined = lhs.fillna(0) + rhs.fillna(0)
        df[target_col] = combined.where(lhs.notna() | rhs.notna(), pd.NA)

        if drop_source_set_cols:
            source_cols_to_drop.append(source_col)

    if source_cols_to_drop:
        df.drop(columns=source_cols_to_drop, inplace=True)

    return df


def preprocess_raw_tennis(df, key_cols=None):
    remove_unnamed_cols(df)
    remove_duplicates(df, key_cols=key_cols)
    coerce_dates(df)
    aggregate_set_stats_into_set0(df)
    drop_percent_cols(df)
    return df


def preprocess_atp_matches(df, drop_leakage=False):
    """Preprocess ATP yearly CSV-style match data.

    Args:
        drop_leakage: when True, removes obvious post-match/outcome columns
            (winner/loser, per-set stats, score and match-time fields).
            Keep this False through preprocessing so labeling can still use
            outcome columns (e.g., WinningPlayerId).
    """
    remove_unnamed_cols(df)

    # Column normalization for consistency across years/exports.
    df.columns = [c.strip() for c in df.columns]

    # Common ATP duplicate keys. Falls back to full-row duplicate removal if absent.
    remove_duplicates(
        df,
        key_cols=["tourney_id", "tourney_date", "match_num", "winner_id", "loser_id"],
    )

    coerce_dates(
        df,
        date_cols=[
            "StartDate",
            "EndDate",
            "PlayerTeam1.RankDate",
            "PlayerTeam2.RankDate",
            "tourney_date",
            "date",
        ],
    )

    numeric_candidates = [
        "PlayerTeam1.SglRollRank",
        "PlayerTeam2.SglRollRank",
        "PlayerTeam1.SglRaceRank",
        "PlayerTeam2.SglRaceRank",
        "NumberOfSets",
        "minutes",
    ]
    for col in numeric_candidates:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if drop_leakage:
        drop_leakage_cols(df)

    return df


if __name__ == "__main__":
    raw = get_raw_tennis_df()
    cleaned = preprocess_raw_tennis(raw)
    print(cleaned.shape)
