#Data Pre-Processing, always assume these functions are done in-place (for efficiency reasons)
import pandas as pd
import joblib
RAW_DATA_PATH = "./data/raw_data.joblib"
def get_raw_tennis_df():
    raw_tennis_df = joblib.load(RAW_DATA_PATH)
    return raw_tennis_df
raw_tennis_df = get_raw_tennis_df()

def remove_unnamed_cols(df):
    df.drop(columns=[c for c in ["Unnamed: 0"] if c in df.columns], inplace=True)


def remove_duplicates(df, key_cols=None):
    if key_cols is None:
        key_cols = ["EventId", "EventYear", "MatchId"]  # default tournament-scoped key
    key_cols = [c for c in key_cols if c in df.columns]

    # Only drop duplicates if we actually have key columns
    if key_cols:
        df.drop_duplicates(subset=key_cols, inplace=True, keep="first")

def coerce_dates(df):
    for c in ["StartDate", "EndDate", "PlayerTeam1.RankDate", "PlayerTeam2.RankDate"]:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce")

#percent columns are not necessary since precision is bad, 
#and we are provided with divisor and dividend, so we can compute this ourselves
def drop_percent_cols(df):
    percent_cols = [c for c in df.columns if c.endswith(".Percent")]
    df.drop(columns=percent_cols, inplace=True)

def preprocess_raw_tennis(raw_tennis_df):
    remove_unnamed_cols(raw_tennis_df)
    remove_duplicates(raw_tennis_df)
    coerce_dates(raw_tennis_df)
    drop_percent_cols(raw_tennis_df)
    return raw_tennis_df

cleaned_tennis_df = preprocess_raw_tennis(raw_tennis_df)