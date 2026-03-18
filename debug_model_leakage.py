"""Debug utilities for diagnosing suspiciously perfect tennis match model metrics.

This script reproduces the feature-construction logic used in checkpoint 2,
then runs leakage diagnostics focused on:
1) Outcome/post-match columns accidentally entering X
2) Identifier columns (player/match IDs) that enable memorization
3) Columns whose values map deterministically to the target
4) Train/test overlap due to duplicated feature rows

Run:
    python debug_model_leakage.py
"""

from __future__ import annotations

import hashlib
import os
import re
from typing import Iterable

import numpy as np
import pandas as pd

from preprocessing import find_leakage_columns, preprocess_atp_matches

RANDOM_SEED = 42
DATA_DIR = "data/csv_data"
YEARS = list(range(2000, 2027))
CSV_PATHS = [os.path.join(DATA_DIR, f"atp_{year}.csv") for year in YEARS]


def load_dataset() -> pd.DataFrame:
    frames = []
    for path in CSV_PATHS:
        if os.path.exists(path):
            tmp = pd.read_csv(path, low_memory=False)
            tmp["source_file"] = os.path.basename(path)
            frames.append(tmp)

    if not frames:
        raise FileNotFoundError("No ATP CSV files were found under data/csv_data/.")

    df = pd.concat(frames, ignore_index=True)
    print(f"Loaded rows: {len(df):,} | columns: {df.shape[1]}")
    df = preprocess_atp_matches(df)
    print(f"After preprocessing rows: {len(df):,} | columns: {df.shape[1]}")
    return df


def build_checkpoint2_like_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Reproduce the notebook's feature engineering and return X, y."""
    working = df.copy()

    required = [
        "WinningPlayerId",
        "PlayerTeam1.PlayerId",
        "PlayerTeam2.PlayerId",
        "PlayerTeam1.SglRollRank",
        "PlayerTeam2.SglRollRank",
    ]
    missing_required = [c for c in required if c not in working.columns]
    if missing_required:
        raise KeyError(f"Required columns missing: {missing_required}")

    working = working.dropna(subset=required).copy()

    # Random side swap to break fixed Team1 orientation
    team1_cols = [c for c in working.columns if c.startswith("PlayerTeam1.")]
    team2_cols = [c for c in working.columns if c.startswith("PlayerTeam2.")]
    suffixes = sorted(
        {c.split("PlayerTeam1.", 1)[1] for c in team1_cols}
        & {c.split("PlayerTeam2.", 1)[1] for c in team2_cols}
    )

    swap_mask = np.random.default_rng(RANDOM_SEED).random(len(working)) < 0.5
    for suffix in suffixes:
        c1 = f"PlayerTeam1.{suffix}"
        c2 = f"PlayerTeam2.{suffix}"
        left = working.loc[swap_mask, c1].copy()
        right = working.loc[swap_mask, c2].copy()
        working.loc[swap_mask, c1] = right.values
        working.loc[swap_mask, c2] = left.values

    # Binary label
    working["team1_wins"] = (
        working["WinningPlayerId"] == working["PlayerTeam1.PlayerId"]
    ).astype(int)

    paired_numeric_suffixes = [
        "SglRollRank",
        "SglRollPoints",
        "SglRaceRank",
        "SglRacePoints",
        "Age",
        "HeightCm",
        "WeightLb",
        "ProYear",
    ]
    paired_categorical_suffixes = [
        "PlayHand",
        "Backhand",
        "PlayerCountryCode",
    ]

    for suffix in paired_numeric_suffixes:
        c1 = f"PlayerTeam1.{suffix}"
        c2 = f"PlayerTeam2.{suffix}"
        if c1 in working.columns and c2 in working.columns:
            f1 = f"team1_{suffix.lower()}"
            f2 = f"team2_{suffix.lower()}"
            working[f1] = pd.to_numeric(working[c1], errors="coerce")
            working[f2] = pd.to_numeric(working[c2], errors="coerce")
            working[f"diff_{suffix.lower()}"] = working[f1] - working[f2]
            working[f"abs_diff_{suffix.lower()}"] = working[f"diff_{suffix.lower()}"].abs()

    for suffix in paired_categorical_suffixes:
        c1 = f"PlayerTeam1.{suffix}"
        c2 = f"PlayerTeam2.{suffix}"
        if c1 in working.columns and c2 in working.columns:
            f1 = f"team1_{suffix.lower()}"
            f2 = f"team2_{suffix.lower()}"
            working[f1] = working[c1].astype(object)
            working[f2] = working[c2].astype(object)
            working[f1] = working[f1].where(pd.notna(working[f1]), np.nan)
            working[f2] = working[f2].where(pd.notna(working[f2]), np.nan)
            working[f"same_{suffix.lower()}"] = (
                working[f1].notna() & working[f2].notna() & (working[f1] == working[f2])
            ).astype(int)

    if "diff_sglrollrank" in working.columns:
        working["rank_diff"] = working["diff_sglrollrank"]
        working["abs_rank_diff"] = working["abs_diff_sglrollrank"]
    if "diff_sglracerank" in working.columns:
        working["race_rank_diff"] = working["diff_sglracerank"]
        working["abs_race_rank_diff"] = working["abs_diff_sglracerank"]
    else:
        working["race_rank_diff"] = np.nan
        working["abs_race_rank_diff"] = np.nan

    working = working.dropna(subset=["rank_diff"]).copy()

    meta_feature_candidates = [
        "Round.ShortName",
        "Round.LongName",
        "CourtSurface",
        "Court.Surface",
        "Court",
        "EventType",
        "InOutdoor",
        "TournamentCity",
    ]

    # Recreate the original notebook selection rule to detect accidental target capture.
    engineered_features_buggy = [
        c
        for c in working.columns
        if c.startswith("team1_")
        or c.startswith("team2_")
        or c.startswith("diff_")
        or c.startswith("abs_diff_")
        or c.startswith("same_")
    ]
    if "team1_wins" in engineered_features_buggy:
        print("\n[BUG DETECTED] 'team1_wins' is selected as a feature by prefix rule (team1_*).")

    # Fixed rule: explicitly exclude the target from engineered features.
    engineered_features = [c for c in engineered_features_buggy if c != "team1_wins"]

    feature_cols = engineered_features + [
        c for c in meta_feature_candidates if c in working.columns
    ]

    X = working[feature_cols].copy()
    y = working["team1_wins"].copy()

    print(f"\nFinal training rows: {len(X):,}")
    print(f"Final feature count: {len(feature_cols)}")
    print("Features used at training time:")
    for c in feature_cols:
        print(f"  - {c}")

    return X, y


def detect_pattern_columns(columns: Iterable[str], patterns: Iterable[str]) -> list[str]:
    comp = [re.compile(p, flags=re.IGNORECASE) for p in patterns]
    out = []
    for c in columns:
        norm = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", c).replace(".", " ").replace("_", " ")
        if any(p.search(norm) for p in comp):
            out.append(c)
    return sorted(set(out))


def deterministic_mapping_check(X: pd.DataFrame, y: pd.Series, max_report: int = 15) -> None:
    """Report columns where each value maps to exactly one class label."""
    suspicious = []
    for col in X.columns:
        s = X[col]
        # skip huge continuous-like columns (almost unique) where this metric is not useful
        nunique = s.nunique(dropna=True)
        if nunique == 0:
            continue
        if nunique > len(s) * 0.95:
            continue

        tmp = pd.DataFrame({"v": s.astype("object"), "y": y.values}).dropna(subset=["v"])
        if tmp.empty:
            continue

        label_counts_per_value = tmp.groupby("v")["y"].nunique(dropna=True)
        deterministic = (label_counts_per_value <= 1).all()
        if deterministic:
            coverage = len(tmp) / len(X)
            suspicious.append((col, nunique, coverage))

    suspicious.sort(key=lambda t: (-t[2], t[0]))
    print("\nColumns with deterministic value->label mapping (possible leakage/memorization):")
    if not suspicious:
        print("  none")
        return
    for col, nunique, coverage in suspicious[:max_report]:
        print(f"  - {col}: unique={nunique}, coverage={coverage:.3f}")


def overlap_check(X: pd.DataFrame, y: pd.Series, test_size: float = 0.2) -> None:
    """Check if identical feature rows appear in both train and test after a random split."""
    n = len(X)
    rng = np.random.default_rng(RANDOM_SEED)
    idx = np.arange(n)
    rng.shuffle(idx)
    cut = int((1 - test_size) * n)
    train_idx = idx[:cut]
    test_idx = idx[cut:]

    X_train = X.iloc[train_idx]
    X_test = X.iloc[test_idx]
    y_train = y.iloc[train_idx]
    y_test = y.iloc[test_idx]

    def row_hash(df: pd.DataFrame) -> pd.Series:
        rows = df.astype("string").fillna("<NA>").agg("|".join, axis=1)
        return rows.map(lambda t: hashlib.md5(t.encode("utf-8")).hexdigest())

    train_hash = row_hash(X_train)
    test_hash = row_hash(X_test)

    train_map = pd.DataFrame({"h": train_hash.values, "y": y_train.values})
    test_map = pd.DataFrame({"h": test_hash.values, "y": y_test.values})

    common_hashes = set(train_map["h"]).intersection(set(test_map["h"]))
    print(f"\nExact feature-row overlap train<->test: {len(common_hashes):,} shared row hashes")

    if common_hashes:
        train_common = train_map[train_map["h"].isin(common_hashes)]
        test_common = test_map[test_map["h"].isin(common_hashes)]
        # If same feature row appears with conflicting labels, memorization is less likely to inflate to 100%
        train_label_set = train_common.groupby("h")["y"].nunique()
        test_label_set = test_common.groupby("h")["y"].nunique()
        deterministic_overlap = (
            (train_label_set == 1)
            & (test_label_set == 1)
        ).sum()
        print(f"Deterministic shared hashes (single label in both splits): {int(deterministic_overlap):,}")


def main() -> None:
    df = load_dataset()

    print("\nVerified column checks:")
    for col in [
        "WinningPlayerId",
        "PlayerTeam1.PlayerId",
        "PlayerTeam2.PlayerId",
        "PlayerTeam1.SglRollRank",
        "PlayerTeam2.SglRollRank",
    ]:
        print(f"  - {col}: {'present' if col in df.columns else 'MISSING'}")

    X, y = build_checkpoint2_like_features(df)

    print(f"\nTarget distribution: mean={y.mean():.4f}, counts={y.value_counts().to_dict()}")

    # Default project leakage detector
    leak_default = find_leakage_columns(X)
    print(f"\nfind_leakage_columns(X) detected {len(leak_default)} columns")
    for c in leak_default[:30]:
        print(f"  - {c}")

    # Additional explicit patterns
    explicit_patterns = [
        r"\\bwinning\\b",
        r"\\bwinner\\b",
        r"\\bloser\\b",
        r"\\bresult\\b",
        r"\\bscore\\b",
        r"\\bmatchtime\\b",
        r"\\bstats\\b",
        r"\\bplayerid\\b",
        r"\\bmatchid\\b",
        r"\\bset\\b",
    ]
    explicit = detect_pattern_columns(X.columns, explicit_patterns)
    print(f"\nExplicit-pattern matched columns in X: {len(explicit)}")
    for c in explicit[:30]:
        print(f"  - {c}")

    deterministic_mapping_check(X, y)
    overlap_check(X, y)


if __name__ == "__main__":
    main()
