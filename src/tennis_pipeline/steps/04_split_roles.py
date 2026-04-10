"""Step 04: align player roles and assign binary target for modeling rows."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd

_DEFAULT_CONFIG: dict[str, Any] = {
    "random_seed": 42,
    "winner_column": "winner_player_id",
    "team1_id_column": "team1_player_id",
    "team2_id_column": "team2_player_id",
    "target_column": "team1_wins",
    "drop_invalid_winner_rows": True,
    "drop_final_role_duplicates": True,
    "final_role_duplicates_keep": "first",
}


def _normalize_config(config: Mapping[str, Any] | None) -> dict[str, Any]:
    normalized = dict(_DEFAULT_CONFIG)
    if config:
        normalized.update(dict(config))

    for key in ("winner_column", "team1_id_column", "team2_id_column", "target_column"):
        if not isinstance(normalized.get(key), str) or not normalized[key].strip():
            raise TypeError(f"config['{key}'] must be a non-empty string")

    seed = normalized.get("random_seed")
    if not isinstance(seed, int):
        raise TypeError("config['random_seed'] must be an int")

    if normalized.get("final_role_duplicates_keep") not in {"first", "last", False}:
        raise TypeError("config['final_role_duplicates_keep'] must be one of {'first', 'last', False}")

    return normalized


def _candidate_pairs(df: pd.DataFrame) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()

    for col in df.columns:
        if col.startswith("team1_"):
            mate = "team2_" + col[len("team1_") :]
        elif col.startswith("PlayerTeam1."):
            mate = "PlayerTeam2." + col[len("PlayerTeam1.") :]
        else:
            continue

        if mate in df.columns:
            pair = (col, mate)
            if pair not in seen:
                pairs.append(pair)
                seen.add(pair)

    return pairs


def run(df_or_path: pd.DataFrame, config: Mapping[str, Any] | None = None) -> pd.DataFrame:
    """Create deterministic Team1/Team2 role alignment and binary target.

    The transformation is idempotent: running this step repeatedly yields the
    same output because swaps are based on a deterministic hash of
    winner/loser/player identifiers plus ``random_seed``.
    """

    if not isinstance(df_or_path, pd.DataFrame):
        raise TypeError("04_split_roles.run expects a pandas DataFrame input")

    cfg = _normalize_config(config)
    winner_col = cfg["winner_column"]
    team1_id_col = cfg["team1_id_column"]
    team2_id_col = cfg["team2_id_column"]
    target_col = cfg["target_column"]

    required = [winner_col, team1_id_col, team2_id_col]
    missing = [col for col in required if col not in df_or_path.columns]
    if missing:
        raise ValueError(f"[split_roles] Missing required columns: {missing}")

    df = df_or_path.copy(deep=True)

    winner_is_team1 = df[winner_col] == df[team1_id_col]
    winner_is_team2 = df[winner_col] == df[team2_id_col]
    valid_winner = winner_is_team1 | winner_is_team2

    if cfg.get("drop_invalid_winner_rows", True):
        df = df[valid_winner].copy(deep=True)
        winner_is_team1 = winner_is_team1.loc[df.index]
        winner_is_team2 = winner_is_team2.loc[df.index]

    winner_ids = df[winner_col]
    loser_ids = np.where(winner_is_team1, df[team2_id_col], df[team1_id_col])

    stable_key = pd.DataFrame(
        {
            "winner_id": winner_ids.astype("string"),
            "loser_id": pd.Series(loser_ids, index=df.index, dtype="string"),
            "seed": str(cfg["random_seed"]),
        },
        index=df.index,
    )
    swap = (pd.util.hash_pandas_object(stable_key, index=False).to_numpy() % 2) == 1

    paired_columns = _candidate_pairs(df)
    for team1_col, team2_col in paired_columns:
        team1_vals = df[team1_col].to_numpy(copy=False)
        team2_vals = df[team2_col].to_numpy(copy=False)

        winner_vals = np.where(winner_is_team1.to_numpy(), team1_vals, team2_vals)
        loser_vals = np.where(winner_is_team1.to_numpy(), team2_vals, team1_vals)

        df[team1_col] = np.where(swap, loser_vals, winner_vals)
        df[team2_col] = np.where(swap, winner_vals, loser_vals)

    # Target is fully determined by deterministic role assignment.
    df[target_col] = (~swap).astype(int)

    dedupe_subset = ["match_id", team1_id_col, team2_id_col]
    if cfg.get("drop_final_role_duplicates", True) and all(col in df.columns for col in dedupe_subset):
        before = len(df)
        df = df.drop_duplicates(
            subset=dedupe_subset,
            keep=cfg.get("final_role_duplicates_keep", "first"),
        ).copy(deep=True)
        df.attrs["final_role_duplicates_dropped"] = before - len(df)

    return df
