"""Model training experiments for tree-based classifiers."""

from __future__ import annotations

import json
import time
from collections.abc import Mapping
from datetime import timezone, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

DEFAULT_MODEL_TRAINING_CONFIG: dict[str, Any] = {
    "enabled": True,
    "profile": "default",
    "debug_leakage": False,
    "verbose_training": False,
    "target_column": "team1_wins",
    "id_columns": ["event_id", "match_id", "match_date", "match_seq", "team1_player_id", "team2_player_id"],
    "date_column": "match_date",
    # Keep a single fixed depth by default (no depth sweep).
    "depth_values": [8],
    "test_size": 0.2,
    "validation_size": 0.2,
    "output_subdir": "model_training",
    "random_state": 42,
    "rf_n_estimators": 300,
    "gbdt_n_estimators": 300,
    "gbdt_learning_rate": 0.05,
    "dt_min_samples_leaf": 25,
    "rf_min_samples_leaf": 15,
    "gbdt_min_samples_leaf": 20,
    "gbdt_subsample": 0.8,
    "market_team1_odds_column": None,
    "market_team2_odds_column": None,
    "market_team1_implied_prob_column": None,
    "market_team2_implied_prob_column": None,
    "market_payout_convention": "decimal",
    "market_edge_bucket_count": 10,
}

MODEL_TRAINING_PROFILES: dict[str, dict[str, Any]] = {
    "default": {},
    "fast": {
        "rf_n_estimators": 50,
        "gbdt_n_estimators": 50,
    },
}


def _normalize_config(config: Mapping[str, Any] | None) -> dict[str, Any]:
    raw_config = dict(config or {})
    cfg = dict(DEFAULT_MODEL_TRAINING_CONFIG)
    profile = str(raw_config.get("profile", cfg["profile"])).strip().lower()
    if profile not in MODEL_TRAINING_PROFILES:
        allowed = ", ".join(sorted(MODEL_TRAINING_PROFILES))
        raise ValueError(f"model_training config['profile'] must be one of: {allowed}")
    cfg.update(MODEL_TRAINING_PROFILES[profile])
    cfg.update(raw_config)
    cfg["profile"] = profile

    if not isinstance(cfg.get("enabled"), bool):
        raise TypeError("model_training config['enabled'] must be a bool")
    if not isinstance(cfg.get("debug_leakage"), bool):
        raise TypeError("model_training config['debug_leakage'] must be a bool")
    if not isinstance(cfg.get("verbose_training"), bool):
        raise TypeError("model_training config['verbose_training'] must be a bool")

    for key in ("target_column", "output_subdir", "date_column", "profile"):
        value = cfg.get(key)
        if not isinstance(value, str) or not value.strip():
            raise TypeError(f"model_training config['{key}'] must be a non-empty string")

    for key in (
        "market_team1_odds_column",
        "market_team2_odds_column",
        "market_team1_implied_prob_column",
        "market_team2_implied_prob_column",
    ):
        value = cfg.get(key)
        if value is not None and (not isinstance(value, str) or not value.strip()):
            raise TypeError(f"model_training config['{key}'] must be null or a non-empty string")

    if not isinstance(cfg.get("market_payout_convention"), str) or not cfg["market_payout_convention"].strip():
        raise TypeError("model_training config['market_payout_convention'] must be a non-empty string")
    cfg["market_payout_convention"] = cfg["market_payout_convention"].strip().lower()
    if cfg["market_payout_convention"] not in {"decimal", "net_odds"}:
        raise ValueError("model_training config['market_payout_convention'] must be one of: decimal, net_odds")

    for key in ("id_columns", "depth_values"):
        value = cfg.get(key)
        if not isinstance(value, list) or not value:
            raise TypeError(f"model_training config['{key}'] must be a non-empty list")

    cfg["depth_values"] = sorted({int(v) for v in cfg["depth_values"] if int(v) > 0})
    if not cfg["depth_values"]:
        raise ValueError("model_training config['depth_values'] must contain at least one positive depth")

    for key in ("test_size", "validation_size"):
        value = cfg.get(key)
        if not isinstance(value, (int, float)):
            raise TypeError(f"model_training config['{key}'] must be numeric")
        if value <= 0 or value >= 0.5:
            raise ValueError(f"model_training config['{key}'] must be in (0, 0.5)")

    if cfg["test_size"] + cfg["validation_size"] >= 0.8:
        raise ValueError("test_size + validation_size must be < 0.8")

    for key in (
        "random_state",
        "rf_n_estimators",
        "gbdt_n_estimators",
        "dt_min_samples_leaf",
        "rf_min_samples_leaf",
        "gbdt_min_samples_leaf",
    ):
        cfg[key] = int(cfg[key])

    cfg["gbdt_learning_rate"] = float(cfg["gbdt_learning_rate"])
    cfg["gbdt_subsample"] = float(cfg["gbdt_subsample"])
    cfg["market_edge_bucket_count"] = int(cfg["market_edge_bucket_count"])
    if cfg["market_edge_bucket_count"] < 2:
        raise ValueError("model_training config['market_edge_bucket_count'] must be >= 2")
    return cfg


def _market_probability_from_odds(odds: pd.Series) -> pd.Series:
    odds_numeric = pd.to_numeric(odds, errors="coerce")
    return np.where(odds_numeric > 0, 1.0 / odds_numeric, np.nan)


def _resolve_market_side_inputs(
    source_df: pd.DataFrame,
    *,
    odds_column: str | None,
    implied_prob_column: str | None,
) -> tuple[pd.Series, pd.Series]:
    implied_prob = (
        pd.to_numeric(source_df[implied_prob_column], errors="coerce") if implied_prob_column and implied_prob_column in source_df.columns else pd.Series(np.nan, index=source_df.index)
    )
    odds = pd.to_numeric(source_df[odds_column], errors="coerce") if odds_column and odds_column in source_df.columns else pd.Series(np.nan, index=source_df.index)

    if implied_prob.isna().all() and odds.notna().any():
        implied_prob = pd.Series(_market_probability_from_odds(odds), index=source_df.index)
    if odds.isna().all() and implied_prob.notna().any():
        odds = pd.to_numeric(np.where(implied_prob > 0, 1.0 / implied_prob, np.nan), errors="coerce")
    return implied_prob, odds


def _compute_expected_value(model_prob: pd.Series, odds: pd.Series, *, payout_convention: str) -> pd.Series:
    if payout_convention == "decimal":
        return (model_prob * odds) - 1.0
    if payout_convention == "net_odds":
        return (model_prob * (odds + 1.0)) - 1.0
    raise ValueError(f"Unsupported payout convention: {payout_convention}")


def _compute_realized_return(outcome: pd.Series, odds: pd.Series, *, payout_convention: str) -> pd.Series:
    if payout_convention == "decimal":
        return np.where(outcome == 1, odds - 1.0, -1.0)
    if payout_convention == "net_odds":
        return np.where(outcome == 1, odds, -1.0)
    raise ValueError(f"Unsupported payout convention: {payout_convention}")


def _compute_market_pricing_evaluation(
    *,
    model_name: str,
    y_test: pd.Series,
    model_probability_team1: np.ndarray,
    market_frame: pd.DataFrame,
    cfg: Mapping[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    team1_implied, team1_odds = _resolve_market_side_inputs(
        market_frame,
        odds_column=cfg.get("market_team1_odds_column"),
        implied_prob_column=cfg.get("market_team1_implied_prob_column"),
    )
    team2_implied, team2_odds = _resolve_market_side_inputs(
        market_frame,
        odds_column=cfg.get("market_team2_odds_column"),
        implied_prob_column=cfg.get("market_team2_implied_prob_column"),
    )

    proba_team1 = pd.Series(model_probability_team1, index=y_test.index, dtype=float)
    proba_team2 = 1.0 - proba_team1
    outcome_team1 = y_test.astype(int)
    outcome_team2 = 1 - outcome_team1

    per_side = pd.concat(
        [
            pd.DataFrame(
                {
                    "model": model_name,
                    "match_index": y_test.index,
                    "side": "team1",
                    "model_probability": proba_team1,
                    "market_implied_probability": team1_implied,
                    "market_odds": team1_odds,
                    "actual_outcome": outcome_team1,
                }
            ),
            pd.DataFrame(
                {
                    "model": model_name,
                    "match_index": y_test.index,
                    "side": "team2",
                    "model_probability": proba_team2,
                    "market_implied_probability": team2_implied,
                    "market_odds": team2_odds,
                    "actual_outcome": outcome_team2,
                }
            ),
        ],
        ignore_index=True,
    )

    valid = per_side["market_implied_probability"].between(0.0, 1.0) & (per_side["market_odds"] > 0.0)
    per_side = per_side[valid].copy()
    if per_side.empty:
        return pd.DataFrame(), {"model": model_name, "status": "skipped_no_valid_market_rows"}

    per_side["probability_delta_model_minus_market"] = per_side["model_probability"] - per_side["market_implied_probability"]
    per_side["absolute_probability_error_vs_market"] = per_side["probability_delta_model_minus_market"].abs()
    per_side["expected_value"] = _compute_expected_value(
        per_side["model_probability"], per_side["market_odds"], payout_convention=str(cfg["market_payout_convention"])
    )
    per_side["realized_return"] = _compute_realized_return(
        per_side["actual_outcome"], per_side["market_odds"], payout_convention=str(cfg["market_payout_convention"])
    )
    per_side["bet_signal"] = per_side["probability_delta_model_minus_market"] > 0.0
    per_side["realized_return_if_bet"] = np.where(per_side["bet_signal"], per_side["realized_return"], 0.0)

    ranked = per_side["probability_delta_model_minus_market"].rank(method="first")
    per_side["edge_bucket"] = pd.qcut(
        ranked,
        q=min(int(cfg["market_edge_bucket_count"]), int(len(per_side))),
        labels=False,
        duplicates="drop",
    )
    per_side["edge_bucket"] = per_side["edge_bucket"].astype(int)

    bucket_summary = (
        per_side.groupby("edge_bucket", as_index=False)
        .agg(
            side_count=("side", "size"),
            bet_count=("bet_signal", "sum"),
            avg_edge=("probability_delta_model_minus_market", "mean"),
            avg_expected_value=("expected_value", "mean"),
            realized_roi_if_bet=("realized_return_if_bet", "mean"),
        )
        .sort_values(by="edge_bucket", kind="stable")
    )

    summary = {
        "model": model_name,
        "rows_evaluated": int(len(per_side)),
        "mean_probability_delta_model_minus_market": float(per_side["probability_delta_model_minus_market"].mean()),
        "mean_abs_probability_error_vs_market": float(per_side["absolute_probability_error_vs_market"].mean()),
        "mean_expected_value": float(per_side["expected_value"].mean()),
        "mean_realized_roi_if_bet": float(per_side["realized_return_if_bet"].mean()),
        "bet_rate": float(per_side["bet_signal"].mean()),
        "edge_bucket_roi": bucket_summary.to_dict(orient="records"),
    }
    return per_side, summary


def _print_leakage_debug_info(x: pd.DataFrame, y: pd.Series, *, target_column: str) -> None:
    """Emit lightweight diagnostics to help identify label leakage."""
    print(f"[leakage-debug] target='{target_column}' rows={len(x)} feature_count={len(x.columns)}")

    suspicious_tokens = ("winner", "winning", "loser", "result", "score", "outcome")
    suspicious_name_hits = sorted([c for c in x.columns if any(t in c.lower() for t in suspicious_tokens)])
    if suspicious_name_hits:
        print(f"[leakage-debug] suspicious feature names: {suspicious_name_hits}")
    else:
        print("[leakage-debug] suspicious feature names: none")

    id_like_cols = sorted([c for c in x.columns if c.lower().endswith("_id") or ".id" in c.lower()])
    if id_like_cols:
        print(f"[leakage-debug] identifier-like columns still in feature matrix: {id_like_cols}")

    numeric = x.select_dtypes(include=[np.number, "bool"])
    corr_hits: list[tuple[str, float]] = []
    for col in numeric.columns:
        aligned = pd.concat([numeric[col], y], axis=1).dropna()
        if len(aligned) < 25:
            continue
        corr = float(aligned.iloc[:, 0].corr(aligned.iloc[:, 1]))
        if np.isfinite(corr):
            corr_hits.append((col, abs(corr)))

    corr_hits = sorted(corr_hits, key=lambda item: item[1], reverse=True)
    if corr_hits:
        top = [f"{name}:{score:.3f}" for name, score in corr_hits[:8]]
        print(f"[leakage-debug] top abs(feature,target) correlations: {top}")

    near_perfect_hits: list[str] = []
    y_numeric = pd.to_numeric(y, errors="coerce")
    for col in numeric.columns:
        feature = pd.to_numeric(numeric[col], errors="coerce")
        aligned = pd.concat([feature, y_numeric], axis=1).dropna()
        if len(aligned) < 25:
            continue
        feature_aligned = aligned.iloc[:, 0]
        y_aligned = aligned.iloc[:, 1]
        same_ratio = float((feature_aligned == y_aligned).mean())
        inv_ratio = float((feature_aligned == (1 - y_aligned)).mean())
        if same_ratio >= 0.95 or inv_ratio >= 0.95:
            near_perfect_hits.append(f"{col}(same={same_ratio:.3f}, inverse={inv_ratio:.3f})")

    if near_perfect_hits:
        print(f"[leakage-debug] near-perfect target copies detected: {near_perfect_hits}")
    else:
        print("[leakage-debug] near-perfect target copies detected: none")


def _temporal_split(df: pd.DataFrame, *, date_column: str, test_size: float, validation_size: float) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    ordered = df.copy(deep=True)
    if date_column in ordered.columns:
        ordered["__sort_date"] = pd.to_datetime(ordered[date_column], errors="coerce")
    else:
        ordered["__sort_date"] = pd.NaT

    sort_columns = ["__sort_date"]
    for col in ("match_seq", "event_id", "match_id"):
        if col in ordered.columns:
            sort_columns.append(col)

    ordered = ordered.sort_values(by=sort_columns, kind="mergesort").reset_index(drop=True)

    n_rows = len(ordered)
    test_count = max(1, int(round(n_rows * test_size)))
    val_count = max(1, int(round(n_rows * validation_size)))

    train_end = n_rows - (test_count + val_count)
    if train_end < 1:
        raise ValueError("Not enough rows for temporal train/validation/test split")

    train_df = ordered.iloc[:train_end].drop(columns=["__sort_date"])
    val_df = ordered.iloc[train_end : train_end + val_count].drop(columns=["__sort_date"])
    test_df = ordered.iloc[train_end + val_count :].drop(columns=["__sort_date"])
    return train_df, val_df, test_df


def run_model_training_experiments(
    df: pd.DataFrame,
    *,
    output_dir: str | Path,
    config: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Train tree-based models and persist metrics/plots."""

    if not isinstance(df, pd.DataFrame):
        raise TypeError("run_model_training_experiments expects a pandas DataFrame")

    cfg = _normalize_config(config)
    if not cfg["enabled"]:
        return {}

    try:
        import matplotlib
        from sklearn.compose import ColumnTransformer
        from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
        from sklearn.impute import SimpleImputer
        from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score, roc_curve
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import OneHotEncoder
        from sklearn.tree import DecisionTreeClassifier
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "model_training requires optional dependencies: scikit-learn and matplotlib"
        ) from exc

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    target_column = cfg["target_column"]
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' is required for model training")

    output_path = Path(output_dir) / cfg["output_subdir"]
    output_path.mkdir(parents=True, exist_ok=True)

    market_config_columns = [
        cfg.get("market_team1_odds_column"),
        cfg.get("market_team2_odds_column"),
        cfg.get("market_team1_implied_prob_column"),
        cfg.get("market_team2_implied_prob_column"),
    ]
    market_columns = {str(c) for c in market_config_columns if isinstance(c, str) and c in df.columns}
    feature_columns = [
        c
        for c in df.columns
        if c not in set(cfg["id_columns"]) and c != target_column and c not in market_columns
    ]
    available_id_columns = [column for column in cfg["id_columns"] if column in df.columns]
    selected_columns = [*available_id_columns, *feature_columns, *sorted(market_columns), target_column]
    model_df = df.loc[:, selected_columns].copy(deep=True)
    model_df = model_df.dropna(subset=[target_column]).reset_index(drop=True)

    y = model_df[target_column].astype(int)
    x = model_df.drop(columns=[target_column])
    if cfg["debug_leakage"]:
        _print_leakage_debug_info(x, y, target_column=target_column)

    split_df = x.copy(deep=True)
    market_eval_columns = sorted(market_columns)
    if market_eval_columns:
        split_df[market_eval_columns] = model_df.loc[:, market_eval_columns]
    split_df[target_column] = y
    train_df, val_df, test_df = _temporal_split(
        split_df,
        date_column=cfg["date_column"],
        test_size=cfg["test_size"],
        validation_size=cfg["validation_size"],
    )

    x_train, y_train = train_df.drop(columns=[target_column]), train_df[target_column].astype(int)
    x_val, y_val = val_df.drop(columns=[target_column]), val_df[target_column].astype(int)
    x_test, y_test = test_df.drop(columns=[target_column]), test_df[target_column].astype(int)
    test_market_frame = x_test.loc[:, market_eval_columns].copy(deep=True) if market_eval_columns else pd.DataFrame(index=x_test.index)
    val_identifiers = x_val.loc[:, available_id_columns].copy(deep=True) if available_id_columns else pd.DataFrame(index=x_val.index)
    test_identifiers = x_test.loc[:, available_id_columns].copy(deep=True) if available_id_columns else pd.DataFrame(index=x_test.index)

    if market_eval_columns:
        x_train = x_train.drop(columns=market_eval_columns, errors="ignore")
        x_val = x_val.drop(columns=market_eval_columns, errors="ignore")
        x_test = x_test.drop(columns=market_eval_columns, errors="ignore")
    if available_id_columns:
        x_train = x_train.drop(columns=available_id_columns, errors="ignore")
        x_val = x_val.drop(columns=available_id_columns, errors="ignore")
        x_test = x_test.drop(columns=available_id_columns, errors="ignore")

    numeric_columns = x_train.select_dtypes(include=[np.number, "bool"]).columns.tolist()
    categorical_columns = [c for c in x_train.columns if c not in set(numeric_columns)]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", Pipeline(steps=[("imputer", SimpleImputer(strategy="median"))]), numeric_columns),
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("encoder", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical_columns,
            ),
        ]
    )

    def _training_log(message: str) -> None:
        if cfg["verbose_training"]:
            print(f"[model-training] {message}")

    model_hyperparams = {
        "decision_tree": {
            "min_samples_leaf": cfg["dt_min_samples_leaf"],
            "random_state": cfg["random_state"],
        },
        "random_forest": {
            "n_estimators": cfg["rf_n_estimators"],
            "min_samples_leaf": cfg["rf_min_samples_leaf"],
            "random_state": cfg["random_state"],
            "n_jobs": -1,
        },
        "gbdt": {
            "n_estimators": cfg["gbdt_n_estimators"],
            "learning_rate": cfg["gbdt_learning_rate"],
            "min_samples_leaf": cfg["gbdt_min_samples_leaf"],
            "subsample": cfg["gbdt_subsample"],
            "random_state": cfg["random_state"],
        },
    }

    model_specs = {
        "decision_tree": lambda depth: DecisionTreeClassifier(
            max_depth=depth,
            min_samples_leaf=cfg["dt_min_samples_leaf"],
            random_state=cfg["random_state"],
        ),
        "random_forest": lambda depth: RandomForestClassifier(
            n_estimators=cfg["rf_n_estimators"],
            max_depth=depth,
            min_samples_leaf=cfg["rf_min_samples_leaf"],
            random_state=cfg["random_state"],
            n_jobs=-1,
        ),
        "gbdt": lambda depth: GradientBoostingClassifier(
            n_estimators=cfg["gbdt_n_estimators"],
            learning_rate=cfg["gbdt_learning_rate"],
            max_depth=depth,
            min_samples_leaf=cfg["gbdt_min_samples_leaf"],
            subsample=cfg["gbdt_subsample"],
            random_state=cfg["random_state"],
        ),
    }

    def _calibration_table(
        y_true: pd.Series,
        y_proba: np.ndarray,
        *,
        bins: int = 10,
    ) -> tuple[pd.DataFrame, float]:
        clipped_proba = np.clip(y_proba, 1e-15, 1.0 - 1e-15)
        bin_edges = np.linspace(0.0, 1.0, bins + 1)
        bin_ids = np.digitize(clipped_proba, bin_edges[1:-1], right=False)

        rows: list[dict[str, float | int]] = []
        total_count = max(1, int(len(clipped_proba)))
        ece = 0.0
        for bin_idx in range(bins):
            mask = bin_ids == bin_idx
            sample_count = int(mask.sum())
            if sample_count == 0:
                positive_rate = np.nan
                mean_predicted_prob = np.nan
                abs_calibration_error = np.nan
                weighted_abs_calibration_error = 0.0
            else:
                y_slice = y_true[mask]
                p_slice = clipped_proba[mask]
                positive_rate = float(np.mean(y_slice))
                mean_predicted_prob = float(np.mean(p_slice))
                abs_calibration_error = float(abs(positive_rate - mean_predicted_prob))
                weighted_abs_calibration_error = abs_calibration_error * (sample_count / total_count)
            ece += weighted_abs_calibration_error
            rows.append(
                {
                    "bin_index": int(bin_idx),
                    "bin_lower": float(bin_edges[bin_idx]),
                    "bin_upper": float(bin_edges[bin_idx + 1]),
                    "sample_count": sample_count,
                    "mean_predicted_probability": mean_predicted_prob,
                    "observed_positive_rate": positive_rate,
                    "absolute_calibration_error": abs_calibration_error,
                    "weighted_absolute_calibration_error": float(weighted_abs_calibration_error),
                }
            )

        return pd.DataFrame(rows), float(ece)

    curve_rows: list[dict[str, float]] = []
    summary_rows: list[dict[str, Any]] = []
    calibration_rows: list[dict[str, Any]] = []
    market_pricing_rows: list[dict[str, Any]] = []
    market_summary_rows: list[dict[str, Any]] = []
    match_probability_prediction_rows: list[dict[str, Any]] = []

    plt.figure(figsize=(12, 8))
    for idx, (model_name, factory) in enumerate(model_specs.items(), start=1):
        _training_log(
            f"start model={model_name} hyperparams={model_hyperparams[model_name]} "
            f"depth_values={cfg['depth_values']}"
        )
        curves: list[dict[str, float]] = []
        best_val_accuracy = -1.0
        best_depth = cfg["depth_values"][0]
        best_pipeline = None

        for depth in cfg["depth_values"]:
            _training_log(f"fit start model={model_name} depth={depth}")
            fit_started = time.perf_counter()
            pipeline = Pipeline(steps=[("prep", preprocessor), ("model", factory(depth))])
            pipeline.fit(x_train, y_train)
            fit_elapsed = time.perf_counter() - fit_started
            _training_log(f"fit end model={model_name} depth={depth} elapsed_sec={fit_elapsed:.2f}")
            train_acc = float(accuracy_score(y_train, pipeline.predict(x_train)))
            val_acc = float(accuracy_score(y_val, pipeline.predict(x_val)))
            curves.append(
                {
                    "model": model_name,
                    "depth": int(depth),
                    "train_accuracy": train_acc,
                    "validation_accuracy": val_acc,
                }
            )
            if val_acc > best_val_accuracy:
                best_val_accuracy = val_acc
                best_depth = int(depth)
                best_pipeline = pipeline

        if best_pipeline is None:  # pragma: no cover
            raise RuntimeError(f"Failed to fit any models for {model_name}")

        curve_rows.extend(curves)
        test_pred = best_pipeline.predict(x_test)
        test_proba = best_pipeline.predict_proba(x_test)[:, 1]
        clipped_test_proba = np.clip(test_proba, 1e-15, 1.0 - 1e-15)

        test_accuracy = float(accuracy_score(y_test, test_pred))
        test_roc_auc = float(roc_auc_score(y_test, test_proba))
        test_log_loss = float(log_loss(y_test, clipped_test_proba, labels=[0, 1]))
        test_brier_score = float(brier_score_loss(y_test, clipped_test_proba))
        calibration_table_df, test_ece_10_bins = _calibration_table(y_test.reset_index(drop=True), clipped_test_proba, bins=10)
        calibration_rows.extend(
            [
                {
                    "model": model_name,
                    **row,
                }
                for row in calibration_table_df.to_dict(orient="records")
            ]
        )
        fpr, tpr, _ = roc_curve(y_test, test_proba)

        train_accuracy_at_best = next(r["train_accuracy"] for r in curves if r["depth"] == best_depth)
        val_accuracy_at_best = next(r["validation_accuracy"] for r in curves if r["depth"] == best_depth)

        summary_rows.append(
            {
                "model": model_name,
                "best_depth": best_depth,
                "train_accuracy_at_best_depth": train_accuracy_at_best,
                "validation_accuracy_at_best_depth": val_accuracy_at_best,
                "test_accuracy": test_accuracy,
                "test_roc_auc": test_roc_auc,
                "test_log_loss": test_log_loss,
                "test_brier_score": test_brier_score,
                "test_ece_10_bins": test_ece_10_bins,
            }
        )
        _training_log(
            f"done model={model_name} best_depth={best_depth} "
            f"val_acc={val_accuracy_at_best:.4f} test_acc={test_accuracy:.4f} "
            f"test_auc={test_roc_auc:.4f} test_log_loss={test_log_loss:.4f} "
            f"test_brier={test_brier_score:.4f} test_ece10={test_ece_10_bins:.4f}"
        )
        if market_eval_columns:
            pricing_df, pricing_summary = _compute_market_pricing_evaluation(
                model_name=model_name,
                y_test=y_test.reset_index(drop=True),
                model_probability_team1=clipped_test_proba,
                market_frame=test_market_frame.reset_index(drop=True),
                cfg=cfg,
            )
            if not pricing_df.empty:
                market_pricing_rows.extend(pricing_df.to_dict(orient="records"))
            market_summary_rows.append(pricing_summary)
        val_proba = np.clip(best_pipeline.predict_proba(x_val)[:, 1], 1e-15, 1.0 - 1e-15)
        val_pred = best_pipeline.predict(x_val)
        val_predictions = val_identifiers.reset_index(drop=True).copy(deep=True)
        val_predictions["model_name"] = model_name
        val_predictions["prob_team1_win"] = val_proba
        val_predictions["predicted_label_team1_win"] = val_pred
        val_predictions["actual_team1_win"] = y_val.reset_index(drop=True).astype(int)
        match_probability_prediction_rows.extend(val_predictions.to_dict(orient="records"))

        test_predictions = test_identifiers.reset_index(drop=True).copy(deep=True)
        test_predictions["model_name"] = model_name
        test_predictions["prob_team1_win"] = clipped_test_proba
        test_predictions["predicted_label_team1_win"] = test_pred
        test_predictions["actual_team1_win"] = y_test.reset_index(drop=True).astype(int)
        match_probability_prediction_rows.extend(test_predictions.to_dict(orient="records"))

        plt.subplot(2, 2, idx)
        plt.plot([r["depth"] for r in curves], [r["train_accuracy"] for r in curves], marker="o", label="Train")
        plt.plot([r["depth"] for r in curves], [r["validation_accuracy"] for r in curves], marker="s", label="Validation")
        plt.title(f"{model_name} accuracy vs depth")
        plt.xlabel("max_depth")
        plt.ylabel("accuracy")
        plt.ylim(0.0, 1.0)
        plt.grid(alpha=0.3)
        plt.legend()

        roc_figure = plt.figure(figsize=(6, 5))
        plt.plot(fpr, tpr, label=f"AUC={test_roc_auc:.3f}")
        plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
        plt.title(f"ROC curve: {model_name}")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.grid(alpha=0.3)
        plt.legend(loc="lower right")
        roc_figure.savefig(output_path / f"roc_curve__{model_name}.png", bbox_inches="tight")
        plt.close(roc_figure)

    pd.DataFrame(curve_rows).to_csv(output_path / "depth_accuracy_curves.csv", index=False)
    pd.DataFrame(summary_rows).sort_values(by="model", kind="stable").to_csv(
        output_path / "model_summary_metrics.csv", index=False
    )
    pd.DataFrame(calibration_rows).sort_values(by=["model", "bin_index"], kind="stable").to_csv(
        output_path / "model_calibration_table.csv", index=False
    )
    pd.DataFrame(match_probability_prediction_rows).to_csv(
        output_path / "match_probability_predictions.csv",
        index=False,
    )
    if market_pricing_rows:
        pd.DataFrame(market_pricing_rows).sort_values(by=["model", "match_index", "side"], kind="stable").to_csv(
            output_path / "market_pricing_evaluation.csv", index=False
        )

    manifest = {
        "configuration": {
            "profile": cfg["profile"],
            "random_state": cfg["random_state"],
            "depth_values": [int(depth) for depth in cfg["depth_values"]],
            "rf_n_estimators": cfg["rf_n_estimators"],
            "gbdt_n_estimators": cfg["gbdt_n_estimators"],
            "gbdt_learning_rate": cfg["gbdt_learning_rate"],
            "dt_min_samples_leaf": cfg["dt_min_samples_leaf"],
            "rf_min_samples_leaf": cfg["rf_min_samples_leaf"],
            "gbdt_min_samples_leaf": cfg["gbdt_min_samples_leaf"],
            "gbdt_subsample": cfg["gbdt_subsample"],
            "test_size": cfg["test_size"],
            "validation_size": cfg["validation_size"],
        },
        "split": {
            "train_rows": int(len(train_df)),
            "validation_rows": int(len(val_df)),
            "test_rows": int(len(test_df)),
            "target_column": target_column,
            "feature_count": int(len(feature_columns)),
        },
        "models": summary_rows,
        "market_pricing_evaluation": {
            "enabled": bool(market_eval_columns),
            "market_columns": market_eval_columns,
            "payout_convention": cfg["market_payout_convention"],
            "edge_bucket_count": cfg["market_edge_bucket_count"],
            "models": market_summary_rows,
        },
        "artifacts": {
            "depth_curve_csv": str(output_path / "depth_accuracy_curves.csv"),
            "summary_csv": str(output_path / "model_summary_metrics.csv"),
            "calibration_table_csv": str(output_path / "model_calibration_table.csv"),
            "match_probability_predictions_csv": str(output_path / "match_probability_predictions.csv"),
            "depth_accuracy_plot": str(output_path / "depth_accuracy_curves.png"),
            "roc_curve_plots": [str(output_path / f"roc_curve__{name}.png") for name in model_specs],
        },
    }
    if market_pricing_rows:
        manifest["artifacts"]["market_pricing_evaluation_csv"] = str(output_path / "market_pricing_evaluation.csv")
    (output_path / "model_training_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    plt.tight_layout()
    plt.savefig(output_path / "depth_accuracy_curves.png", bbox_inches="tight")
    plt.close()

    return manifest


def run_feature_set_training_experiment(
    feature_set_tables: Mapping[str, pd.DataFrame],
    *,
    output_dir: str | Path,
    config: Mapping[str, Any] | None = None,
    start_run_index: int = 1,
    total_runs: int | None = None,
) -> dict[str, Any]:
    """Train model experiments for each materialized feature set."""

    try:
        import matplotlib
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("feature-set training plots require optional dependency: matplotlib") from exc

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    manifests: dict[str, Any] = {}
    run_summary_rows: list[dict[str, Any]] = []
    feature_set_count = len(feature_set_tables)
    inferred_total_runs = total_runs if total_runs is not None else feature_set_count
    for offset, (feature_set_name, feature_set_df) in enumerate(feature_set_tables.items()):
        run_number = start_run_index + offset
        print(
            f"[pipeline] running feature-set model-training experiment: {feature_set_name} "
            f"(run {run_number} / {inferred_total_runs})"
        )
        started_at = datetime.now(timezone.utc)
        start_perf = time.perf_counter()
        print(f"[pipeline] feature-set training start ({feature_set_name}): {started_at.isoformat()}")
        run_config = dict(config or {})
        run_config["output_subdir"] = str(Path("model_training_feature_sets") / feature_set_name)
        run_manifest = run_model_training_experiments(
            feature_set_df,
            output_dir=output_dir,
            config=run_config,
        )
        manifests[feature_set_name] = run_manifest
        for model_metrics in run_manifest.get("models", []):
            model_name = model_metrics.get("model")
            if isinstance(model_name, str):
                run_summary_rows.append(
                    {
                        "feature_set": feature_set_name,
                        "model": model_name,
                        "test_log_loss": float(model_metrics.get("test_log_loss", np.nan)),
                        "test_brier_score": float(model_metrics.get("test_brier_score", np.nan)),
                        "test_ece_10_bins": float(model_metrics.get("test_ece_10_bins", np.nan)),
                        "test_roc_auc": float(model_metrics.get("test_roc_auc", np.nan)),
                        # Accuracy is retained as a secondary diagnostic metric.
                        "test_accuracy": float(model_metrics.get("test_accuracy", np.nan)),
                    }
                )
        ended_at = datetime.now(timezone.utc)
        elapsed_seconds = time.perf_counter() - start_perf
        print(
            f"[pipeline] feature-set training end ({feature_set_name}): "
            f"{ended_at.isoformat()} (elapsed {elapsed_seconds:.2f}s)"
        )

    summary_dir = Path(output_dir) / "model_training_feature_sets"
    summary_dir.mkdir(parents=True, exist_ok=True)

    if run_summary_rows:
        summary_df = pd.DataFrame(run_summary_rows).sort_values(
            by=["model", "feature_set"],
            kind="stable",
        )
        summary_csv = summary_dir / "feature_set_pricing_metric_comparison.csv"
        summary_df.to_csv(summary_csv, index=False)

        plotted_models = list(summary_df["model"].drop_duplicates())
        feature_sets = list(summary_df["feature_set"].drop_duplicates())
        x = np.arange(len(feature_sets))
        bar_width = 0.8 / max(1, len(plotted_models))

        pricing_metrics = [
            ("test_log_loss", "Test log loss (lower is better)"),
            ("test_brier_score", "Test Brier score (lower is better)"),
            ("test_ece_10_bins", "Test ECE (10 bins, lower is better)"),
        ]

        fig, axes = plt.subplots(1, len(pricing_metrics), figsize=(18, 6), sharex=True)
        for metric_idx, (metric_key, metric_label) in enumerate(pricing_metrics):
            ax = axes[metric_idx]
            for model_idx, model_name in enumerate(plotted_models):
                model_slice = (
                    summary_df[summary_df["model"] == model_name]
                    .set_index("feature_set")
                    .reindex(feature_sets)
                )
                y_vals = model_slice[metric_key].to_numpy()
                ax.bar(
                    x + (model_idx - (len(plotted_models) - 1) / 2.0) * bar_width,
                    y_vals,
                    width=bar_width,
                    label=model_name if metric_idx == 0 else None,
                )

            ax.set_title(metric_label)
            ax.set_xlabel("feature set")
            ax.set_xticks(x)
            ax.set_xticklabels(feature_sets, rotation=20, ha="right")
            ax.grid(axis="y", alpha=0.3)
        axes[0].set_ylabel("metric value")
        axes[0].legend(title="model")
        fig.tight_layout()

        comparison_plot = summary_dir / "feature_set_pricing_metric_comparison.png"
        fig.savefig(comparison_plot, bbox_inches="tight")
        plt.close(fig)

        for feature_set_name, run_manifest in manifests.items():
            artifacts = run_manifest.setdefault("artifacts", {})
            artifacts["feature_set_pricing_metric_comparison_csv"] = str(summary_csv)
            artifacts["feature_set_pricing_metric_comparison_plot"] = str(comparison_plot)
    return manifests
