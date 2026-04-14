"""Model training experiments for tree-based classifiers."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

DEFAULT_MODEL_TRAINING_CONFIG: dict[str, Any] = {
    "enabled": True,
    "debug_leakage": False,
    "target_column": "team1_wins",
    "id_columns": ["event_id", "match_id", "match_date", "match_seq", "team1_player_id", "team2_player_id"],
    "date_column": "match_date",
    "depth_values": [1, 2, 3, 4, 5, 6, 8, 10, 12],
    "test_size": 0.2,
    "validation_size": 0.2,
    "output_subdir": "model_training",
    "random_state": 42,
    "rf_n_estimators": 300,
    "gbdt_n_estimators": 300,
    "gbdt_learning_rate": 0.05,
}


_LEAKAGE_PATTERN = re.compile(
    r"(winner|loser|result|score|outcome|post_|_post|after|stats|setscore|tiebreakscore|"
    r"breakpoints|returnpointswon|servicepointswon|totalpointswon|aces|doublefaults|sets[_\.\[])",
    flags=re.IGNORECASE,
)
_ANOMALY_SAFE_EXACT = {"anomaly_score", "anomaly_flag", "surface_anomaly_z"}
_ANOMALY_SAFE_PREFIXES = ("anom_",)
_ANOMALY_SAFE_TOKENS = ("_anomaly_",)


def _normalize_config(config: Mapping[str, Any] | None) -> dict[str, Any]:
    cfg = dict(DEFAULT_MODEL_TRAINING_CONFIG)
    if config:
        cfg.update(dict(config))

    if not isinstance(cfg.get("enabled"), bool):
        raise TypeError("model_training config['enabled'] must be a bool")
    if not isinstance(cfg.get("debug_leakage"), bool):
        raise TypeError("model_training config['debug_leakage'] must be a bool")

    for key in ("target_column", "output_subdir", "date_column"):
        value = cfg.get(key)
        if not isinstance(value, str) or not value.strip():
            raise TypeError(f"model_training config['{key}'] must be a non-empty string")

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

    for key in ("random_state", "rf_n_estimators", "gbdt_n_estimators"):
        cfg[key] = int(cfg[key])

    cfg["gbdt_learning_rate"] = float(cfg["gbdt_learning_rate"])
    return cfg


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


def _is_leakage_feature(col: str, target_column: str) -> bool:
    if col == target_column:
        return False
    if col in _ANOMALY_SAFE_EXACT:
        return False
    if col.startswith(_ANOMALY_SAFE_PREFIXES):
        return False
    if any(token in col for token in _ANOMALY_SAFE_TOKENS):
        return False
    return bool(_LEAKAGE_PATTERN.search(col))


def _select_training_feature_columns(df: pd.DataFrame, cfg: Mapping[str, Any]) -> list[str]:
    id_columns = set(cfg["id_columns"])
    target_column = str(cfg["target_column"])
    return [
        c
        for c in df.columns
        if c not in id_columns and c != target_column and not _is_leakage_feature(c, target_column)
    ]


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
        from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve
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

    feature_columns = _select_training_feature_columns(df, cfg)
    model_df = df.loc[:, [*feature_columns, target_column]].copy(deep=True)
    model_df = model_df.dropna(subset=[target_column]).reset_index(drop=True)

    y = model_df[target_column].astype(int)
    x = model_df.drop(columns=[target_column])
    if cfg["debug_leakage"]:
        _print_leakage_debug_info(x, y, target_column=target_column)

    split_df = x.copy(deep=True)
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

    model_specs = {
        "decision_tree": lambda depth: DecisionTreeClassifier(max_depth=depth, random_state=cfg["random_state"]),
        "random_forest": lambda depth: RandomForestClassifier(
            n_estimators=cfg["rf_n_estimators"],
            max_depth=depth,
            random_state=cfg["random_state"],
            n_jobs=-1,
        ),
        "gbdt": lambda depth: GradientBoostingClassifier(
            n_estimators=cfg["gbdt_n_estimators"],
            learning_rate=cfg["gbdt_learning_rate"],
            max_depth=depth,
            random_state=cfg["random_state"],
        ),
    }

    curve_rows: list[dict[str, float]] = []
    summary_rows: list[dict[str, Any]] = []

    plt.figure(figsize=(12, 8))
    for idx, (model_name, factory) in enumerate(model_specs.items(), start=1):
        curves: list[dict[str, float]] = []
        best_val_accuracy = -1.0
        best_depth = cfg["depth_values"][0]
        best_pipeline = None

        for depth in cfg["depth_values"]:
            pipeline = Pipeline(steps=[("prep", preprocessor), ("model", factory(depth))])
            pipeline.fit(x_train, y_train)
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

        test_accuracy = float(accuracy_score(y_test, test_pred))
        test_roc_auc = float(roc_auc_score(y_test, test_proba))
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
            }
        )

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

    manifest = {
        "split": {
            "train_rows": int(len(train_df)),
            "validation_rows": int(len(val_df)),
            "test_rows": int(len(test_df)),
            "target_column": target_column,
            "feature_count": int(len(feature_columns)),
        },
        "models": summary_rows,
        "artifacts": {
            "depth_curve_csv": str(output_path / "depth_accuracy_curves.csv"),
            "summary_csv": str(output_path / "model_summary_metrics.csv"),
            "depth_accuracy_plot": str(output_path / "depth_accuracy_curves.png"),
            "roc_curve_plots": [str(output_path / f"roc_curve__{name}.png") for name in model_specs],
        },
    }
    (output_path / "model_training_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    plt.tight_layout()
    plt.savefig(output_path / "depth_accuracy_curves.png", bbox_inches="tight")
    plt.close()

    return manifest
