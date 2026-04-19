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

    feature_columns = [c for c in df.columns if c not in set(cfg["id_columns"]) and c != target_column]
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

    curve_rows: list[dict[str, float]] = []
    summary_rows: list[dict[str, Any]] = []

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
        _training_log(
            f"done model={model_name} best_depth={best_depth} "
            f"val_acc={val_accuracy_at_best:.4f} test_acc={test_accuracy:.4f} test_auc={test_roc_auc:.4f}"
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
            test_accuracy = model_metrics.get("test_accuracy")
            if isinstance(model_name, str) and isinstance(test_accuracy, (int, float)):
                run_summary_rows.append(
                    {
                        "feature_set": feature_set_name,
                        "model": model_name,
                        "test_accuracy": float(test_accuracy),
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
        summary_csv = summary_dir / "feature_set_accuracy_comparison.csv"
        summary_df.to_csv(summary_csv, index=False)

        plotted_models = list(summary_df["model"].drop_duplicates())
        feature_sets = list(summary_df["feature_set"].drop_duplicates())
        x = np.arange(len(feature_sets))
        bar_width = 0.8 / max(1, len(plotted_models))

        fig, ax = plt.subplots(figsize=(12, 6))
        for idx, model_name in enumerate(plotted_models):
            model_slice = (
                summary_df[summary_df["model"] == model_name]
                .set_index("feature_set")
                .reindex(feature_sets)
            )
            y_vals = model_slice["test_accuracy"].fillna(0.0).to_numpy()
            ax.bar(
                x + (idx - (len(plotted_models) - 1) / 2.0) * bar_width,
                y_vals,
                width=bar_width,
                label=model_name,
            )

        ax.set_title("Feature-set experiment: test accuracy comparison")
        ax.set_xlabel("feature set")
        ax.set_ylabel("test accuracy")
        ax.set_ylim(0.0, 1.0)
        ax.set_xticks(x)
        ax.set_xticklabels(feature_sets, rotation=20, ha="right")
        ax.grid(axis="y", alpha=0.3)
        ax.legend(title="model")
        fig.tight_layout()

        comparison_plot = summary_dir / "feature_set_accuracy_comparison.png"
        fig.savefig(comparison_plot, bbox_inches="tight")
        plt.close(fig)

        for feature_set_name, run_manifest in manifests.items():
            artifacts = run_manifest.setdefault("artifacts", {})
            artifacts["feature_set_accuracy_comparison_csv"] = str(summary_csv)
            artifacts["feature_set_accuracy_comparison_plot"] = str(comparison_plot)
    return manifests
