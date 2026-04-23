"""Step 01: load raw input and normalize base parsing behavior."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any
import glob
import joblib
import pandas as pd

_DEFAULT_PARSE_CONFIG: dict[str, Any] = {
    "read_csv_kwargs": {},
    "trim_column_names": True,
    "drop_unnamed_columns": True,
}

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_DEFAULT_RAW_DATA_PATH = _PROJECT_ROOT / "data" / "raw_data.joblib"
_DEFAULT_CSV_GLOB = _PROJECT_ROOT / "data" / "csv_data" / "*.csv"


def _normalize_config(config: Mapping[str, Any] | None) -> dict[str, Any]:
    normalized = dict(_DEFAULT_PARSE_CONFIG)
    if config:
        normalized.update(dict(config))

    read_csv_kwargs = normalized.get("read_csv_kwargs")
    if read_csv_kwargs is None:
        normalized["read_csv_kwargs"] = {}
    elif not isinstance(read_csv_kwargs, Mapping):
        raise TypeError("config['read_csv_kwargs'] must be a mapping")
    else:
        normalized["read_csv_kwargs"] = dict(read_csv_kwargs)

    return normalized


def _load_from_path(path: str | Path, read_csv_kwargs: Mapping[str, Any]) -> pd.DataFrame:
    path_obj = Path(path)
    suffix = path_obj.suffix.lower()

    if suffix in {".csv", ".txt"}:
        return pd.read_csv(path_obj, **dict(read_csv_kwargs))
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(path_obj, **dict(read_csv_kwargs))
    if suffix in {".joblib"}:
        return joblib.load(path_obj, **dict(read_csv_kwargs))
    raise ValueError(
        "Unsupported file extension for load step. "
        "Supported: .csv, .txt, .parquet, .pq, .joblib"
    )


def _load_or_create_default_raw_data(read_csv_kwargs: Mapping[str, Any]) -> pd.DataFrame:
    """Load data/raw_data.joblib or build it from data/csv_data/*.csv."""

    if _DEFAULT_RAW_DATA_PATH.exists():
        return joblib.load(_DEFAULT_RAW_DATA_PATH)

    csv_files = sorted(glob.glob(str(_DEFAULT_CSV_GLOB)))
    if not csv_files:
        raise FileNotFoundError(
            "No default raw dataset found. Expected either "
            f"{_DEFAULT_RAW_DATA_PATH} or at least one CSV matching {_DEFAULT_CSV_GLOB}."
        )

    dfs: list[pd.DataFrame] = []
    for file_path in csv_files:
        dfs.append(pd.read_csv(file_path, low_memory=False, **dict(read_csv_kwargs)))

    merged = pd.concat(dfs, ignore_index=True)
    _DEFAULT_RAW_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(merged, _DEFAULT_RAW_DATA_PATH)
    return merged


def run(df_or_path: pd.DataFrame | str | Path | None, config: Mapping[str, Any] | None = None) -> pd.DataFrame:
    """Load raw input and apply minimal parse-level normalization.

    Parameters
    ----------
    df_or_path:
        A preloaded dataframe or a filesystem path to a tabular file.
    config:
        Optional settings. Supported keys:
        - read_csv_kwargs: kwargs forwarded to pandas readers.
        - trim_column_names: strip whitespace around column names.
        - drop_unnamed_columns: drop columns whose names start with ``Unnamed:``.

    Returns
    -------
    pd.DataFrame
        A new dataframe with normalized parsing behavior applied.
    """

    cfg = _normalize_config(config)

    if isinstance(df_or_path, pd.DataFrame):
        df = df_or_path.copy(deep=True)
    elif df_or_path is None:
        df = _load_or_create_default_raw_data(cfg["read_csv_kwargs"]).copy(deep=True)
    elif isinstance(df_or_path, (str, Path)):
        df = _load_from_path(df_or_path, cfg["read_csv_kwargs"]).copy(deep=True)
    else:
        raise TypeError("df_or_path must be a pandas DataFrame, str, Path, or None")

    if cfg.get("trim_column_names", True):
        df.columns = [str(col).strip() for col in df.columns]
    
    if cfg.get("drop_unnamed_columns", True):
        keep_cols = [col for col in df.columns if not str(col).startswith("Unnamed:")]
        df = df.loc[:, keep_cols].copy(deep=True)

    return df
