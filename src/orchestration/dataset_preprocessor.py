from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)


class DatasetPreprocessError(ValueError):
    pass


def _read_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in {".json", ".jsonl"}:
        return pd.read_json(path)
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    raise DatasetPreprocessError(f"Unsupported file format: {suffix}")


def _require_config_field(config: dict[str, Any], field_name: str) -> str:
    value = config.get(field_name)
    if not value or not isinstance(value, str):
        raise DatasetPreprocessError(f"{field_name} must be provided as a non-empty string in config")
    return value


def _validate_column_exists(df: pd.DataFrame, column_name: str, config_key: str) -> None:
    if column_name not in df.columns:
        raise DatasetPreprocessError(f"{config_key} not found in dataset: '{column_name}'")


def convert_to_recbole_inter(
    input_path: str,
    config: dict[str, Any] | None = None,
    dataset_name: str | None = None,
) -> Path:
    src = Path(input_path)
    if not src.exists():
        raise DatasetPreprocessError(f"Input dataset not found: {input_path}")

    if config is None:
        config = {"USER_ID_FIELD": "user_id", "ITEM_ID_FIELD": "item_id", "RATING_FIELD": "rating", "TIME_FIELD": "timestamp"}
    if not isinstance(config, dict):
        raise DatasetPreprocessError("config must be a dictionary")

    user_col = _require_config_field(config, "USER_ID_FIELD")
    item_col = _require_config_field(config, "ITEM_ID_FIELD")
    rating_col = config.get("RATING_FIELD")
    time_col = config.get("TIME_FIELD")

    if rating_col is not None and (not isinstance(rating_col, str) or not rating_col):
        raise DatasetPreprocessError("RATING_FIELD must be a non-empty string when provided")
    if time_col is not None and (not isinstance(time_col, str) or not time_col):
        raise DatasetPreprocessError("TIME_FIELD must be a non-empty string when provided")

    df = _read_table(src)
    if df.empty:
        raise DatasetPreprocessError("Input dataset is empty")

    _validate_column_exists(df, user_col, "USER_ID_FIELD")
    _validate_column_exists(df, item_col, "ITEM_ID_FIELD")
    if rating_col is not None and rating_col not in df.columns:
        logger.warning("RATING_FIELD '%s' not found; defaulting ratings to 1.0", rating_col)
        rating_col = None
    if time_col is not None and time_col not in df.columns:
        logger.warning("TIME_FIELD '%s' not found; defaulting timestamps to 0.0", time_col)
        time_col = None

    dataset = dataset_name or src.stem

    out = pd.DataFrame()
    out["user_id:token"] = df[user_col].astype(str)
    out["item_id:token"] = df[item_col].astype(str)

    if rating_col is not None:
        out["rating:float"] = pd.to_numeric(df[rating_col], errors="coerce").fillna(1.0).astype(float)
    else:
        out["rating:float"] = 1.0

    if time_col is not None:
        out["timestamp:float"] = pd.to_numeric(df[time_col], errors="coerce").fillna(0.0).astype(float)
    else:
        out["timestamp:float"] = 0.0

    out["user_id:token"] = out["user_id:token"].replace("", pd.NA)
    out["item_id:token"] = out["item_id:token"].replace("", pd.NA)
    out = out.dropna(subset=["user_id:token", "item_id:token"]).copy()

    out_dir = Path("dataset") / dataset
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{dataset}.inter"
    out.to_csv(out_path, sep="\t", index=False)
    logger.info("Converted dataset saved to %s", out_path)
    return out_path
