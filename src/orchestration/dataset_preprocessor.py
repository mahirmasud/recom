from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

USER_CANDIDATES = ["user_id", "user", "uid", "customer_id", "userid", "userid"]
ITEM_CANDIDATES = ["item_id", "item", "iid", "product_id", "movie_id", "itemid", "productid"]
RATING_CANDIDATES = ["rating", "score", "stars", "rank", "preference"]
TIMESTAMP_CANDIDATES = ["timestamp", "time", "event_time", "datetime", "created_at", "ts"]


class DatasetPreprocessError(ValueError):
    pass


def _normalize(col: str) -> str:
    return col.strip().lower()


def _read_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in {".json", ".jsonl"}:
        return pd.read_json(path)
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    raise DatasetPreprocessError(f"Unsupported file format: {suffix}")


def _pick_column(df: pd.DataFrame, candidates: list[str], required: bool) -> str | None:
    norm_to_orig = {_normalize(c): c for c in df.columns}
    for cand in candidates:
        if cand in norm_to_orig:
            return norm_to_orig[cand]
    if required:
        raise DatasetPreprocessError(f"Unable to infer required column from candidates={candidates}")
    return None


def convert_to_recbole_inter(input_path: str, dataset_name: str | None = None) -> Path:
    src = Path(input_path)
    if not src.exists():
        raise DatasetPreprocessError(f"Input dataset not found: {input_path}")

    dataset = dataset_name or src.stem
    df = _read_table(src)
    if df.empty:
        raise DatasetPreprocessError("Input dataset is empty")

    user_col = _pick_column(df, USER_CANDIDATES, required=True)
    item_col = _pick_column(df, ITEM_CANDIDATES, required=True)
    rating_col = _pick_column(df, RATING_CANDIDATES, required=False)
    ts_col = _pick_column(df, TIMESTAMP_CANDIDATES, required=False)

    out = pd.DataFrame()
    out["user_id:token"] = df[user_col].astype(str)
    out["item_id:token"] = df[item_col].astype(str)

    if rating_col:
        out["rating:float"] = pd.to_numeric(df[rating_col], errors="coerce").fillna(1.0).astype(float)
    else:
        out["rating:float"] = 1.0

    if ts_col:
        out["timestamp:float"] = pd.to_numeric(df[ts_col], errors="coerce").fillna(0).astype(float)
    else:
        out["timestamp:float"] = 0.0

    out = out.dropna(subset=["user_id:token", "item_id:token"]).copy()
    out["user_id:token"] = out["user_id:token"].replace("", pd.NA)
    out["item_id:token"] = out["item_id:token"].replace("", pd.NA)
    out = out.dropna(subset=["user_id:token", "item_id:token"])

    out_dir = Path("dataset") / dataset
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{dataset}.inter"
    out.to_csv(out_path, sep="\t", index=False)
    logger.info("Converted dataset saved to %s", out_path)
    return out_path
