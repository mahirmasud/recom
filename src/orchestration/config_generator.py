from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import yaml

from src.utils.schemas import DatasetProfile, OrchestrationParams, StageSelection


class RecBoleConfigGenerator:
    def __init__(self, base: Dict[str, Any] | None = None):
        self.base = base or {}

    def _base_config(self, params: OrchestrationParams, device: str) -> Dict[str, Any]:
        return {
            "epochs": 50,
            "train_batch_size": params.retrieval_batch_size,
            "eval_batch_size": params.retrieval_batch_size,
            "topk": [10, params.recommendation_yield_limit],
            "metrics": ["Recall", "NDCG", "Hit"],
            "valid_metric": "NDCG@10",
            "device": device,
        }

    def generate(self, selection: StageSelection, profile: DatasetProfile, params: OrchestrationParams, device: str = "cpu") -> Dict[str, Dict[str, Any]]:
        base = self._base_config(params, device)
        configs: Dict[str, Dict[str, Any]] = {}
        for stage_name, model_name in {
            "retrieval": selection.retrieval,
            "sequential": selection.sequential,
            "ranking": selection.ranking,
        }.items():
            if model_name is None:
                continue
            cfg = dict(base)
            cfg.update(self.base)
            cfg["model"] = model_name
            cfg["neg_sampling"] = {"uniform": 1}
            cfg["USER_ID_FIELD"] = profile.identity_fields[0] if profile.identity_fields else "user_id"
            cfg["ITEM_ID_FIELD"] = profile.identity_fields[1] if len(profile.identity_fields) > 1 else "item_id"
            configs[stage_name] = cfg
        return configs

    def dump_yaml(self, configs: Dict[str, Dict[str, Any]], output_dir: str) -> Dict[str, str]:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        emitted: Dict[str, str] = {}
        for stage, cfg in configs.items():
            p = out / f"{stage}.yaml"
            p.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
            emitted[stage] = str(p)
        return emitted
