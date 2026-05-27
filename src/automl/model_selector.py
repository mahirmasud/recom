from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

from src.utils.schemas import DatasetProfile, OrchestrationParams, StageSelection


LOGGER = logging.getLogger(__name__)


class AutoMLModelSelector:
    MODEL_CATALOG = {
        "retrieval": ["LightGCN", "BPR", "ItemKNN"],
        "sequential": ["SASRec", "GRU4Rec"],
        "ranking": ["DeepFM", "xDeepFM", "WideDeep"],
        "reranking": ["LambdaMART", "MMR"],
    }

    DEFAULT_WEIGHT_MAP = {
        "LightGCN": {"sparsity": 0.9, "personalization": 0.7, "signal": 0.4},
        "BPR": {"signal": 0.8, "personalization": 0.6, "discovery": 0.2},
        "ItemKNN": {"reliability": 0.8, "discovery": 0.5, "sparsity": -0.2},
        "SASRec": {"seq": 1.0, "signal": 0.6, "personalization": 0.3},
        "GRU4Rec": {"seq": 0.9, "sparsity": 0.2, "personalization": 0.4},
        "DeepFM": {"richness": 0.8, "signal": 0.5, "reliability": 0.6},
        "xDeepFM": {"richness": 0.9, "signal": 0.7, "reliability": 0.4},
        "WideDeep": {"reliability": 0.7, "discovery": 0.5, "richness": 0.4},
        "LambdaMART": {"reliability": 0.7, "richness": 0.6, "signal": 0.6},
        "MMR": {"diversity": 1.0, "discovery": 0.7, "personalization": 0.2},
    }

    def __init__(self, weight_map: Optional[Dict[str, Dict[str, float]]] = None):
        self.weight_map = weight_map or self.DEFAULT_WEIGHT_MAP

    def _score_models(self, candidates: List[str], feature_vector: Dict[str, float], weight_map: Dict[str, Dict[str, float]]) -> List[Tuple[str, float]]:
        scored = []
        for model in candidates:
            score = 0.0
            for feature_name, feature_value in feature_vector.items():
                score += feature_value * weight_map.get(model, {}).get(feature_name, 0.0)
            scored.append((model, score))
        return sorted(scored, key=lambda x: x[1], reverse=True)

    def select(self, profile: DatasetProfile, params: OrchestrationParams) -> StageSelection:
        feature_vector = {
            "sparsity": profile.sparsity,
            "richness": profile.feature_richness,
            "seq": profile.sequence_availability,
            "signal": profile.signal_strength,
            "reliability": profile.feature_reliability,
            "personalization": params.personalization_focus_pct / 100.0,
            "discovery": params.discovery_factor_pct / 100.0,
            "diversity": params.diversity_index,
        }

        retrieval = self._score_models(self.MODEL_CATALOG["retrieval"], feature_vector, self.weight_map)[0][0]
        sequential: Optional[str] = None
        if profile.has_timestamp:
            sequential = self._score_models(self.MODEL_CATALOG["sequential"], feature_vector, self.weight_map)[0][0]
        ranking = self._score_models(self.MODEL_CATALOG["ranking"], feature_vector, self.weight_map)[0][0]
        reranking = self._score_models(self.MODEL_CATALOG["reranking"], feature_vector, self.weight_map)[0][0]

        LOGGER.info("Model selection: retrieval=%s sequential=%s ranking=%s reranking=%s", retrieval, sequential, ranking, reranking)
        return StageSelection(retrieval=retrieval, sequential=sequential, ranking=ranking, reranking=reranking)
