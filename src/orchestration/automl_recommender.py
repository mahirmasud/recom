from __future__ import annotations

import json
import logging
from dataclasses import asdict
from typing import Any, Dict, List, Optional

from src.orchestration.prepare import PreparationService
from src.pipeline.candidate_generator import CandidateGenerator
from src.pipeline.candidate_pipeline import CandidateGenerationPipeline
from src.utils.schemas import CandidateItem, OrchestrationParams

LOGGER = logging.getLogger(__name__)


class AutoMLRecommenderOrchestrator:
    def __init__(self):
        self.preparation = PreparationService()
        self.config_generator = self.preparation.config_generator
        self.candidate_generator = CandidateGenerator()
        self.pipeline = CandidateGenerationPipeline()

    def prepare(self, mapping_json: Dict[str, Any], params: OrchestrationParams, device: str = "cpu") -> Dict[str, Any]:
        prepared = self.preparation.prepare(mapping_json, params, device=device)
        return {"profile": prepared.profile, "selection": prepared.selection, "configs": prepared.configs}

    def recommend(
        self,
        user_id: Any,
        mapping_json: Dict[str, Any],
        params: OrchestrationParams,
        static_candidates: Optional[List[CandidateItem]] = None,
        candidates_mode: str = "dynamic",
        device: str = "cpu",
    ) -> Dict[str, Any]:
        prepared = self.preparation.prepare(mapping_json, params, device=device)
        selected = prepared.selection
        LOGGER.info("AutoML selected pipeline: %s", asdict(selected))

        dynamic = self.candidate_generator.generate_dynamic(user_id, selected.retrieval, mapping_json, params)
        retrieval_candidates = dynamic
        if candidates_mode == "static" and static_candidates:
            retrieval_candidates = self.candidate_generator.merge_static_seed(dynamic, static_candidates, params.candidate_search_limit)

        result = self.pipeline.run(user_id, selected, retrieval_candidates, params)
        result["selected_models"] = {
            "retrieval": selected.retrieval,
            "ranking": selected.ranking,
            "reranking": selected.reranking,
        }
        result["business_rules_summary"] = result["applied_rules"]
        result["recommendations"] = [asdict(item) for item in result["recommendations"]]
        return result


def load_mapping(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
