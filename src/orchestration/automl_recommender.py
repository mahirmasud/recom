from __future__ import annotations

import json
import logging
from dataclasses import asdict
from typing import Any, Dict, List

from src.automl.dataset_profile import DatasetProfileBuilder
from src.automl.model_selector import AutoMLModelSelector
from src.orchestration.config_generator import RecBoleConfigGenerator
from src.pipeline.candidate_pipeline import CandidateGenerationPipeline
from src.utils.schemas import CandidateItem, OrchestrationParams

LOGGER = logging.getLogger(__name__)


class AutoMLRecommenderOrchestrator:
    def __init__(self):
        self.profile_builder = DatasetProfileBuilder()
        self.model_selector = AutoMLModelSelector()
        self.config_generator = RecBoleConfigGenerator()
        self.pipeline = CandidateGenerationPipeline()

    def prepare(self, mapping_json: Dict[str, Any], params: OrchestrationParams, device: str = "cpu") -> Dict[str, Any]:
        profile = self.profile_builder.build(mapping_json)
        selected = self.model_selector.select(profile, params)
        configs = self.config_generator.generate(selected, profile, params, device=device)
        return {"profile": profile, "selection": selected, "configs": configs}

    def recommend(
        self,
        user_id: Any,
        mapping_json: Dict[str, Any],
        params: OrchestrationParams,
        stage_candidates: Dict[str, List[CandidateItem]],
        device: str = "cpu",
    ) -> Dict[str, Any]:
        prepared = self.prepare(mapping_json, params, device=device)
        LOGGER.info("Running recommendation for user=%s using models=%s", user_id, asdict(prepared["selection"]))
        result = self.pipeline.run(user_id, prepared["selection"], stage_candidates, params)
        result["stage_wise_selected_models"] = asdict(prepared["selection"])
        result["recommendations"] = [asdict(item) for item in result["recommendations"]]
        return result

    def recommend_batch(
        self,
        user_stage_candidates: Dict[Any, Dict[str, List[CandidateItem]]],
        mapping_json: Dict[str, Any],
        params: OrchestrationParams,
        device: str = "cpu",
    ) -> List[Dict[str, Any]]:
        prepared = self.prepare(mapping_json, params, device=device)
        outputs: List[Dict[str, Any]] = []
        for user_id, stage_candidates in user_stage_candidates.items():
            result = self.pipeline.run(user_id, prepared["selection"], stage_candidates, params)
            result["stage_wise_selected_models"] = asdict(prepared["selection"])
            result["recommendations"] = [asdict(item) for item in result["recommendations"]]
            outputs.append(result)
        return outputs


def load_mapping(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
