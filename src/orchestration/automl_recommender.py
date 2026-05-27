from __future__ import annotations

import json
from dataclasses import asdict
from typing import Any, Dict, List

from src.automl.dataset_profile import DatasetProfileBuilder
from src.automl.model_selector import AutoMLModelSelector
from src.orchestration.config_generator import RecBoleConfigGenerator
from src.pipeline.candidate_pipeline import CandidateGenerationPipeline
from src.utils.schemas import CandidateItem, OrchestrationParams


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
        result = self.pipeline.run(user_id, prepared["selection"], stage_candidates, params)
        result["stage_wise_selected_models"] = asdict(prepared["selection"])
        return result


def load_mapping(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
