from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

from src.automl.dataset_profile import DatasetProfileBuilder
from src.automl.model_selector import AutoMLModelSelector
from src.orchestration.config_generator import RecBoleConfigGenerator
from src.utils.schemas import OrchestrationParams


@dataclass
class PreparedArtifacts:
    profile: Any
    selection: Any
    configs: Dict[str, Any]
    retriever_model: str


class PreparationService:
    def __init__(self):
        self.profile_builder = DatasetProfileBuilder()
        self.model_selector = AutoMLModelSelector()
        self.config_generator = RecBoleConfigGenerator()

    def prepare(self, mapping_json: Dict[str, Any], params: OrchestrationParams, device: str = "cpu") -> PreparedArtifacts:
        profile = self.profile_builder.build(mapping_json)
        selection = self.model_selector.select(profile, params)
        configs = self.config_generator.generate(selection, profile, params, device=device)
        return PreparedArtifacts(profile=profile, selection=selection, configs=configs, retriever_model=selection.retrieval)
