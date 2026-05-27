from __future__ import annotations

import logging
from typing import List

from src.pipeline.business_rules import BusinessRuleInjector
from src.pipeline.fusion import FinalRankingFusionEngine
from src.utils.schemas import CandidateItem, OrchestrationParams, StageSelection

LOGGER = logging.getLogger(__name__)


class CandidateGenerationPipeline:
    def __init__(self):
        self.rules = BusinessRuleInjector()
        self.fusion = FinalRankingFusionEngine()

    def run(
        self,
        user_id: str,
        stage_selection: StageSelection,
        retrieval_candidates: List[CandidateItem],
        params: OrchestrationParams,
    ) -> dict:
        retrieval = retrieval_candidates[: params.candidate_search_limit]

        LOGGER.info("Ranking stage started: model=%s", stage_selection.ranking)
        ranking = sorted(retrieval, key=lambda x: x.model_score, reverse=True)

        LOGGER.info("Reranking stage applied: %s", stage_selection.reranking)
        filtered = self.rules.apply(ranking, params)
        ranked = self.fusion.score(filtered, params)

        return {
            "user_id": user_id,
            "selected_models": stage_selection,
            "candidate_generation_method": "dynamic",
            "applied_rules": {
                "diversity_constraints": params.diversity_index,
                "category_cap": params.category_capping_threshold,
                "recency_decay": params.recency_decay_coefficient,
                "yield_limit": params.recommendation_yield_limit,
            },
            "filtering_breakdown": {
                "retrieval_candidates": len(retrieval),
                "post_ranking": len(ranking),
                "post_business_rules": len(filtered),
                "final": len(ranked),
            },
            "recommendations": ranked,
        }
