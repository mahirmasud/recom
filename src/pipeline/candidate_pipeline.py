from __future__ import annotations

import logging
from typing import Any, Dict, List

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
        user_id: Any,
        stage_selection: StageSelection,
        stage_candidates: Dict[str, List[CandidateItem]],
        params: OrchestrationParams,
    ) -> Dict[str, Any]:
        retrieval = stage_candidates.get("retrieval", [])[: params.candidate_search_limit]
        sequential = stage_candidates.get("sequential", retrieval) if stage_selection.sequential else retrieval
        ranking = stage_candidates.get("ranking", sequential)

        merged = ranking
        filtered = self.rules.apply(merged, params)
        ranked = self.fusion.score(filtered, params)

        return {
            "user_id": user_id,
            "selected_models": stage_selection,
            "applied_rules": {
                "yield_limit": params.recommendation_yield_limit,
                "personalization_focus_pct": params.personalization_focus_pct,
                "discovery_factor_pct": params.discovery_factor_pct,
                "sponsored_promotions_pct": params.sponsored_promotions_pct,
                "diversity_index": params.diversity_index,
                "category_cap": params.category_capping_threshold,
                "recency_decay": params.recency_decay_coefficient,
            },
            "filtering_breakdown": {
                "retrieval_candidates": len(retrieval),
                "post_sequential": len(sequential),
                "post_ranking": len(ranking),
                "post_business_rules": len(filtered),
                "final": len(ranked),
            },
            "metrics": {
                "avg_final_score": sum(i.final_score for i in ranked) / max(len(ranked), 1),
                "unique_categories": len({i.category for i in ranked if i.category is not None}),
            },
            "recommendations": ranked,
        }
