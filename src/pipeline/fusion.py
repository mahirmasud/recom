from __future__ import annotations

from typing import List

from src.utils.schemas import CandidateItem, OrchestrationParams


class FinalRankingFusionEngine:
    def score(self, items: List[CandidateItem], params: OrchestrationParams) -> List[CandidateItem]:
        personalization_weight = params.personalization_focus_pct / 100.0
        promotion_threshold = params.promotions_injection_percentile_threshold
        diversity_index = params.diversity_index

        for item in items:
            item.final_score = (
                (item.model_score * personalization_weight)
                + (item.recency_boost * params.recency_decay_coefficient)
                + (item.sponsored_boost * promotion_threshold)
                + (item.diversity_penalty * diversity_index)
            )
        return sorted(items, key=lambda x: x.final_score, reverse=True)[: params.recommendation_yield_limit]
