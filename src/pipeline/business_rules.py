from __future__ import annotations

from collections import Counter
from typing import List

import numpy as np

from src.utils.schemas import CandidateItem, OrchestrationParams


class BusinessRuleInjector:
    def apply(self, candidates: List[CandidateItem], params: OrchestrationParams) -> List[CandidateItem]:
        if not candidates:
            return candidates

        scores = np.array([c.model_score for c in candidates], dtype=float)
        threshold = float(np.quantile(scores, params.promotions_injection_percentile_threshold))
        promo_slots = int(params.recommendation_yield_limit * (params.sponsored_promotions_pct / 100.0))
        category_counter: Counter[str] = Counter()

        for c in candidates:
            c.recency_boost = np.exp(-params.recency_decay_coefficient * (1.0 - c.metadata.get("recency", 0.5)))
            c.sponsored_boost = 1.0 if (c.metadata.get("sponsored", False) and c.model_score >= threshold) else 0.0
            c.diversity_penalty = -category_counter[c.category] * params.diversity_index if c.category else 0.0
            c.business_score = c.recency_boost + c.sponsored_boost + c.diversity_penalty
            if c.category:
                category_counter[c.category] += 1

        # Enforce category cap.
        filtered: List[CandidateItem] = []
        category_counter.clear()
        sponsored_taken = 0
        for c in sorted(candidates, key=lambda x: x.model_score, reverse=True):
            if c.category and category_counter[c.category] >= params.category_capping_threshold:
                continue
            if c.sponsored_boost > 0 and sponsored_taken < promo_slots:
                sponsored_taken += 1
            filtered.append(c)
            if c.category:
                category_counter[c.category] += 1
            if len(filtered) >= params.candidate_search_limit:
                break
        return filtered
