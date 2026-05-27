from __future__ import annotations

import hashlib
import logging
from typing import Any, Dict, List, Optional

from src.utils.schemas import CandidateItem, OrchestrationParams

LOGGER = logging.getLogger(__name__)


class CandidateGenerator:
    """Single entry-point for candidate generation via retrieval models."""

    def _estimated_item_count(self, mapping_json: Dict[str, Any]) -> int:
        provenance = mapping_json.get("mapping_provenance", {})
        return int(provenance.get("estimated_item_count", 1000))

    def _seed(self, user_id: Any, retrieval_model: str) -> int:
        key = f"{user_id}:{retrieval_model}".encode("utf-8")
        return int(hashlib.md5(key).hexdigest()[:8], 16)

    def generate_dynamic(
        self,
        user_id: Any,
        retrieval_model: str,
        mapping_json: Dict[str, Any],
        params: OrchestrationParams,
    ) -> List[CandidateItem]:
        LOGGER.info("Retrieval stage started: model=%s", retrieval_model)
        total_items = self._estimated_item_count(mapping_json)
        top_k = min(params.candidate_search_limit, max(10, total_items // 10))
        base = self._seed(user_id, retrieval_model)
        categories = mapping_json.get("content_features", []) or ["default"]

        candidates: List[CandidateItem] = []
        for i in range(top_k):
            item_num = ((base + (i * 7919)) % total_items) + 1
            score = max(0.01, 1.0 - (i / max(top_k, 1)))
            category = str(categories[i % len(categories)])
            candidates.append(
                CandidateItem(
                    item_id=f"I{item_num}",
                    category=category,
                    model_score=score,
                    metadata={
                        "global_score": max(0.01, score * 0.9),
                        "recency": ((base + i) % 100) / 100.0,
                        "sponsored": ((base + i) % 19 == 0),
                        "source": "dynamic_retrieval",
                    },
                )
            )
        LOGGER.info("Generated %s candidates for user %s", len(candidates), user_id)
        return candidates

    def merge_static_seed(
        self,
        dynamic_candidates: List[CandidateItem],
        static_candidates: Optional[List[CandidateItem]],
        max_candidates: int,
    ) -> List[CandidateItem]:
        if not static_candidates:
            return dynamic_candidates

        by_item = {str(c.item_id): c for c in dynamic_candidates}
        for c in static_candidates:
            if str(c.item_id) not in by_item:
                c.metadata = {**c.metadata, "source": "static_seed"}
                dynamic_candidates.append(c)

        return dynamic_candidates[:max_candidates]
