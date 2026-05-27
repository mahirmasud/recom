from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class DatasetProfile:
    identity_fields: List[str]
    content_features: List[str]
    interaction_signals: Dict[str, float]
    feature_confidence: Dict[str, float]
    ignored_fields: List[str]
    entities: Dict[str, List[str]] = field(default_factory=dict)
    has_timestamp: bool = False
    sparsity: float = 1.0
    feature_richness: float = 0.0
    sequence_availability: float = 0.0
    signal_strength: float = 0.0
    feature_reliability: float = 0.0


@dataclass
class OrchestrationParams:
    recommendation_yield_limit: int = 20
    personalization_focus_pct: float = 70.0
    discovery_factor_pct: float = 20.0
    sponsored_promotions_pct: float = 10.0
    diversity_index: float = 0.4
    retrieval_batch_size: int = 512
    candidate_search_limit: int = 1000
    recency_decay_coefficient: float = 0.7
    category_capping_threshold: int = 3
    promotions_injection_percentile_threshold: float = 0.75


@dataclass
class StageSelection:
    retrieval: str
    sequential: Optional[str]
    ranking: str
    reranking: str


@dataclass
class CandidateItem:
    item_id: Any
    category: Optional[str]
    model_score: float
    recency_boost: float = 0.0
    sponsored_boost: float = 0.0
    diversity_penalty: float = 0.0
    business_score: float = 0.0
    final_score: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
