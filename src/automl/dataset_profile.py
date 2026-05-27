from __future__ import annotations

from typing import Any, Dict

from src.utils.schemas import DatasetProfile


class DatasetProfileBuilder:
    """Builds normalized profile metrics from mapping metadata."""

    def build(self, mapping: Dict[str, Any]) -> DatasetProfile:
        identity = mapping.get("identity", [])
        content = mapping.get("content_features", [])
        signals = mapping.get("interaction_signals", {})
        provenance = mapping.get("mapping_provenance", {})
        ignored = mapping.get("ignore", [])
        entities = mapping.get("entity_mapping", {})

        confidence = provenance.get("field_confidence", {})
        reliability = sum(confidence.values()) / max(len(confidence), 1)
        feature_richness = len(content) / max((len(content) + len(identity) + len(ignored)), 1)
        has_timestamp = any("time" in f.lower() or "timestamp" in f.lower() for f in signals.keys())
        seq_availability = 1.0 if has_timestamp else 0.0
        signal_strength = sum(float(v) for v in signals.values()) / max(len(signals), 1)

        user_count = provenance.get("estimated_user_count", 1000)
        item_count = provenance.get("estimated_item_count", 1000)
        interaction_count = provenance.get("estimated_interaction_count", 10000)
        sparsity = 1.0 - (interaction_count / max(user_count * item_count, 1))
        sparsity = min(1.0, max(0.0, sparsity))

        return DatasetProfile(
            identity_fields=identity,
            content_features=content,
            interaction_signals=signals,
            feature_confidence=confidence,
            ignored_fields=ignored,
            entities=entities,
            has_timestamp=has_timestamp,
            sparsity=sparsity,
            feature_richness=feature_richness,
            sequence_availability=seq_availability,
            signal_strength=signal_strength,
            feature_reliability=reliability,
        )
