from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

from src.orchestration.training_orchestrator import TrainingOrchestrator


@dataclass
class TournamentResult:
    best_model: str
    best_score: float
    all_scores: Dict[str, float]


class AutoMLTournamentEngine:
    def __init__(self, trainer: TrainingOrchestrator):
        self.trainer = trainer

    @staticmethod
    def _objective(metrics: Dict[str, float], k: int = 10) -> float:
        recall = metrics.get(f"Recall@{k}", 0.0)
        ndcg = metrics.get(f"NDCG@{k}", 0.0)
        hit = metrics.get(f"Hit@{k}", 0.0)
        return 0.4 * recall + 0.4 * ndcg + 0.2 * hit

    def run_stage(self, stage_models: List[str], stage_config_map: Dict[str, Dict[str, Any]], dataset_name: str) -> TournamentResult:
        scores: Dict[str, float] = {}
        for model in stage_models:
            cfg = stage_config_map[model]
            res = self.trainer.train(model=model, config_dict=cfg, dataset_name=dataset_name)
            metric_map = res.get("best_valid_result", {}) if isinstance(res, dict) else {}
            scores[model] = self._objective(metric_map)
        best_model, best_score = sorted(scores.items(), key=lambda x: x[1], reverse=True)[0]
        return TournamentResult(best_model=best_model, best_score=best_score, all_scores=scores)
