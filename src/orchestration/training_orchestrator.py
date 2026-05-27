from __future__ import annotations

import logging
from typing import Any, Dict

LOGGER = logging.getLogger(__name__)

try:
    from recbole.quick_start import run_recbole
except ImportError:  # pragma: no cover
    run_recbole = None


class TrainingOrchestrator:
    def train(self, model: str, config_dict: Dict[str, Any], dataset_name: str) -> Dict[str, Any]:
        if run_recbole is None:
            raise RuntimeError("RecBole is not installed. Please install dependencies first.")
        LOGGER.info("Training model=%s dataset=%s", model, dataset_name)
        result = run_recbole(model=model, dataset=dataset_name, config_dict=config_dict, saved=True)
        return result
