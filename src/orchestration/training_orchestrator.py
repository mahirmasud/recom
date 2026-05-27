from __future__ import annotations

import logging
import os
import sys
from typing import Any, Dict

LOGGER = logging.getLogger(__name__)


def _load_recbole() -> Any:
    """Import RecBole lazily so env changes are reflected at runtime."""
    try:
        from recbole.quick_start import run_recbole
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "RecBole import failed in the active Python environment. "
            f"python={sys.executable} cwd={os.getcwd()} error={exc}"
        ) from exc
    return run_recbole


class TrainingOrchestrator:
    def train(self, model: str, config_dict: Dict[str, Any], dataset_name: str) -> Dict[str, Any]:
        run_recbole = _load_recbole()
        LOGGER.info("Training model=%s dataset=%s", model, dataset_name)
        result = run_recbole(model=model, dataset=dataset_name, config_dict=config_dict, saved=True)
        return result
