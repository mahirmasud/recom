from __future__ import annotations

import argparse
import json
import logging
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List

from src.automl.model_selector import AutoMLModelSelector
from src.automl.tournament import AutoMLTournamentEngine
from src.orchestration.automl_recommender import AutoMLRecommenderOrchestrator, load_mapping
from src.orchestration.training_orchestrator import TrainingOrchestrator
from src.utils.schemas import CandidateItem, OrchestrationParams

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s :: %(message)s")
LOGGER = logging.getLogger("main")


def load_params(path: str | None) -> OrchestrationParams:
    if not path:
        return OrchestrationParams()
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    return OrchestrationParams(**raw)


def load_stage_candidates(path: str) -> Dict[str, List[CandidateItem]]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    out: Dict[str, List[CandidateItem]] = {}
    for stage, items in raw.items():
        out[stage] = [CandidateItem(**item) for item in items]
    return out


def cmd_prepare(args: argparse.Namespace) -> None:
    orchestrator = AutoMLRecommenderOrchestrator()
    mapping = load_mapping(args.mapping)
    params = load_params(args.params)
    prepared = orchestrator.prepare(mapping, params, device=args.device)

    print("Dataset Profile:")
    print(json.dumps(asdict(prepared["profile"]), indent=2))
    print("\nSelected Models:")
    print(json.dumps(asdict(prepared["selection"]), indent=2))

    if args.export_yaml_dir:
        emitted = orchestrator.config_generator.dump_yaml(prepared["configs"], args.export_yaml_dir)
        print("\nGenerated config YAML:")
        print(json.dumps(emitted, indent=2))


def cmd_recommend(args: argparse.Namespace) -> None:
    orchestrator = AutoMLRecommenderOrchestrator()
    mapping = load_mapping(args.mapping)
    params = load_params(args.params)
    candidates = load_stage_candidates(args.candidates)
    result = orchestrator.recommend(args.user_id, mapping, params, candidates, device=args.device)
    print(json.dumps(result, indent=2, default=str))


def cmd_train(args: argparse.Namespace) -> None:
    orchestrator = AutoMLRecommenderOrchestrator()
    mapping = load_mapping(args.mapping)
    params = load_params(args.params)
    prepared = orchestrator.prepare(mapping, params, device=args.device)

    selector = AutoMLModelSelector()
    catalog = selector.MODEL_CATALOG
    trainer = TrainingOrchestrator()
    tournament = AutoMLTournamentEngine(trainer)

    stage_winners: Dict[str, Dict[str, Any]] = {}
    for stage in ["retrieval", "ranking"] + (["sequential"] if prepared["selection"].sequential else []):
        models = catalog[stage]
        stage_cfgs = {m: {**prepared["configs"].get(stage, {}), "model": m} for m in models}
        result = tournament.run_stage(models, stage_cfgs, args.dataset)
        stage_winners[stage] = asdict(result)

    print(json.dumps(stage_winners, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="AutoML RecBole Recommender CLI")
    sub = parser.add_subparsers(dest="cmd", required=True)

    def add_common(p: argparse.ArgumentParser) -> None:
        p.add_argument("--mapping", required=True, help="Path to mapping JSON")
        p.add_argument("--params", default=None, help="Path to orchestration params JSON")
        p.add_argument("--device", default="cpu", choices=["cpu", "cuda"])

    p_prepare = sub.add_parser("prepare", help="Build profile, select models, generate configs")
    add_common(p_prepare)
    p_prepare.add_argument("--export-yaml-dir", default=None)
    p_prepare.set_defaults(func=cmd_prepare)

    p_rec = sub.add_parser("recommend", help="Run full recommendation pipeline")
    add_common(p_rec)
    p_rec.add_argument("--candidates", required=True, help="Stage candidates JSON")
    p_rec.add_argument("--user-id", required=True)
    p_rec.set_defaults(func=cmd_recommend)

    p_train = sub.add_parser("train", help="Run AutoML tournament training by stage")
    add_common(p_train)
    p_train.add_argument("--dataset", required=True, help="RecBole dataset name")
    p_train.set_defaults(func=cmd_train)

    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()
    args.func(args)
