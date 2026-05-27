from __future__ import annotations

import argparse
import json
import logging
from dataclasses import asdict
from pathlib import Path
from typing import List

from src.orchestration.automl_recommender import AutoMLRecommenderOrchestrator, load_mapping
from src.orchestration.training_orchestrator import TrainingOrchestrator
from src.orchestration.dataset_preprocessor import convert_to_recbole_inter
from src.automl.model_selector import AutoMLModelSelector
from src.automl.tournament import AutoMLTournamentEngine
from src.utils.schemas import CandidateItem, OrchestrationParams

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s :: %(message)s")


def load_params(path: str | None) -> OrchestrationParams:
    if not path:
        return OrchestrationParams()
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    return OrchestrationParams(**raw)


def load_candidates(path: str) -> List[CandidateItem]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(raw, dict) and "retrieval" in raw:
        raw = raw["retrieval"]
    return [CandidateItem(**item) for item in raw]


def cmd_prepare(args: argparse.Namespace) -> None:
    orchestrator = AutoMLRecommenderOrchestrator()
    mapping = load_mapping(args.mapping)
    params = load_params(args.params)
    prepared = orchestrator.prepare(mapping, params, device=args.device)
    print(json.dumps(asdict(prepared["profile"]), indent=2))
    print(json.dumps(asdict(prepared["selection"]), indent=2))


def cmd_recommend(args: argparse.Namespace) -> None:
    orchestrator = AutoMLRecommenderOrchestrator()
    mapping = load_mapping(args.mapping)
    params = load_params(args.params)
    static_candidates = load_candidates(args.candidates) if args.candidates else None
    result = orchestrator.recommend(
        args.user_id,
        mapping,
        params,
        static_candidates=static_candidates,
        candidates_mode=args.candidates_mode,
        device=args.device,
    )
    print(json.dumps(result, indent=2, default=str))


def cmd_preprocess(args: argparse.Namespace) -> None:
    output_path = convert_to_recbole_inter(args.input, args.dataset_name)
    dataset_name = args.dataset_name or Path(args.input).stem
    print(f"Converted dataset:\n{output_path}")
    print("\nTraining command:")
    print(
        f"python main.py train --mapping examples/sample_mapping.json --dataset {dataset_name} --device cpu"
    )


def cmd_train(args: argparse.Namespace) -> None:
    orchestrator = AutoMLRecommenderOrchestrator()
    mapping = load_mapping(args.mapping)
    params = load_params(args.params)
    prepared = orchestrator.prepare(mapping, params, device=args.device)

    selector = AutoMLModelSelector()
    trainer = TrainingOrchestrator()
    tournament = AutoMLTournamentEngine(trainer)
    stage_winners = {}
    for stage in ["retrieval", "ranking"]:
        models = selector.MODEL_CATALOG[stage]
        stage_cfgs = {m: {**prepared["configs"].get(stage, {}), "model": m} for m in models}
        stage_winners[stage] = asdict(tournament.run_stage(models, stage_cfgs, args.dataset))
    print(json.dumps(stage_winners, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="AutoML RecBole Recommender CLI")
    sub = parser.add_subparsers(dest="cmd", required=True)

    def add_common(p: argparse.ArgumentParser) -> None:
        p.add_argument("--mapping", required=True)
        p.add_argument("--params", default=None)
        p.add_argument("--device", default="cpu", choices=["cpu", "cuda"])

    p_prepare = sub.add_parser("prepare")
    add_common(p_prepare)
    p_prepare.set_defaults(func=cmd_prepare)

    p_rec = sub.add_parser("recommend")
    add_common(p_rec)
    p_rec.add_argument("--user-id", required=True)
    p_rec.add_argument("--candidates", default=None, help="Optional static seed candidates for debug only")
    p_rec.add_argument("--candidates_mode", default="dynamic", choices=["dynamic", "static"])
    p_rec.set_defaults(func=cmd_recommend)

    p_pre = sub.add_parser("preprocess")
    p_pre.add_argument("--input", required=True, help="Raw dataset path (.csv/.json/.xlsx/.xls)")
    p_pre.add_argument("--dataset-name", default=None, help="Optional RecBole dataset folder name")
    p_pre.set_defaults(func=cmd_preprocess)

    p_train = sub.add_parser("train")
    add_common(p_train)
    p_train.add_argument("--dataset", required=True)
    p_train.set_defaults(func=cmd_train)
    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()
    args.func(args)
