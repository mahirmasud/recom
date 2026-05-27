import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.orchestration.automl_recommender import AutoMLRecommenderOrchestrator, load_mapping
from src.utils.schemas import CandidateItem, OrchestrationParams


def main():
    mapping = load_mapping("examples/sample_mapping.json")
    params = OrchestrationParams()
    orchestrator = AutoMLRecommenderOrchestrator()

    stage_candidates = {
        "retrieval": [
            CandidateItem(item_id="I1", category="Books", model_score=0.81, metadata={"recency": 0.9, "sponsored": True}),
            CandidateItem(item_id="I2", category="Books", model_score=0.79, metadata={"recency": 0.4, "sponsored": False}),
            CandidateItem(item_id="I3", category="Tech", model_score=0.75, metadata={"recency": 0.8, "sponsored": False}),
        ],
        "ranking": [
            CandidateItem(item_id="I1", category="Books", model_score=0.83, metadata={"recency": 0.9, "sponsored": True}),
            CandidateItem(item_id="I3", category="Tech", model_score=0.80, metadata={"recency": 0.8, "sponsored": False}),
            CandidateItem(item_id="I4", category="Home", model_score=0.71, metadata={"recency": 0.6, "sponsored": False}),
        ],
    }

    result = orchestrator.recommend("U100", mapping, params, stage_candidates, device="cpu")
    print("Selected Models:", result["stage_wise_selected_models"])
    print("Rules:", result["applied_rules"])
    print("Filtering:", result["filtering_breakdown"])
    print("Recommendations:", [(x.item_id, round(x.final_score, 4)) for x in result["recommendations"]])


if __name__ == "__main__":
    main()
