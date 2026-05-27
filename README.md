# AutoML RecBole Recommender Orchestration

Production-grade, modular AutoML recommender orchestration system built on top of RecBole.

## Project Structure

- `src/automl/`: dataset profiling, model selection, tournament engine
- `src/orchestration/`: config generation, training orchestration, system facade
- `src/pipeline/`: business rules, fusion, candidate pipeline
- `src/models/`: model integration extension point
- `src/utils/`: shared schemas
- `examples/`: runnable example input/output

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run Inference Demo

```bash
python examples/run_example.py
```

## Run Training (RecBole)

Use `AutoMLRecommenderOrchestrator.prepare(...)` to produce stage configs, then feed configs into `TrainingOrchestrator` and `AutoMLTournamentEngine`.

```python
from src.orchestration.automl_recommender import AutoMLRecommenderOrchestrator
from src.orchestration.training_orchestrator import TrainingOrchestrator
from src.automl.tournament import AutoMLTournamentEngine
```

## Orchestration Parameters

Edit `OrchestrationParams` values to control:
- recommendation yield
- personalization vs discovery
- sponsored promotion share
- diversity constraints
- retrieval batch sizes
- search limits
- recency decay
- category caps

## Pipeline Stages

1. **Dataset Profile Builder** derives sparsity, feature richness, sequence readiness, signal strength, and reliability.
2. **AutoML Model Selector** dynamically scores candidate models for retrieval, sequential, ranking, and reranking.
3. **RecBole Config Generator** creates runtime config dictionaries for selected models.
4. **Tournament Engine** can run candidate model competitions using Recall@K, NDCG@K, and HitRate.
5. **Candidate Pipeline** executes `Retrieval -> Sequential (optional) -> Ranking -> Reranking`.
6. **Business Rule Injection** applies yield, discovery, promotions, diversity, category caps, and recency rules.
7. **Final Fusion Engine** calculates final score composition and ranking output.

## Output Per User Request

- final ranked recommendation list
- stage-wise selected models
- applied business rules summary
- candidate filtering breakdown
- diversity and score metrics

## Notes

- No hardcoded single-model choice: scoring uses dataset profile + runtime orchestration params.
- CPU/GPU switch supported via `device` in prepare/recommend paths.
- Pipeline supports batch inference by passing per-user candidate batches.
