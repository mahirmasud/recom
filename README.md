# AutoML RecBole Recommender Orchestration

Production-grade, modular AutoML recommender orchestration system built on top of RecBole.

## Project Structure

- `src/automl/`: dataset profiling, dynamic model selection, AutoML tournament
- `src/orchestration/`: config generation (dict/YAML), training orchestration, top-level system facade
- `src/pipeline/`: candidate processing, business rules, final fusion scoring
- `src/models/`: extension hooks for custom model wrappers
- `src/utils/`: schemas and shared dataclasses
- `examples/`: runnable end-to-end demo

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Training Flow

```python
from src.orchestration.automl_recommender import AutoMLRecommenderOrchestrator
from src.orchestration.training_orchestrator import TrainingOrchestrator
from src.automl.tournament import AutoMLTournamentEngine

orchestrator = AutoMLRecommenderOrchestrator()
prepared = orchestrator.prepare(mapping_json, params, device="cuda")

trainer = TrainingOrchestrator()
tournament = AutoMLTournamentEngine(trainer)
```

### Optional YAML export

```python
yaml_paths = orchestrator.config_generator.dump_yaml(prepared["configs"], "./generated_configs")
```

## Inference Flow

```bash
python examples/run_example.py
```

Inference output includes:
1. ranked recommendation list
2. stage-wise selected models
3. applied business rules summary
4. candidate filtering breakdown
5. diversity and scoring metrics

## Orchestration Parameters

Tune `OrchestrationParams` for:
- `recommendation_yield_limit`
- `personalization_focus_pct`
- `discovery_factor_pct`
- `sponsored_promotions_pct`
- `diversity_index`
- `retrieval_batch_size`
- `candidate_search_limit`
- `recency_decay_coefficient`
- `category_capping_threshold`
- `promotions_injection_percentile_threshold`

## Pipeline Stages

1. **Dataset Profile Builder**: derives sparsity, feature richness, sequence availability, signal strength, and reliability.
2. **AutoML Model Selector**: dynamically scores all stage candidate models using profile + orchestration features.
3. **RecBole Config Generator**: emits runtime config dicts and optional YAML files.
4. **Training Orchestrator**: executes RecBole runs.
5. **AutoML Tournament Engine**: chooses best model by Recall@K / NDCG@K / Hit@K objective.
6. **Candidate Pipeline**: Retrieval → Sequential (optional) → Ranking → Reranking.
7. **Business Rule Injection**: applies yield, sponsorship, diversity, category caps, recency effects.
8. **Final Fusion Engine**: computes final blended score and output ranking.

## Engineering Notes

- Model selection is driven by dataset profile signals + runtime orchestration params (no fixed single-model dependency).
- CPU/GPU controlled via `device` in `prepare(...)` and inference.
- Supports batch inference via `recommend_batch(...)`.
- Logs model selections for explainability.


## Run with `main.py`

### 1) Prepare profile + model selection + configs
```bash
python main.py prepare --mapping examples/sample_mapping.json --device cpu --export-yaml-dir generated_configs
```

### 2) Run inference pipeline
```bash
python main.py recommend --mapping examples/sample_mapping.json --candidates examples/sample_candidates.json --user-id U100 --device cpu
```

### 3) Run AutoML tournament training (requires RecBole-ready dataset)
```bash
python main.py train --mapping examples/sample_mapping.json --dataset ml-100k --device cpu
```
