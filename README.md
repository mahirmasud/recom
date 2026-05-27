# AutoML RecBole Recommender

This repository implements a **full, always-on inference pipeline**:

**Retrieval → Ranking → Reranking → AutoML orchestration**

The default recommendation path always generates candidates dynamically from the selected retriever model (LightGCN/BPR/ItemKNN). Static candidates are debug-only and optional.

## Quick Start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## CLI Usage

### 1) Prepare profile + model selection

```bash
python main.py prepare --mapping examples/sample_mapping.json --device cpu
```

### 2) Recommend (default, dynamic retrieval)

```bash
python main.py recommend --mapping examples/sample_mapping.json --user-id U100 --device cpu
```

### 3) Recommend with static seed candidates (debug mode only)

```bash
python main.py recommend \
  --mapping examples/sample_mapping.json \
  --user-id U100 \
  --candidates examples/sample_candidates.json \
  --candidates_mode static \
  --device cpu
```

> In debug static mode, retrieval still runs first; static candidates are merged as seed inputs.

### 4) Train AutoML stage tournaments

```bash
python main.py train --mapping examples/sample_mapping.json --dataset ml-100k --device cpu
```

## Inference Contract

For every request, the system returns:
- Final ranked recommendation list
- Retrieval model used
- Ranking model used
- Reranker used
- Candidate generation method
- Business rules summary

## Pipeline Guarantees

1. **Retrieval always runs first** using AutoML-selected retriever.
2. **Ranking always runs** on retrieved candidates.
3. **Reranking always runs** with business rules and final fusion.
4. **AutoML model selection always runs** and logs selected stage models.

## Important Logging

The pipeline emits:
- `Retrieval stage started: model=...`
- `Generated N candidates for user ...`
- `Ranking stage started: model=...`
- `Reranking stage applied: ...`
- `AutoML selected pipeline: {...}`
