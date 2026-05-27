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

### 4) Convert raw dataset to RecBole `.inter`

```bash
python main.py preprocess --input path/to/your_dataset.csv
```

This creates:
- `dataset/<dataset_name>/<dataset_name>.inter`

And prints the training command:

```bash
python main.py train --mapping examples/sample_mapping.json --dataset <dataset_name> --device cpu
```

### 5) Train AutoML stage tournaments

```bash
python main.py train --mapping examples/sample_mapping.json --dataset ml-100k --device cpu
```

## Using a config file (`--params`)

You can pass runtime orchestration settings from a JSON config file in `prepare`, `recommend`, and `train` commands via `--params`.

### Example config file

Create `config/params.json`:

```json
{
  "recommendation_yield_limit": 30,
  "personalization_focus_pct": 75.0,
  "discovery_factor_pct": 15.0,
  "sponsored_promotions_pct": 10.0,
  "diversity_index": 0.5,
  "retrieval_batch_size": 512,
  "candidate_search_limit": 1200,
  "recency_decay_coefficient": 0.65,
  "category_capping_threshold": 2,
  "promotions_injection_percentile_threshold": 0.8
}
```

### Example commands with config

```bash
python main.py prepare \
  --mapping examples/sample_mapping.json \
  --params config/params.json \
  --device cpu
```

```bash
python main.py recommend \
  --mapping examples/sample_mapping.json \
  --user-id U100 \
  --params config/params.json \
  --device cpu
```

```bash
python main.py train \
  --mapping examples/sample_mapping.json \
  --dataset ml-100k \
  --params config/params.json \
  --device cpu
```

> Note: `preprocess` only converts raw data to RecBole `.inter` format and does not consume `--params`.

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
