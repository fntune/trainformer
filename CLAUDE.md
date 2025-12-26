# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

This is a monorepo containing:

1. **trainformer** (planned) - Python-first training library for deep learning. See `PLAN.md` for full design spec.
2. **product-matching/** - Production metric learning pipeline with ArcFace (Hydra + custom training loop)
3. **SSL/** - Self-supervised learning pipeline (PyTorch Lightning, git submodule)

## Commands

### Product Matching
```bash
cd product-matching
pip install -r requirements.txt
python run.py                                    # train with default config
python run.py dataloader=local model.backbone.model_name=efficientnet_b3
python run.py -m data.batch_size=16,32,64        # multirun sweep
```

### Hydra Patterns
- Override: `key=value` (no `--` prefix)
- Add new param: `+key=value`
- Switch config group: `dataloader=colab` or `model=arcface`
- Multirun: `-m key=val1,val2,val3`

## Architecture

### Product Matching (`product-matching/`)
```
run.py              → Entry point, fold splitting, calls train()
src/train.py        → Custom training loop with AMP, FAISS eval
src/models/dolg.py  → Model definition
src/models/components/
  ├── backbones.py    → TimmBackbone wrapper (timm models)
  ├── loss_heads.py   → ArcFace, SubcenterArcFace
  └── pooler_heads.py → GeM pooling
src/testing.py      → sweep_matching, compute_scores (F1 optimization)
conf/               → Hydra configs (model/, loss/, transforms/, etc.)
```

### Key Patterns

1. **Hydra Instantiation**: Models/transforms/losses instantiated via `hydra.utils.instantiate()` from config `_target_` paths

2. **TimmBackbone**: Wrapper around timm models with `remove_fc=true`, used across all vision tasks

3. **Metric Learning**: ArcFace loss with learnable weights in `loss_heads.py`, configured via `conf/loss/`

4. **Evaluation**: FAISS-based nearest neighbor search with threshold sweeping for F1 optimization

### Trainformer Design (see PLAN.md)

The planned trainformer library follows these patterns:
- **Task Protocol**: `train_step()`, `eval_step()`, `configure(DatasetInfo)`, `parameters()`
- **Trainer**: Universal loop with phased initialization, callbacks, multi-backend logging
- **PipelineContext**: Config source tracking (USER/DATA/DERIVED), lifecycle management (STEP/EPOCH/RUN)

## Environment Variables
```
WANDB_API_KEY         # or run `wandb login`
AWS_ACCESS_KEY_ID
AWS_SECRET_ACCESS_KEY
```
