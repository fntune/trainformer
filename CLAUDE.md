# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

Monorepo with:
1. **trainformer/** (v0.1.0) - Python-first training library for deep learning
2. **product-matching/** - Production metric learning pipeline (Hydra + ArcFace)
3. **SSL/** - Self-supervised learning pipeline (PyTorch Lightning, git submodule)

## Commands

```bash
# Install
pip install -e .                                 # editable install
pip install -e ".[dev]"                          # with test/lint deps

# Test
pytest tests/                                    # all tests
pytest tests/test_trainer.py -v                  # single file
pytest tests/test_trainer.py::test_fit -v        # single test
pytest tests/ -m "not slow"                      # skip slow tests

# Lint
ruff check trainformer/                          # lint
ruff format trainformer/                         # format

# CLI
trainformer train --task=ImageClassification --model=resnet50 --data=data/train
trainformer train --task=MetricLearning --model=efficientnet_b0 --data=data/sop
trainformer train --task=SSL.simclr --model=resnet50 --data=data/imagenet
trainformer predict --checkpoint=model.pth --data=data/test --task=ImageClassification

# Product Matching
cd product-matching && pip install -r requirements.txt
python run.py                                    # train with default config
python run.py dataloader=local model.backbone.model_name=efficientnet_b3
python run.py -m data.batch_size=16,32,64        # multirun sweep
```

## Trainformer Architecture

```
trainformer/
├── trainer.py      → Universal loop (AMP, compile, callbacks, logging)
├── types.py        → Task protocol, DatasetInfo, ConfigSource
├── tasks/
│   ├── vision/     → Classification, MetricLearning, SSL (SimCLR/MoCo/DINO/MAE)
│   ├── nlp/        → CausalLM, Seq2Seq, MaskedLM
│   └── multimodal/ → CLIP, VLM
├── models/components/
│   ├── backbones.py → TimmBackbone (timm wrapper)
│   ├── losses.py    → ArcFace, CosFace, SubcenterArcFace, NTXent, InfoNCE
│   ├── poolers.py   → GeM pooling
│   └── heads.py     → Classification, projection heads
├── callbacks/      → EarlyStopping, ModelCheckpoint, EMA, KNN
├── adapters/       → LoRA
├── data/           → Image, text loaders, samplers
└── eval/           → FAISS feature index, retrieval metrics
```

### Core Patterns

| Pattern | Description |
|---------|-------------|
| **Task Protocol** | `train_step(batch) → loss`, `eval_step(batch) → metrics`, `configure(DatasetInfo)` |
| **DatasetInfo** | Single metadata object (num_classes, input_shape) flows through pipeline |
| **ConfigSource** | Tracks value origin: USER, DATA, DERIVED, ENV |
| **TaskBase** | Lifecycle hooks: `on_train_begin()`, `on_epoch_end()`, `on_step_end()` |

### Product Matching (`product-matching/`)

```
run.py              → Entry point, fold splitting
src/train.py        → Custom training loop with AMP, FAISS eval
src/models/dolg.py  → Model definition
src/models/components/
  ├── backbones.py    → TimmBackbone wrapper
  ├── loss_heads.py   → ArcFace, SubcenterArcFace
  └── pooler_heads.py → GeM pooling
conf/               → Hydra configs
```

Hydra patterns: `key=value` (override), `+key=value` (add), `-m key=v1,v2` (multirun)

## Environment Variables
```
WANDB_API_KEY         # or run `wandb login`
AWS_ACCESS_KEY_ID
AWS_SECRET_ACCESS_KEY
```
