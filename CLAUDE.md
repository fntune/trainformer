# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

This is a multi-project monorepo for deep learning training pipelines focused on metric learning and product matching. It contains two main projects:

- **SSL/** - Self-supervised learning pipeline using PyTorch Lightning
- **product-matching/** - Product matching with ArcFace/metric learning (custom training loop)

Both projects use Hydra for configuration management and support W&B logging.

## Commands

### SSL Project (PyTorch Lightning)
```bash
cd SSL
pip install -r requirements.txt
python run.py                                    # train with default config
python run.py --config-name experiment.yaml      # use different config
python run.py max_epochs=20 data.batch_size=64   # override params
python run.py -m data.batch_size=16,32,64        # multirun sweep
```

### Product Matching Project
```bash
cd product-matching
pip install -r requirements.txt
python run.py                                              # train with default config
python run.py dataloader=local model.backbone.model_name=efficientnet_b3
```

### Common Hydra Patterns
- Override params: `key=value` (no `--` prefix)
- Add new params: `+key=value`
- Switch config groups: `dataloader=colab` or `model=arcface`
- Multirun sweep: `-m key=val1,val2,val3`

## Architecture

### SSL Project Structure
```
SSL/
├── run.py                    # Entry point - instantiates DataModule, LightningModule, Trainer
├── conf/                     # Hydra configs (data, module, trainer, callbacks, etc.)
└── src/
    ├── modules/              # LightningModule implementations (base.py, dino.py, cgd.py, dolg.py)
    ├── datamodule/           # LightningDataModule with TensorDict support
    ├── components/           # Reusable nn.Modules (backbones, loss_heads, pooler_heads)
    ├── callbacks/            # Lightning callbacks (checkpoint, metrics, knn_online)
    └── core/                 # LR schedulers, loggers, utilities
```

### Product Matching Project Structure
```
product-matching/
├── run.py                    # Entry point - data prep, fold splitting, calls train()
├── conf/                     # Hydra configs
└── src/
    ├── train.py              # Custom training loop with AMP, FAISS evaluation
    ├── models/               # Model definitions (dolg.py) + components/
    ├── datamodules/          # ImageDataset
    └── testing.py            # Metric computation (sweep_matching, compute_scores)
```

### Key Design Patterns

1. **Hydra Instantiation**: Models, transforms, losses are instantiated via `hydra.utils.instantiate()` from config `_target_` paths

2. **Backbone Abstraction**: Both projects use `TimmBackbone` wrapper around timm models with `remove_fc=true`

3. **Metric Learning Losses**: ArcFace and SubcenterArcFace in `components/loss_heads.py`, configured via `loss/` configs

4. **Evaluation**: FAISS-based nearest neighbor search with threshold sweeping for F1 optimization

## Environment Variables
```
WANDB_API_KEY    # or run `wandb login` once
AWS_ACCESS_KEY_ID
AWS_SECRET_ACCESS_KEY
```
