# GleasonXAI Project Rules for Claude Code

## Critical Context

**Project:** Pathologist-like explainable AI for prostate cancer grading (published in Nature Communications 2025)
**Domain:** Medical imaging, deep learning, semantic segmentation, explainable AI
**Primary Paradigm:** SOFT LABEL TRAINING with multi-annotator consensus (NOT hard labels)

## Documentation-First Approach

**ALWAYS read documentation BEFORE modifying code:**

1. **Start here:** [docs/AI_ASSISTANT_GUIDE.md](docs/AI_ASSISTANT_GUIDE.md) - Comprehensive guide specifically for AI assistants
2. **Architecture questions:** [docs/01_ARCHITECTURE.md](docs/01_ARCHITECTURE.md)
3. **Data pipeline questions:** [docs/02_DATA_PIPELINE.md](docs/02_DATA_PIPELINE.md)
4. **Training questions:** [docs/03_TRAINING.md](docs/03_TRAINING.md)
5. **API reference:** [docs/07_CODE_REFERENCE.md](docs/07_CODE_REFERENCE.md)

## Core Principles (NON-NEGOTIABLE)

### 1. Soft Labels Are Sacred
- **NEVER** use hard label losses (CrossEntropyLoss) with this dataset
- **ALWAYS** use `SoftDiceLoss` or `TreeLoss` for training
- Soft labels = averaging annotations from multiple pathologists
- Implementation: [src/gleasonxai/gleason_data.py:60-78](src/gleasonxai/gleason_data.py#L60-L78)

### 2. Macro Averaging for Imbalanced Data
- **NEVER** use micro averaging (pixel-weighted) for loss functions
- **ALWAYS** use macro averaging (class-balanced) to handle imbalance
- Background/benign tissue dominates ~70% of pixels
- Config: [configs/loss_functions/soft_dice_balanced.yaml](configs/loss_functions/soft_dice_balanced.yaml)

### 3. Hierarchical Label System
- Level 0: Gleason patterns (3, 4, 5) - coarse
- **Level 1: Fine-grained explanations (10 classes) - DEFAULT and RECOMMENDED**
- Level 2: Sub-explanations - research only
- Training on explanations (level 1) OUTPERFORMS direct pattern prediction (Dice: 0.713 vs. 0.691)

### 4. Ensemble Is Standard
- Production inference uses **3 independently trained models**
- Aggregate via softmax probability averaging
- Script: [scripts/run_gleasonXAI.py](scripts/run_gleasonXAI.py)

### 5. Microns Calibration Matters
- Different scanners → different microns/pixel ratios
- **ALWAYS** use `MicronsCalibrated` images for training/inference
- Ensures scale consistency across datasets

## Common Tasks - Decision Tree

### "How does X work?" or "Where is Y implemented?"
1. Check [docs/AI_ASSISTANT_GUIDE.md](docs/AI_ASSISTANT_GUIDE.md#common-tasks-and-how-to-help)
2. Check [docs/07_CODE_REFERENCE.md](docs/07_CODE_REFERENCE.md)
3. Then read the specific source file

### "Add new loss function"
1. Read [docs/03_TRAINING.md](docs/03_TRAINING.md) for context
2. Add to [src/gleasonxai/model_utils.py](src/gleasonxai/model_utils.py)
3. Create config in [configs/loss_functions/](configs/loss_functions/)
4. **Verify:** Must support soft labels (NOT just hard labels)
5. Add unit test in [tests/](tests/)

### "Modify training procedure"
1. **Do NOT modify code directly** - use Hydra config overrides
2. Training params: [configs/optimization/](configs/optimization/)
3. Model architecture: [configs/model/](configs/model/)
4. Override syntax: `python scripts/run_training.py dataset.label_level=1 optimization.optimizer.lr=5e-4`

### "Debug training issues"
1. Check [docs/AI_ASSISTANT_GUIDE.md#common-pitfalls-and-how-to-avoid-them](docs/AI_ASSISTANT_GUIDE.md#common-pitfalls-and-how-to-avoid-them)
2. Verify environment variables: `DATASET_LOCATION`, `EXPERIMENT_LOCATION`
3. Check loss function compatibility (soft vs. hard labels)
4. Review batch size vs. GPU memory

### "Run inference on new data"
1. Read [scripts/run_gleasonXAI.py](scripts/run_gleasonXAI.py)
2. Ensure 3 model checkpoints available
3. Use sliding window inference for large images

## File Modification Guidelines

### High-Impact Files (MODIFY WITH EXTREME CARE)
These files are core to the published paper's methodology:
- [src/gleasonxai/gleason_data.py](src/gleasonxai/gleason_data.py) - Dataset loading, soft label generation
- [src/gleasonxai/lightning_modul.py](src/gleasonxai/lightning_modul.py) - Training loop, metrics
- [src/gleasonxai/model_utils.py](src/gleasonxai/model_utils.py) - SoftDiceLoss implementation

**Before modifying:**
1. Read the entire file
2. Read corresponding documentation
3. Check if change can be done via config instead
4. Add comprehensive tests
5. Verify backward compatibility with paper results

### Safe to Modify
- [configs/](configs/) - Configuration files (preferred way to change behavior)
- [scripts/evaluation_utils.py](scripts/evaluation_utils.py) - Visualization
- [scripts/publication_utils.py](scripts/publication_utils.py) - Paper-specific utilities

### Testing Checklist
```bash
# 1. Unit tests
pytest tests/test_*.py

# 2. Integration tests
pytest tests/integration_tests/

# 3. Quick training smoke test
python scripts/run_training.py \
    dataset.label_level=1 \
    trainer.max_epochs=1 \
    dataloader.batch_size=2

# 4. Verify metrics logged correctly
```

## Environment Setup

### Required Environment Variables
```bash
export DATASET_LOCATION=/path/to/datasets
export EXPERIMENT_LOCATION=/path/to/experiments
```

**Alternative:** Use `.env` file with `uv run --env-file=.env`

### Python Version
- **Exact requirement:** Python 3.10.13
- **Reason:** Dependency compatibility, reproducibility

### Dependency Management
- **Tool:** `uv` (modern alternative to pip/conda)
- **Install:** `uv sync`
- **Run scripts:** `uv run scripts/run_training.py ...`

## Code Style and Conventions

### Docstrings
- **Style:** Google format
- **Required for:** All public functions, classes, methods
- **Include:** Args, Returns, Raises, Examples

### Type Hints
- **Required for:** All function signatures
- **Use:** `from typing import Union, Optional, Literal, Dict, List, Tuple`

### Configuration
- **All hyperparameters:** Managed via Hydra YAML configs
- **Never hardcode:** Magic numbers, paths, model settings
- **Use:** `${oc.env:VARIABLE}` for environment variables in configs

## Common Pitfalls (READ THIS BEFORE ANY CODE CHANGE)

1. **Using hard label loss with soft labels** → Poor performance
   - Fix: Use `SoftDiceLoss`, NOT `CrossEntropyLoss`

2. **Forgetting environment variables** → `KeyError: 'DATASET_LOCATION'`
   - Fix: Export `DATASET_LOCATION` and `EXPERIMENT_LOCATION`

3. **Label level mismatch** → Model output classes ≠ data classes
   - Fix: Ensure `dataset.num_classes` matches `model.classes`

4. **Missing ImageNet normalization** → Model won't converge
   - Fix: All augmentation pipelines must include normalization

5. **Using micro averaging** → Background class dominates metrics
   - Fix: Use `average='macro'` for class-balanced metrics

## Research Context

**Publication:** Nature Communications, Volume 16, Article 8959 (2025)
**DOI:** [10.1038/s41467-025-64712-4](https://doi.org/10.1038/s41467-025-64712-4)
**arXiv:** [2410.15012](https://arxiv.org/abs/2410.15012)

**Key Innovation:** First inherently explainable AI for Gleason grading using pathologist terminology

**Performance:** Dice 0.713±0.003 (explanations) vs. 0.691±0.010 (direct patterns)

**Dataset:** 1,015 tissue microarray cores, 54 pathologists, multi-annotator soft labels

## Quick Commands Reference

```bash
# Training (default: level 1, soft dice loss)
python scripts/run_training.py

# Training with config overrides
python scripts/run_training.py \
    dataset.label_level=1 \
    loss_functions=soft_dice_balanced \
    optimization.optimizer.lr=5e-4 \
    trainer.max_epochs=400

# Evaluation
python scripts/test.py \
    --experiment_path $DATASET_LOCATION/GleasonXAI \
    --checkpoint GleasonFinal2/label_level1 \
    --glob_checkpoints "SoftDiceBalanced-*"

# Inference (ensemble)
python scripts/run_gleasonXAI.py \
    --images /path/to/images \
    --save_path /path/to/output

# Run tests
pytest tests/

# Setup dataset (first time only)
python scripts/setup.py --gleasonxai_data GleasonXAI_data.zip --download --calibrate
```

## Special Instructions for AI Assistants

1. **Always read docs first:** Don't guess how things work - the documentation is comprehensive
2. **Explain the "why":** Include scientific context, not just "what" the code does
3. **Reference files with line numbers:** Use format `file.py:123` or `file.py:123-145`
4. **Respect the soft label paradigm:** This is central to GleasonXAI's success
5. **Test recommendations mentally:** Against the architecture and data pipeline before suggesting
6. **Suggest best practices:** Based on the paper's validated approach
7. **Provide complete examples:** Working code that can be copy-pasted

## Repository Structure Quick Reference

```
GleasonXAI/
├── src/gleasonxai/              # Core Python package (2564 lines)
│   ├── gleason_data.py          # Dataset (606 lines) - CRITICAL
│   ├── lightning_modul.py       # Training (594 lines) - CRITICAL
│   ├── model_utils.py           # Utilities (182 lines) - CRITICAL
│   ├── augmentations.py         # Augmentation (306 lines)
│   ├── tree_loss.py             # Hierarchical loss (180 lines)
│   └── ...
├── scripts/                     # Executable scripts
│   ├── run_training.py          # Training entry point
│   ├── test.py                  # Evaluation
│   └── run_gleasonXAI.py        # Inference (ensemble)
├── configs/                     # Hydra YAML configs (33 files)
│   ├── config.yaml              # Master config
│   ├── dataset/                 # Dataset configs
│   ├── model/                   # Model architecture configs
│   └── loss_functions/          # Loss function configs
├── docs/                        # Documentation (6 markdown files)
│   ├── AI_ASSISTANT_GUIDE.md    # START HERE
│   ├── 00_OVERVIEW.md
│   ├── 01_ARCHITECTURE.md
│   ├── 02_DATA_PIPELINE.md
│   ├── 03_TRAINING.md
│   └── 07_CODE_REFERENCE.md
├── tests/                       # Unit and integration tests
└── pyproject.toml               # Dependencies (managed by uv)
```

## When in Doubt

1. Read [docs/AI_ASSISTANT_GUIDE.md](docs/AI_ASSISTANT_GUIDE.md)
2. Check [docs/AI_ASSISTANT_GUIDE.md#common-pitfalls-and-how-to-avoid-them](docs/AI_ASSISTANT_GUIDE.md#common-pitfalls-and-how-to-avoid-them)
3. Search for similar implementations in the codebase
4. Ask user for clarification rather than guessing

---

**Last Updated:** 2026-01-29
**Documentation Version:** 1.2.0
**Codebase Version:** 1.2.0
