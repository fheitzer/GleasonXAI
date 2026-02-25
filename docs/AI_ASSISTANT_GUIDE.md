# AI Assistant Guide for GleasonXAI

**Purpose:** This document provides comprehensive context for AI assistants (like Claude) to effectively work with the GleasonXAI codebase.

**Last Updated:** 2026-01-29

---

## Quick Start for AI Assistants

### What is GleasonXAI?

GleasonXAI is a **pathologist-like explainable AI system** for prostate cancer grading published in **Nature Communications (2025)**. It provides:

1. **Interpretable predictions** using medical terminology pathologists understand
2. **Pixel-wise segmentation** showing exactly which tissue regions correspond to which Gleason patterns
3. **Soft label training** from 54 pathologists to capture diagnostic uncertainty
4. **State-of-the-art performance** (Dice 0.713 vs. 0.691 for direct Gleason grading)

### Core Technology Stack

- **Python 3.10.13** (exact version requirement)
- **PyTorch 2.1.1** + **PyTorch Lightning 2.2.0**
- **Model:** U-Net with EfficientNet-B4 encoder (via segmentation-models-pytorch)
- **Dependency Management:** `uv` (modern alternative to pip/conda)
- **Configuration:** Hydra (compositional YAML configs)
- **Experiment Tracking:** Weights & Biases, TensorBoard

### Repository Structure (Key Locations)

```
GleasonXAI/
├── src/gleasonxai/              # Python package (2564 lines)
│   ├── gleason_data.py         # Dataset class (606 lines)
│   ├── lightning_modul.py      # Training module (594 lines)
│   ├── augmentations.py        # Data augmentation (306 lines)
│   ├── tree_loss.py            # Hierarchical loss (180 lines)
│   ├── model_utils.py          # Utilities (182 lines)
│   ├── gleason_utils.py        # Visualization (226 lines)
│   ├── jdt_losses.py           # Experimental losses (437 lines)
│   └── loss_functions.py       # Loss wrappers (33 lines)
│
├── scripts/                     # Executable scripts
│   ├── run_training.py         # Training entry point
│   ├── test.py                 # Evaluation
│   ├── run_gleasonXAI.py       # Inference (ensemble)
│   └── setup.py                # Dataset setup
│
├── configs/                     # Hydra YAML configs (33 files)
│   ├── config.yaml             # Master config
│   ├── dataset/                # Dataset configs
│   ├── model/                  # Model architecture configs
│   ├── loss_functions/         # Loss configs
│   └── ...
│
├── docs/                        # Comprehensive documentation
│   ├── 00_OVERVIEW.md          # Project overview
│   ├── 01_ARCHITECTURE.md      # Model architecture
│   ├── 02_DATA_PIPELINE.md     # Data loading/processing
│   ├── 03_TRAINING.md          # Training guide
│   ├── 07_CODE_REFERENCE.md    # API reference
│   └── AI_ASSISTANT_GUIDE.md   # This file
│
├── tests/                       # Unit and integration tests
├── pyproject.toml              # Dependencies and metadata
└── README.md                   # User documentation
```

---

## Key Concepts to Understand

### 1. Hierarchical Label System

GleasonXAI uses **3-level hierarchical classification**:

```
Level 0: Gleason Patterns (4 classes)
├── Background
├── Gleason 3 (well-formed glands)
├── Gleason 4 (poorly formed/fused glands)
└── Gleason 5 (solid sheets/necrosis)

Level 1: Fine-Grained Explanations (10 classes) [DEFAULT]
├── Background
├── Benign
├── Individual glands (Gleason 3)
├── Compressed glands (Gleason 3)
├── Poorly formed glands (Gleason 4)
├── Cribriform glands (Gleason 4)
├── Glomeruloid glands (Gleason 4)
├── Solid groups (Gleason 4)
├── Single cells (Gleason 4)
├── Cords (Gleason 4)
└── Comedonecrosis (Gleason 5)

Level 2: Sub-Explanations (more detailed)
```

**Default training:** Level 1 (fine-grained explanations) - provides interpretability while maintaining accuracy.

**Key insight:** Training on explanations (level 1) outperforms direct Gleason pattern prediction (level 0).

### 2. Soft Label Training

**Problem:** Pathologists disagree ~30% of the time on Gleason grading.

**Solution:** Generate **soft labels** by averaging annotations from multiple pathologists.

**Example:**
```python
# 5 pathologists annotate pixel (100, 100):
# - 3 vote "individual glands" (class 2)
# - 1 votes "compressed glands" (class 3)
# - 1 votes "poorly formed glands" (class 4)

# Soft label probabilities:
[0.0, 0.0, 0.6, 0.2, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0]
```

**Benefit:** Captures diagnostic uncertainty and improves generalization.

**Implementation:** [src/gleasonxai/gleason_data.py:60-78](../src/gleasonxai/gleason_data.py#L60-L78) in `prepare_torch_inputs()`

### 3. Ensemble Prediction

**Strategy:** Use 3 independently trained models with identical architecture but different random seeds.

**Aggregation:** Average softmax probabilities across models.

**Benefits:**
- Reduces variance
- Improves calibration
- Better handles uncertainty

**Implementation:** [scripts/run_gleasonXAI.py](../scripts/run_gleasonXAI.py)

### 4. Hydra Configuration System

All parameters managed via **compositional YAML configs**.

**Structure:**
```
configs/
├── config.yaml                 # Master config with defaults
├── dataset/                    # Dataset variants
│   └── segmentation_microns_calibrated.yaml
├── model/                      # Model architectures
│   ├── segmentation_efficientnet.yaml  [DEFAULT]
│   ├── segmentation_deeplabv3p_efficientnet.yaml
│   └── ...
├── loss_functions/             # Loss functions
│   ├── soft_dice_balanced.yaml  [RECOMMENDED]
│   ├── soft_dice_balanced_multilevel.yaml
│   └── ...
└── ...
```

**Override syntax:**
```bash
# Dot notation for nested params
python scripts/run_training.py dataset.label_level=1

# Config group selection
python scripts/run_training.py model=segmentation_resnet

# Combined
python scripts/run_training.py \
    dataset.label_level=0 \
    model=segmentation_deeplabv3p_efficientnet \
    optimization.optimizer.lr=5e-4
```

---

## Common Tasks and How to Help

### Task 1: Understanding the Codebase

**When asked:** "How does X work?" or "Where is Y implemented?"

**Approach:**
1. Check [07_CODE_REFERENCE.md](07_CODE_REFERENCE.md) for detailed API docs
2. Check [01_ARCHITECTURE.md](01_ARCHITECTURE.md) for model details
3. Check [02_DATA_PIPELINE.md](02_DATA_PIPELINE.md) for data processing
4. Read source code with context from docs

**Key files to reference:**
- **Dataset:** [src/gleasonxai/gleason_data.py](../src/gleasonxai/gleason_data.py)
- **Training:** [src/gleasonxai/lightning_modul.py](../src/gleasonxai/lightning_modul.py)
- **Loss functions:** [src/gleasonxai/model_utils.py](../src/gleasonxai/model_utils.py), [src/gleasonxai/tree_loss.py](../src/gleasonxai/tree_loss.py)

### Task 2: Debugging Training Issues

**When asked:** "Training loss not decreasing" or "Out of memory error"

**Checklist:**
1. **Verify data loading:**
   ```python
   dataset = GleasonX(path="/path/to/data", label_level=1)
   img, label, metadata = dataset[0]
   print(img.shape, label.shape)
   print(label.min(), label.max())  # Should be [0, 1] for soft labels
   ```

2. **Check loss function compatibility:**
   - **Soft labels require soft losses:** Use `SoftDiceLoss`, NOT `CrossEntropyLoss`
   - **Verify loss config:** [configs/loss_functions/soft_dice_balanced.yaml](../configs/loss_functions/soft_dice_balanced.yaml)

3. **GPU memory issues:**
   - Reduce batch size: `dataloader.batch_size=2`
   - Reduce image size: `dataset.image_size=256`
   - Check [03_TRAINING.md#troubleshooting](03_TRAINING.md#troubleshooting)

4. **Learning rate too high/low:**
   - Default: `lr=1e-3`
   - Try: `lr=1e-4` or `lr=5e-4`

### Task 3: Adding New Features

**When asked:** "Can you add support for X?"

**Considerations:**
1. **Maintain consistency with existing patterns:**
   - Use Hydra configs for all hyperparameters
   - Follow PyTorch Lightning module structure
   - Add unit tests in `tests/`

2. **Key extension points:**
   - **New loss function:** Add to [src/gleasonxai/model_utils.py](../src/gleasonxai/model_utils.py) + config in [configs/loss_functions/](../configs/loss_functions/)
   - **New model architecture:** Add config in [configs/model/](../configs/model/), use segmentation-models-pytorch
   - **New augmentation:** Add to [src/gleasonxai/augmentations.py](../src/gleasonxai/augmentations.py) + config
   - **New metric:** Add to `initialize_torchmetrics()` in [lightning_modul.py](../src/gleasonxai/lightning_modul.py)

3. **Testing:**
   - Add unit test in `tests/test_*.py`
   - Run: `pytest tests/`

### Task 4: Modifying Training Workflow

**When asked:** "How do I change the training procedure?"

**Key parameters:**

| Parameter | Location | Default | Description |
|-----------|----------|---------|-------------|
| Learning rate | [configs/optimization/optuna_tuned.yaml](../configs/optimization/optuna_tuned.yaml) | 1e-3 | Adam learning rate |
| Batch size | [configs/dataloader/](../configs/dataloader/) | 4-8 | Images per batch |
| Max epochs | [configs/trainer/small_epochs.yaml](../configs/trainer/small_epochs.yaml) | 400 | Max training epochs |
| Early stopping patience | [configs/optimization/](../configs/optimization/) | 3 | Epochs without improvement |
| Image size | [configs/dataset/](../configs/dataset/) | 512 | Input image dimensions |
| Label level | [configs/dataset/](../configs/dataset/) | 1 | Hierarchy level |

**Common modifications:**

```bash
# Change learning rate
python scripts/run_training.py optimization.optimizer.lr=5e-4

# Longer training
python scripts/run_training.py trainer.max_epochs=600

# Different label level
python scripts/run_training.py dataset.label_level=0  # Direct Gleason patterns
```

### Task 5: Understanding Paper Results

**When asked:** "How were the paper results generated?"

**Paper configuration:**
- **Dataset:** Tissue microarray (TMA) with 1,015 images
- **Model:** U-Net with EfficientNet-B4 encoder
- **Training:** Level 1 (fine-grained explanations), SoftDiceLoss (macro averaging)
- **Ensemble:** 3 models with different random seeds
- **Metrics:** Dice score, accuracy, F1, AUROC, calibration error

**Reproduce paper results:**
```bash
# 1. Setup data
python scripts/setup.py --gleasonxai_data GleasonXAI_data.zip --download --calibrate

# 2. Train 3 models
for seed in 1 2 3; do
    python scripts/run_training.py \
        dataset.label_level=1 \
        loss_functions=soft_dice_balanced \
        seed=$seed \
        experiment=GleasonFinal2/label_level1/SoftDiceBalanced-${seed}
done

# 3. Evaluate
python scripts/test.py \
    --experiment_path $DATASET_LOCATION/GleasonXAI \
    --checkpoint GleasonFinal2/label_level1 \
    --glob_checkpoints "SoftDiceBalanced-*"

# 4. Generate figures
jupyter notebook notebooks/evaluate_paper_results.ipynb
```

**Key result:** Dice 0.713±0.003 (explanations) vs. 0.691±0.010 (direct patterns)

**Reference:** [Nature Communications paper](https://doi.org/10.1038/s41467-025-64712-4)

### Task 6: Inference on New Data

**When asked:** "How do I run predictions on my images?"

**Quick start:**
```bash
# 1. Setup (download model weights)
python scripts/setup.py --gleasonxai_data GleasonXAI_data.zip

# 2. Run inference
python scripts/run_gleasonXAI.py \
    --images /path/to/my_images \
    --save_path /path/to/output
```

**What happens:**
1. Loads 3 model checkpoints (ensemble)
2. For each image:
   - Applies sliding window inference (if large)
   - Generates predictions from each model
   - Averages predictions (ensemble)
   - Creates color-coded segmentation overlay
3. Saves results to `save_path/`

**Advanced options:**
```bash
python scripts/run_gleasonXAI.py \
    --images /path/to/images \
    --save_path /path/to/output \
    --checkpoint_1 /custom/path/model1.ckpt \
    --checkpoint_2 /custom/path/model2.ckpt \
    --checkpoint_3 /custom/path/model3.ckpt \
    --checkpoint_absolute  # Use absolute paths
```

---

## Important Implementation Details

### 1. Soft Label Handling

**Critical:** Many standard losses don't support soft labels out-of-the-box.

**Correct losses for soft labels:**
- ✅ `SoftDiceLoss` (custom implementation)
- ✅ `TreeLoss` with `SoftDiceLoss` base
- ✅ `OneHotDICE` (MONAI wrapper)
- ❌ `CrossEntropyLoss` (requires hard labels)
- ❌ `OneHotCE` (converts soft→hard via argmax, loses info)

**Implementation:** [src/gleasonxai/model_utils.py:SoftDiceLoss](../src/gleasonxai/model_utils.py)

### 2. Class Imbalance

**Problem:** Background and benign tissue dominate pixel counts (~70%).

**Solution:** Use **macro averaging** (class-balanced) instead of micro (pixel-weighted).

**Comparison:**
```python
# Micro averaging (BAD for imbalanced data)
SoftDiceLoss(average='micro')  # Dominated by majority classes

# Macro averaging (GOOD for imbalanced data)
SoftDiceLoss(average='macro')  # Treats all classes equally
```

**Default:** [configs/loss_functions/soft_dice_balanced.yaml](../configs/loss_functions/soft_dice_balanced.yaml) uses `average: macro`

### 3. Image Scaling

**Issue:** Different scanners produce different microns/pixel ratios.

**Solution:** Rescale all images to **consistent microns/pixel** before training.

**Implementation:**
- **Original images:** Variable resolution (stored in `TMA/original/`)
- **Calibrated images:** Consistent microns/pixel (stored in `TMA/MicronsCalibrated/`)

**Create calibrated images:**
```bash
python scripts/setup.py --calibrate
```

**Why it matters:** Model learns scale-invariant features, generalizes across scanners.

### 4. Sliding Window Inference

**Use case:** Images larger than GPU memory can handle.

**Implementation:** MONAI's `SlidingWindowInferer`

```python
from monai.inferers import SlidingWindowInferer

inferer = SlidingWindowInferer(
    roi_size=(512, 512),      # Patch size
    sw_batch_size=4,          # Parallel patches
    overlap=0.5,              # 50% overlap
    mode='gaussian',          # Gaussian blending
)

predictions = inferer(large_image, model)
```

**Configuration:** [configs/augmentations/microns_calibrated_sw.yaml](../configs/augmentations/microns_calibrated_sw.yaml)

**Enabled in:** Validation/test/inference (not training)

### 5. Environment Variables

**Required:**
```bash
export DATASET_LOCATION=/path/to/datasets
export EXPERIMENT_LOCATION=/path/to/experiments
```

**Alternative (with uv):**
Create `.env` file:
```env
DATASET_LOCATION=/path/to/datasets
EXPERIMENT_LOCATION=/path/to/experiments
```

Then run:
```bash
uv run --env-file=.env scripts/run_training.py ...
```

---

## Common Pitfalls and How to Avoid Them

### Pitfall 1: Using Hard Labels with Soft Label Data

**Symptom:** Poor performance despite correct architecture.

**Cause:** Loss function expects hard labels but data provides soft labels.

**Fix:** Use `SoftDiceLoss` or `TreeLoss`, NOT `CrossEntropyLoss`.

### Pitfall 2: Not Setting Environment Variables

**Symptom:** `KeyError: 'DATASET_LOCATION'`

**Cause:** Hydra configs reference `${oc.env:DATASET_LOCATION}` but variable not set.

**Fix:**
```bash
export DATASET_LOCATION=/path/to/data
export EXPERIMENT_LOCATION=/path/to/logs
```

### Pitfall 3: Incorrect Label Level Configuration

**Symptom:** Mismatch between model output classes and data classes.

**Cause:** Training with `label_level=1` (10 classes) but model configured for different number.

**Fix:** Ensure consistency:
```yaml
# In config
dataset:
  label_level: 1
  num_classes: 10  # Should match level 1 class count

model:
  classes: ${dataset.num_classes}  # Reference dataset config
```

### Pitfall 4: Forgetting to Normalize Augmentation

**Symptom:** Model doesn't converge or produces random predictions.

**Cause:** EfficientNet expects ImageNet-normalized inputs but augmentation pipeline missing normalization.

**Fix:** All augmentation pipelines must end with:
```python
albumentations.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
)
```

### Pitfall 5: Incorrect Metric Interpretation

**Symptom:** Confusion about whether higher or lower is better.

**Clarity:**
- **Accuracy, DICE, F1, AUROC:** Higher is better (range [0, 1])
- **Loss, calibration error:** Lower is better (range [0, ∞))

**Early stopping:** Monitors `val_loss_epoch` with `mode='min'`

---

## Code Modification Guidelines

### When Adding New Features

1. **Follow existing patterns:**
   - Use Hydra configs for all hyperparameters
   - Add docstrings (Google style)
   - Type hints for function signatures
   - Unit tests in `tests/`

2. **Maintain compatibility:**
   - Don't break existing configs
   - Preserve backward compatibility where possible
   - Update documentation

3. **Test thoroughly:**
   ```bash
   pytest tests/
   python scripts/run_training.py \
       [your_new_config] \
       trainer.max_epochs=1  # Quick smoke test
   ```

### When Modifying Core Logic

**High-impact files (modify carefully):**
- [src/gleasonxai/gleason_data.py](../src/gleasonxai/gleason_data.py) - Dataset loading
- [src/gleasonxai/lightning_modul.py](../src/gleasonxai/lightning_modul.py) - Training loop
- [src/gleasonxai/model_utils.py](../src/gleasonxai/model_utils.py) - Loss functions

**Safe to modify:**
- [configs/](../configs/) - Configuration files
- [scripts/evaluation_utils.py](../scripts/evaluation_utils.py) - Visualization
- [notebooks/](../notebooks/) - Analysis notebooks

**Testing checklist:**
1. Run unit tests: `pytest tests/`
2. Run integration test: `pytest tests/integration_tests/`
3. Run quick training: `max_epochs=1`
4. Verify metrics logged correctly

### Version Control Best Practices

1. **Commit configs with code:**
   - Config changes should be versioned
   - Use semantic versioning (see [CHANGELOG.md](../CHANGELOG.md))

2. **Document changes:**
   - Update [CHANGELOG.md](../CHANGELOG.md)
   - Update relevant docs in `docs/`

3. **Reproducibility:**
   - Pin dependencies in `pyproject.toml`
   - Use `uv.lock` for exact versions
   - Record random seeds in configs

---

## Research Context and Citations

### Scientific Background

**Prostate cancer:** Second most common cancer in men worldwide.

**Gleason grading:** Standard method for assessing prostate cancer aggressiveness.
- Developed by Dr. Donald Gleason in 1960s
- Based on architectural patterns of cancer cells
- **Gleason score** = sum of two most prevalent patterns (e.g., 3+4=7)
- **Problem:** High inter-observer variability (~30% disagreement)

**GleasonXAI contribution:**
- First **inherently explainable** AI for Gleason grading
- Uses **pathologist terminology** instead of abstract features
- **Soft label training** captures diagnostic uncertainty
- **Superior performance** while maintaining interpretability

### Key Publications

**Primary paper:**
```bibtex
@article{mehrtens2025gleasonxai,
  title={Pathologist-like explainable AI for interpretable Gleason grading in prostate cancer},
  author={Mehrtens, Hendrik A. and Mittmann, Gesa and others},
  journal={Nature Communications},
  volume={16},
  number={8959},
  year={2025},
  doi={10.1038/s41467-025-64712-4}
}
```

**Related work:**
- Tellez et al. (2018): Augmentation strategies for histopathology
- Gleason2019 Challenge: Benchmark dataset
- MONAI: Medical imaging AI framework

### Dataset Information

- **Primary dataset:** 1,015 tissue microarray (TMA) cores
- **Annotators:** 54 international pathologists (10 countries)
- **Experience:** Median 15 years clinical experience
- **Annotation type:** Detailed localized pattern descriptions
- **Availability:** [Figshare](https://springernature.figshare.com/articles/dataset/Pathologist-like_explainable_AI_for_interpretable_Gleason_grading_in_prostate_cancer/27301845)

---

## Quick Reference: File Locations

### Core Python Modules

| Module | Path | Lines | Purpose |
|--------|------|-------|---------|
| Dataset | `src/gleasonxai/gleason_data.py` | 606 | Data loading |
| Training | `src/gleasonxai/lightning_modul.py` | 594 | PyTorch Lightning module |
| Augmentation | `src/gleasonxai/augmentations.py` | 306 | Data transforms |
| Hierarchical Loss | `src/gleasonxai/tree_loss.py` | 180 | Multi-level loss |
| Loss Functions | `src/gleasonxai/model_utils.py` | 182 | SoftDiceLoss, metrics |
| Utilities | `src/gleasonxai/gleason_utils.py` | 226 | Visualization |

### Key Configuration Files

| Config | Path | Purpose |
|--------|------|---------|
| Master | `configs/config.yaml` | Default settings |
| Dataset | `configs/dataset/segmentation_microns_calibrated.yaml` | Data params |
| Model | `configs/model/segmentation_efficientnet.yaml` | Architecture |
| Loss | `configs/loss_functions/soft_dice_balanced.yaml` | Loss function |
| Optimizer | `configs/optimization/optuna_tuned.yaml` | Training params |
| Augmentation | `configs/augmentations/microns_calibrated_sw.yaml` | Transforms |

### Key Scripts

| Script | Path | Purpose |
|--------|------|---------|
| Training | `scripts/run_training.py` | Main training |
| Testing | `scripts/test.py` | Evaluation |
| Inference | `scripts/run_gleasonXAI.py` | Ensemble prediction |
| Setup | `scripts/setup.py` | Data preparation |

---

## Summary: Most Important Things to Remember

1. **GleasonXAI is inherently explainable** - not post-hoc explainability
2. **Default training:** Level 1 (explanations) with `SoftDiceLoss` (macro)
3. **Soft labels require soft losses** - use `SoftDiceLoss`, not `CrossEntropyLoss`
4. **Ensemble of 3 models** for final predictions
5. **Macro averaging handles class imbalance** - always use for imbalanced medical data
6. **Microns calibration** ensures scale consistency across scanners
7. **Hydra configs** manage all parameters - use command-line overrides
8. **Documentation is comprehensive** - reference `docs/` for detailed info

---

## Getting Help

### Documentation Hierarchy

1. **This file (AI_ASSISTANT_GUIDE.md)** - Overview and common tasks
2. **00_OVERVIEW.md** - Project overview and quick start
3. **01_ARCHITECTURE.md** - Model architecture details
4. **02_DATA_PIPELINE.md** - Data loading and preprocessing
5. **03_TRAINING.md** - Training procedures and loss functions
6. **07_CODE_REFERENCE.md** - Detailed API reference

### For Specific Questions

- **"How does X work?"** → Check [07_CODE_REFERENCE.md](07_CODE_REFERENCE.md)
- **"How do I train a model?"** → Check [03_TRAINING.md](03_TRAINING.md)
- **"What is the model architecture?"** → Check [01_ARCHITECTURE.md](01_ARCHITECTURE.md)
- **"How is data loaded?"** → Check [02_DATA_PIPELINE.md](02_DATA_PIPELINE.md)
- **"How do I reproduce the paper?"** → Check [00_OVERVIEW.md](00_OVERVIEW.md) and [03_TRAINING.md](03_TRAINING.md)

### External Resources

- **Nature paper:** https://www.nature.com/articles/s41467-025-64712-4
- **arXiv preprint:** https://arxiv.org/abs/2410.15012
- **Dataset:** https://springernature.figshare.com/articles/dataset/Pathologist-like_explainable_AI_for_interpretable_Gleason_grading_in_prostate_cancer/27301845
- **DKFZ news:** https://www.dkfz.de/en/news/press-releases/detail/transparent-artificial-intelligence-improves-assessment-of-prostate-cancer-aggressiveness

---

**Last updated:** 2026-01-29
**Documentation version:** 1.2.0
**Codebase version:** 1.2.0

---

## For Future AI Assistants

This documentation was created to provide comprehensive context for working with the GleasonXAI codebase. When helping users:

1. **Prefer reading the documentation first** before analyzing code directly
2. **Reference specific files and line numbers** when explaining implementations
3. **Provide complete, working examples** when showing code
4. **Explain the "why" not just the "what"** - include scientific context
5. **Suggest best practices** based on the paper's validated approach
6. **Test recommendations** mentally against the architecture and data pipeline
7. **Respect the soft label paradigm** - it's central to GleasonXAI's success

The codebase is well-structured and thoroughly documented. Use these resources to provide accurate, helpful assistance!
