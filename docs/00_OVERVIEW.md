# GleasonXAI: Comprehensive Repository Overview

## Project Information

**Name:** GleasonXAI
**Version:** 1.2.0 (as of January 2025)
**Purpose:** Pathologist-like explainable AI for interpretable Gleason grading in prostate cancer
**Institution:** German Cancer Research Center (DKFZ), Heidelberg
**Authors:** Hendrik Mehrtens, Gesa Mittmann

### Publication

- **Journal:** Nature Communications, Volume 16, Article 8959 (2025)
- **DOI:** [10.1038/s41467-025-64712-4](https://doi.org/10.1038/s41467-025-64712-4)
- **Preprint:** [arXiv:2410.15012](https://doi.org/10.48550/arXiv.2410.15012)
- **Dataset:** Available on [Figshare](https://springernature.figshare.com/articles/dataset/Pathologist-like_explainable_AI_for_interpretable_Gleason_grading_in_prostate_cancer/27301845)

---

## Scientific Context

### What is Gleason Grading?

Gleason grading is the **gold standard** for assessing prostate cancer aggressiveness. Pathologists examine biopsy tissue under a microscope and assign grades (3, 4, or 5) based on the architectural patterns of cancer cells:

- **Gleason 3:** Well-formed, individual glands
- **Gleason 4:** Poorly formed, fused, or cribriform glands
- **Gleason 5:** Solid sheets, single cells, or necrosis

The **Gleason Score** combines the two most prevalent patterns (e.g., 3+4=7), which determines treatment decisions and patient outcomes.

### The Challenge

Gleason grading is:
- **Subjective:** High inter-observer variability among pathologists
- **Complex:** Requires years of training to master
- **Critical:** Directly impacts patient treatment decisions

Traditional AI systems provide Gleason predictions without explaining *why*, making pathologists hesitant to trust them.

### GleasonXAI's Innovation

GleasonXAI is an **inherently explainable AI** system that:

1. **Speaks pathologist language:** Uses medical terminology pathologists already use
2. **Shows spatial reasoning:** Provides pixel-wise segmentation maps showing which tissue regions correspond to which patterns
3. **Captures uncertainty:** Trained on soft labels from multiple pathologists to model diagnostic variability
4. **Achieves superior performance:** Dice score 0.713±0.003 vs. 0.691±0.010 for direct Gleason pattern prediction

Unlike post-hoc explainability methods (e.g., GradCAM), GleasonXAI's architecture is **designed for interpretability from the ground up**.

---

## Key Features

### 1. Hierarchical Label System

GleasonXAI supports a **3-level hierarchical classification system**:

```
Level 0: Gleason Patterns (3 classes)
├── Gleason 3
├── Gleason 4
└── Gleason 5

Level 1: Fine-Grained Explanations (10 classes)
├── Benign
├── Gleason 3 explanations:
│   ├── Individual glands
│   └── Compressed glands
├── Gleason 4 explanations:
│   ├── Poorly formed glands
│   ├── Cribriform glands
│   ├── Glomeruloid glands
│   ├── Solid groups of tumor cells
│   ├── Single cells
│   └── Cords
└── Gleason 5 explanations:
    └── Comedonecrosis

Level 2: Sub-Explanations (even more detailed)
```

**Default training:** Level 1 (fine-grained explanations)
**Why this matters:** Provides clinically meaningful explanations while maintaining diagnostic accuracy

### 2. Multi-Annotator Soft Labels

- **Dataset:** 1,015 tissue microarray (TMA) images
- **Annotators:** 54 international pathologists (median 15 years experience)
- **Soft label generation:** Averages annotations across multiple pathologists per image
- **Benefits:** Captures diagnostic uncertainty and reduces noise from individual biases

### 3. Ensemble Prediction

GleasonXAI uses **3 independently trained models** with identical architecture but different random seeds:
- Reduces variance
- Improves robustness
- Provides confidence estimates

### 4. Sliding Window Inference (SWI)

For large whole-slide images (WSIs):
- Processes images in overlapping patches
- Aggregates predictions with Gaussian weighting
- Handles arbitrarily large images

### 5. Advanced Loss Functions

- **Primary:** `SoftDiceLoss` with macro averaging (class-balanced)
- **Multi-level:** `TreeLoss` for hierarchical label learning
- **Experimental:** Jaccard-Dice-Tversky (JDT) loss family

---

## Technology Stack

### Core Framework
- **Python:** 3.10.13 (exact version requirement)
- **Deep Learning:** PyTorch 2.1.1
- **Training Framework:** PyTorch Lightning 2.2.0
- **Dependency Management:** `uv` (modern, fast alternative to pip/conda)

### Key Libraries
- **Segmentation Models:** `segmentation-models-pytorch` (SMP)
- **Medical Imaging:** `MONAI` (Medical Open Network for AI)
- **Augmentation:** `albumentations`
- **Metrics:** `torchmetrics`
- **Configuration:** `hydra-core` (compositional config management)
- **Hyperparameter Tuning:** `optuna`
- **Experiment Tracking:** Weights & Biases (wandb), TensorBoard

### Model Architecture
- **Base Model:** U-Net (encoder-decoder architecture)
- **Encoder:** EfficientNet-B4 (ImageNet pretrained)
- **Output:** Pixel-wise semantic segmentation (10 classes for level 1)

---

## Repository Structure

```
GleasonXAI/
│
├── src/gleasonxai/              # Core package (Python modules)
│   ├── __init__.py
│   ├── gleason_data.py         # Dataset loading and processing (606 lines)
│   ├── gleason_utils.py        # Visualization and utilities (226 lines)
│   ├── lightning_modul.py      # PyTorch Lightning modules (594 lines)
│   ├── loss_functions.py       # Loss function wrappers (33 lines)
│   ├── model_utils.py          # Model utilities and metrics (182 lines)
│   ├── tree_loss.py            # Hierarchical loss functions (180 lines)
│   ├── jdt_losses.py           # Jaccard-Dice-Tversky losses (437 lines)
│   └── augmentations.py        # Data augmentation pipelines (306 lines)
│
├── scripts/                     # Executable scripts
│   ├── run_training.py         # Main training entry point
│   ├── train.py                # Training implementation
│   ├── test.py                 # Testing and evaluation
│   ├── run_gleasonXAI.py       # Inference script (ensemble)
│   ├── setup.py                # Dataset setup and download
│   ├── evaluation_utils.py     # Evaluation and visualization
│   ├── publication_utils.py    # Paper-specific utilities
│   ├── calculate_dataset_characteristics.py
│   ├── calculate_fleiss_kappa.py
│   ├── create_downscaled_dataset.py
│   └── create_metric_visualization.py
│
├── configs/                     # Hydra configuration files (33 YAML files)
│   ├── config.yaml             # Master configuration
│   ├── dataset/                # Dataset configurations
│   ├── model/                  # Model architecture configs
│   ├── loss_functions/         # Loss function configs
│   ├── optimization/           # Optimizer and training configs
│   ├── augmentations/          # Data augmentation configs
│   ├── trainer/                # PyTorch Lightning trainer configs
│   ├── dataloader/             # DataLoader configs
│   ├── logger/                 # Logging configs
│   └── hparam_search/          # Hyperparameter search configs
│
├── tests/                       # Unit and integration tests
│   ├── test_gleason_data.py
│   ├── test_gleason_utils.py
│   ├── test_model_utils.py
│   └── integration_tests/
│
├── notebooks/                   # Jupyter notebooks
│   ├── evaluate_paper_results.ipynb
│   └── feature_visualization.ipynb
│
├── jobs/                        # Job submission scripts (HPC)
├── outputs/                     # Training outputs
├── wandb/                       # Weights & Biases logs
│
├── pyproject.toml              # Project metadata and dependencies
├── uv.lock                     # Locked dependency versions
├── README.md                   # User documentation
├── CHANGELOG.md                # Version history
└── LICENSE
```

---

## Workflow Overview

### 1. Setup Phase
```bash
# Install dependencies
uv sync

# Set environment variables
export DATASET_LOCATION=/path/to/data
export EXPERIMENT_LOCATION=/path/to/logs

# Download and prepare datasets
python scripts/setup.py --gleasonxai_data GleasonXAI_data.zip --download --calibrate
```

### 2. Training Phase
```bash
# Train on fine-grained explanations (level 1) with SoftDiceLoss
python scripts/run_training.py \
    dataset.label_level=1 \
    loss_functions=soft_dice_balanced \
    experiment=MyExperiment
```

### 3. Testing Phase
```bash
# Evaluate on test set
python scripts/test.py \
    --experiment_path $DATASET_LOCATION/GleasonXAI \
    --checkpoint GleasonFinal2/label_level1 \
    --glob_checkpoints "SoftDiceBalanced-*"
```

### 4. Inference Phase
```bash
# Run predictions on new images
python scripts/run_gleasonXAI.py \
    --images /path/to/images \
    --save_path /path/to/output
```

---

## Data Flow Diagram

```
┌─────────────────────────────────────────────────────────────┐
│              Raw Data Sources                               │
│  ┌──────────────┬──────────────┬──────────────────┐        │
│  │ Tissue       │ Gleason2019  │ Harvard Arvaniti │        │
│  │ Microarray   │ Challenge    │ Dataset          │        │
│  └──────┬───────┴──────┬───────┴────────┬─────────┘        │
└─────────┼──────────────┼────────────────┼──────────────────┘
          │              │                │
          └──────────────┴────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────────┐
│                GleasonX Dataset Class                       │
│  • Load images and annotations                              │
│  • Parse multi-annotator labels                             │
│  • Apply label hierarchy mapping                            │
│  • Generate soft labels (average across annotators)         │
│  • Create segmentation masks                                │
│  • Apply tissue filtering (background removal)              │
│  • Split into train/val/test (70/15/15)                     │
└────────────────────────┬────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────────┐
│              Albumentations Pipeline                        │
│  • Scale/crop operations                                    │
│  • Tellez transforms (elastic, blur, noise)                 │
│  • Rotation, flipping                                       │
│  • Normalization (ImageNet stats)                           │
└────────────────────────┬────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────────┐
│              PyTorch DataLoader                             │
│  • Batch creation                                           │
│  • Shuffling (train only)                                   │
│  • Multi-worker loading                                     │
└────────────────────────┬────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────────┐
│         U-Net Model (EfficientNet-B4 Encoder)               │
│  • Forward pass                                             │
│  • Output: [Batch, Classes, Height, Width]                  │
└────────────────────────┬────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────────┐
│              Loss Computation                               │
│  • SoftDiceLoss (primary)                                   │
│  • TreeLoss (multi-level hierarchical)                      │
│  • Class balancing (macro averaging)                        │
└────────────────────────┬────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────────┐
│              Backpropagation & Optimization                 │
│  • Adam optimizer                                           │
│  • Learning rate: 1e-3                                      │
│  • Weight decay: 0.02                                       │
│  • Early stopping (patience=3)                              │
└────────────────────────┬────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────────┐
│              Metrics & Logging                              │
│  • Accuracy (micro/macro)                                   │
│  • Dice score (soft/hard)                                   │
│  • F1, AUROC, calibration error                             │
│  • W&B / TensorBoard logging                                │
└─────────────────────────────────────────────────────────────┘
```

---

## Version History

### v1.2.0 (2025-05-06) - Current
- Restructured to modern Python packaging (src-layout)
- Migrated from Conda to `uv` for dependency management
- Added comprehensive unit and integration tests
- Moved scripts to dedicated `scripts/` directory
- Started semantic versioning

### v1.1.0 (2025-01-20)
- Improved documentation
- Documented required folder structure

### v1.0.0 (2024-09-12)
- Original release

---

## System Requirements

- **Python:** 3.10.13 (exact version)
- **GPU:** NVIDIA GPU with CUDA support
- **OS:** Linux (tested), macOS/Windows (untested)
- **RAM:** 16GB+ recommended
- **Disk:** ~50GB for datasets and model weights

---

## Quick Start

```bash
# 1. Clone repository
cd /path/to/ProQuant-AI/GleasonXAI

# 2. Install dependencies
uv sync

# 3. Set environment variables
export DATASET_LOCATION=/path/to/data
export EXPERIMENT_LOCATION=/path/to/logs

# 4. Download data from Figshare
# Visit: https://springernature.figshare.com/articles/dataset/Pathologist-like_explainable_AI_for_interpretable_Gleason_grading_in_prostate_cancer/27301845

# 5. Setup datasets
uv run scripts/setup.py --gleasonxai_data GleasonXAI_data.zip

# 6. Run inference on sample image
uv run scripts/run_gleasonXAI.py \
    --images /path/to/image.jpg \
    --save_path /path/to/output
```

---

## Further Documentation

- [01_ARCHITECTURE.md](01_ARCHITECTURE.md) - Detailed model architecture and design decisions
- [02_DATA_PIPELINE.md](02_DATA_PIPELINE.md) - Data loading, preprocessing, and augmentation
- [03_TRAINING.md](03_TRAINING.md) - Training procedures, loss functions, and optimization
- [04_CONFIGURATION.md](04_CONFIGURATION.md) - Hydra configuration system guide
- [05_INFERENCE.md](05_INFERENCE.md) - Running predictions and ensemble inference
- [06_EVALUATION.md](06_EVALUATION.md) - Metrics, visualization, and paper results
- [07_CODE_REFERENCE.md](07_CODE_REFERENCE.md) - Detailed API reference for all modules
- [08_DEVELOPMENT.md](08_DEVELOPMENT.md) - Contributing, testing, and development workflow

---

## Citation

If you use GleasonXAI in your research, please cite:

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

---

## Contact & Support

- **Issues:** [GitHub Issues](https://github.com/path/to/repo/issues)
- **Email:** hendrikalexander.mehrtens@dkfz-heidelberg.de, gesa.mittmann@dkfz-heidelberg.de
- **Institution:** German Cancer Research Center (DKFZ), Heidelberg, Germany
