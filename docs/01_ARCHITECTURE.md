# GleasonXAI Architecture Deep Dive

This document provides a comprehensive technical overview of the GleasonXAI architecture, design decisions, and implementation details.

---

## Table of Contents

1. [Model Architecture Overview](#model-architecture-overview)
2. [U-Net Segmentation Model](#u-net-segmentation-model)
3. [EfficientNet-B4 Encoder](#efficientnet-b4-encoder)
4. [Multi-Level Prediction System](#multi-level-prediction-system)
5. [Sliding Window Inference](#sliding-window-inference)
6. [Ensemble Strategy](#ensemble-strategy)
7. [Design Decisions and Rationale](#design-decisions-and-rationale)

---

## Model Architecture Overview

GleasonXAI is built on a **U-Net architecture** with an **EfficientNet-B4 encoder**, implemented using the `segmentation-models-pytorch` (SMP) library.

### Why Semantic Segmentation?

Traditional Gleason grading AI systems output:
- **Class label:** "This is Gleason 4"
- **No spatial information:** Where is the Gleason 4 pattern?

GleasonXAI outputs:
- **Pixel-wise predictions:** Each pixel gets a class label
- **Spatial reasoning:** Pathologists can see exactly which regions support the diagnosis
- **Interpretability:** Matches how pathologists actually work (visually examining tissue)

### High-Level Architecture

```
┌────────────────────────────────────────────────────────────┐
│                    Input Image                             │
│                    [3, H, W]                               │
└────────────────┬───────────────────────────────────────────┘
                 │
                 ↓
┌────────────────────────────────────────────────────────────┐
│              EfficientNet-B4 Encoder                       │
│  ┌──────────┬──────────┬──────────┬──────────┬─────────┐ │
│  │ Block 0  │ Block 1  │ Block 2  │ Block 3  │ Block 4 │ │
│  │ [48,     │ [24,     │ [32,     │ [56,     │ [160,   │ │
│  │  H/2,    │  H/4,    │  H/8,    │  H/16,   │  H/32,  │ │
│  │  W/2]    │  W/4]    │  W/8]    │  W/16]   │  W/32]  │ │
│  └────┬─────┴────┬─────┴────┬─────┴────┬─────┴────┬────┘ │
└───────┼──────────┼──────────┼──────────┼──────────┼───────┘
        │          │          │          │          │
        │          │          │          │          ↓
        │          │          │          │     ┌─────────┐
        │          │          │          │     │Bottleneck│
        │          │          │          │     │ [272,   │
        │          │          │          │     │  H/32,  │
        │          │          │          │     │  W/32]  │
        │          │          │          │     └────┬────┘
        │          │          │          │          │
        │          │          │          │          ↓
        │          │          │          │     ┌──────────┐
        │          │          │          └─────┤Decoder 4 │
        │          │          │                └────┬─────┘
        │          │          │                     │
        │          │          │                     ↓
        │          │          │                ┌──────────┐
        │          │          └────────────────┤Decoder 3 │
        │          │                           └────┬─────┘
        │          │                                │
        │          │                                ↓
        │          │                           ┌──────────┐
        │          └────────────────────────────┤Decoder 2 │
        │                                       └────┬─────┘
        │                                            │
        │                                            ↓
        │                                       ┌──────────┐
        └────────────────────────────────────────┤Decoder 1 │
                                                 └────┬─────┘
                                                      │
                                                      ↓
┌────────────────────────────────────────────────────────────┐
│                  Segmentation Head                         │
│                  Conv2d(16, num_classes, 3x3)              │
└────────────────┬───────────────────────────────────────────┘
                 │
                 ↓
┌────────────────────────────────────────────────────────────┐
│                  Output Predictions                        │
│                  [num_classes, H, W]                       │
│                  (10 classes for level 1)                  │
└────────────────────────────────────────────────────────────┘
```

---

## U-Net Segmentation Model

### What is U-Net?

U-Net is a **fully convolutional encoder-decoder architecture** designed for biomedical image segmentation:

**Key characteristics:**
- **Encoder (Contracting Path):** Progressively downsamples input, extracting hierarchical features
- **Decoder (Expanding Path):** Progressively upsamples, reconstructing spatial resolution
- **Skip Connections:** Concatenate encoder features to decoder, preserving spatial information

### Why U-Net for GleasonXAI?

1. **Spatial precision:** Accurately localizes patterns at pixel level
2. **Multi-scale features:** Captures both local details (individual glands) and global context (tissue architecture)
3. **Proven track record:** State-of-the-art for medical image segmentation
4. **Skip connections:** Prevent loss of fine-grained spatial information during downsampling

### Implementation Details

**Library:** `segmentation-models-pytorch` (SMP)

```python
import segmentation_models_pytorch as smp

model = smp.Unet(
    encoder_name="efficientnet-b4",      # Encoder backbone
    encoder_weights="imagenet",          # Pretrained on ImageNet
    in_channels=3,                       # RGB input
    classes=10,                          # 10 classes for label_level=1
    activation=None,                     # No activation (logits output)
)
```

**Configuration:** [configs/model/segmentation_efficientnet.yaml](../configs/model/segmentation_efficientnet.yaml)

```yaml
_target_: segmentation_models_pytorch.Unet
encoder_name: efficientnet-b4
encoder_weights: imagenet
in_channels: 3
classes: ${dataset.num_classes}
activation: null
```

---

## EfficientNet-B4 Encoder

### What is EfficientNet?

EfficientNet is a family of convolutional neural networks optimized for **efficiency** (accuracy vs. computational cost).

**Key innovation:** **Compound scaling** - simultaneously scales:
- **Depth:** Number of layers
- **Width:** Number of channels per layer
- **Resolution:** Input image size

**EfficientNet-B4 specifications:**
- **Parameters:** ~19M
- **Input resolution:** 380×380 (baseline)
- **Depth:** 32 layers
- **ImageNet Top-1 accuracy:** ~82.6%

### Why EfficientNet-B4?

1. **Strong pretrained features:** ImageNet pretraining provides robust low-level feature extraction (edges, textures, patterns)
2. **Efficient:** Good accuracy/parameter trade-off
3. **Not too large:** B4 balances performance and memory usage (vs. B5-B7)
4. **Proven for histopathology:** Widely used in medical imaging

### EfficientNet-B4 Block Structure

```
Input [3, H, W]
    ↓
Stem Conv [48, H/2, W/2]
    ↓
Block 0: MBConv1 (k3x3) × 2  → [24, H/2, W/2]
    ↓
Block 1: MBConv6 (k3x3) × 4  → [32, H/4, W/4]
    ↓
Block 2: MBConv6 (k5x5) × 4  → [56, H/8, W/8]
    ↓
Block 3: MBConv6 (k3x3) × 6  → [112, H/16, W/16]
    ↓
Block 4: MBConv6 (k5x5) × 6  → [160, H/16, W/16]
    ↓
Block 5: MBConv6 (k5x5) × 8  → [272, H/32, W/32]
    ↓
Block 6: MBConv6 (k3x3) × 2  → [448, H/32, W/32]
    ↓
Head Conv [1792, H/32, W/32]
```

**MBConv:** Mobile Inverted Bottleneck Convolution
- **Inverted residual:** Expands then compresses channels
- **Depthwise separable convolutions:** Reduces parameters
- **Squeeze-and-excitation:** Attention mechanism for channel recalibration

---

## Multi-Level Prediction System

GleasonXAI supports **hierarchical classification** with 3 label levels.

### Label Hierarchy

```
Level 0: Gleason Patterns (3 classes)
│
├── 0: Gleason 3
├── 1: Gleason 4
└── 2: Gleason 5

Level 1: Fine-Grained Explanations (10 classes) [DEFAULT]
│
├── 0: Background
├── 1: Benign
├── 2: Individual glands (Gleason 3)
├── 3: Compressed glands (Gleason 3)
├── 4: Poorly formed glands (Gleason 4)
├── 5: Cribriform glands (Gleason 4)
├── 6: Glomeruloid glands (Gleason 4)
├── 7: Solid groups of tumor cells (Gleason 4)
├── 8: Single cells (Gleason 4)
├── 9: Cords (Gleason 4)
└── 10: Comedonecrosis (Gleason 5)

Level 2: Sub-Explanations (even more detailed)
```

### Configuration

Set `dataset.label_level` in training config:

```bash
# Train on Gleason patterns directly
python scripts/run_training.py dataset.label_level=0 ...

# Train on fine-grained explanations (recommended)
python scripts/run_training.py dataset.label_level=1 ...

# Train on sub-explanations
python scripts/run_training.py dataset.label_level=2 ...
```

### Why Multiple Levels?

1. **Flexibility:** Different use cases need different granularity
2. **Hierarchical learning:** Can train on explanations then map to Gleason grades
3. **Research:** Compare direct vs. explanation-based learning

### Label Remapping

**Implementation:** [src/gleasonxai/tree_loss.py](../src/gleasonxai/tree_loss.py)

The `TreeLoss` and `LabelRemapper` classes handle mapping between levels:

```python
from gleasonxai.tree_loss import TreeLoss

# Multi-level loss: jointly optimize all levels
tree_loss = TreeLoss(
    label_hierarchy=label_hierarchy,  # Nested dict of class relationships
    loss_weights=[0.5, 1.0, 0.5],    # Weights per level
)

# Forward pass
loss = tree_loss(predictions, targets, level=1)
```

---

## Sliding Window Inference

### The Problem

Whole-slide images (WSIs) can be **gigapixel-sized** (e.g., 100,000 × 100,000 pixels):
- **Cannot fit in GPU memory**
- **Cannot resize** without losing critical details

### The Solution: Sliding Window Inference (SWI)

Process large images in **overlapping patches**, then aggregate predictions.

```
┌────────────────────────────────────────┐
│         Large Image (e.g., 4096×4096)  │
│                                        │
│  ┌──────┐                              │
│  │Patch1│                              │
│  └──────┘                              │
│      ┌──────┐                          │
│      │Patch2│  (stride = 256)          │
│      └──────┘                          │
│          ┌──────┐                      │
│          │Patch3│                      │
│          └──────┘                      │
│                     ...                │
│                            ┌──────┐    │
│                            │PatchN│    │
│                            └──────┘    │
└────────────────────────────────────────┘
```

### Implementation

**Library:** `monai.inferers.SlidingWindowInferer`

```python
from monai.inferers import SlidingWindowInferer

inferer = SlidingWindowInferer(
    roi_size=(512, 512),           # Patch size
    sw_batch_size=4,               # Process 4 patches in parallel
    overlap=0.5,                   # 50% overlap between patches
    mode="gaussian",               # Gaussian weighting for overlaps
)

# Apply to large image
predictions = inferer(inputs, model)
```

**Configuration:** [configs/augmentations/microns_calibrated_sw.yaml](../configs/augmentations/microns_calibrated_sw.yaml)

### Gaussian Weighting

Overlapping regions get predictions from multiple patches. How to combine?

**Gaussian weighting:** Center of patch weighted higher than edges
- **Rationale:** Edges may have boundary artifacts
- **Effect:** Smooth blending between patches

```
Patch weight map (Gaussian):

    Low ←──────→ High ←──────→ Low

    0.1 0.3 0.5 0.7 0.9 1.0 0.9 0.7 0.5 0.3 0.1
    ↑                   ↑                   ↑
   Edge              Center              Edge
```

---

## Ensemble Strategy

GleasonXAI uses an **ensemble of 3 models** for final predictions.

### Why Ensemble?

1. **Reduces variance:** Different random initializations capture different aspects
2. **Improves robustness:** Less sensitive to outliers
3. **Better calibration:** Averaged probabilities more reliable
4. **Minimal cost:** Reuses same architecture and hyperparameters

### Implementation

**Models:**
- Model 1: Seed 1, trained independently
- Model 2: Seed 2, trained independently
- Model 3: Seed 3, trained independently

**Aggregation:** Average softmax probabilities

```python
# Ensemble prediction
def ensemble_predict(image, models):
    predictions = []
    for model in models:
        logits = model(image)
        probs = torch.softmax(logits, dim=1)
        predictions.append(probs)

    # Average probabilities
    ensemble_probs = torch.stack(predictions).mean(dim=0)

    # Final prediction
    pred_class = ensemble_probs.argmax(dim=1)

    return pred_class, ensemble_probs
```

**Script:** [scripts/run_gleasonXAI.py](../scripts/run_gleasonXAI.py)

### Model Checkpoints

Default locations (after running `setup.py`):

```
$DATASET_LOCATION/GleasonXAI/GleasonFinal2/label_level1/
├── SoftDiceBalanced-1/version_0/checkpoints/best_model.ckpt
├── SoftDiceBalanced-2/version_0/checkpoints/best_model.ckpt
└── SoftDiceBalanced-3/version_0/checkpoints/best_model.ckpt
```

---

## Design Decisions and Rationale

### 1. Why U-Net over other architectures?

**Alternatives considered:**
- **DeepLabV3+:** Excellent for natural images, but U-Net skip connections better preserve fine details
- **FCN:** Simpler, but U-Net outperforms on medical imaging
- **SegFormer:** Transformer-based, but requires more data and compute

**Decision:** U-Net provides best balance of accuracy, interpretability, and training efficiency for medical imaging.

**Evidence:** Configs include alternatives ([configs/model/](../configs/model/)):
- `segmentation_deeplabv3p_efficientnet.yaml`
- `segmentation_fcn.yaml`
- `segmentation_unetplusplus_efficientnet.yaml`

### 2. Why EfficientNet-B4 encoder?

**Alternatives:**
- **ResNet50/101:** Widely used, but EfficientNet more parameter-efficient
- **DenseNet:** Good feature reuse, but EfficientNet faster
- **EfficientNet-B5/B6:** Better accuracy, but diminishing returns vs. compute cost

**Decision:** B4 optimal trade-off for histopathology (pretrained features + efficiency)

### 3. Why semantic segmentation over classification?

**Alternative:** Image-level classification (entire image → Gleason grade)

**Problems:**
- **No spatial localization:** Can't show *where* the pattern is
- **Mixed patterns:** Images often contain multiple Gleason patterns
- **Not explainable:** Black box prediction

**Decision:** Segmentation provides pixel-wise explanations, matching pathologist workflow

### 4. Why soft labels over hard labels?

**Alternative:** Use single "ground truth" annotation per image

**Problems:**
- **High inter-observer variability:** Pathologists disagree ~30% of the time
- **Loss of information:** Discards diagnostic uncertainty
- **Overfitting:** Model forced to memorize arbitrary single annotation

**Decision:** Soft labels (average of multiple annotators) capture uncertainty and improve generalization

**Evidence from paper:** Soft label training outperformed single annotator (Dice: 0.713 vs. 0.691)

### 5. Why 3-model ensemble?

**Alternatives:**
- **Single model:** Simpler, but higher variance
- **5+ model ensemble:** Better performance, but diminishing returns

**Decision:** 3 models balances performance gain vs. inference cost

**Practical note:** Inference time ~3× slower, but still acceptable for clinical use

---

## Model Variants Available

All variants implemented in [configs/model/](../configs/model/):

| Model                  | Encoder         | Params | Config File                          |
|------------------------|-----------------|--------|--------------------------------------|
| **U-Net (default)**    | EfficientNet-B4 | ~22M   | `segmentation_efficientnet.yaml`     |
| U-Net++                | EfficientNet-B4 | ~25M   | `segmentation_unetplusplus_efficientnet.yaml` |
| DeepLabV3+             | EfficientNet-B4 | ~20M   | `segmentation_deeplabv3p_efficientnet.yaml` |
| U-Net                  | ResNet50        | ~30M   | `segmentation_resnet.yaml`           |
| FCN                    | ResNet50        | ~28M   | `segmentation_fcn.yaml`              |
| U-Net                  | DenseNet121     | ~25M   | `segmentation_densenet.yaml`         |

**Usage:**

```bash
# Train with alternative model
python scripts/run_training.py \
    model=segmentation_deeplabv3p_efficientnet \
    dataset.label_level=1 \
    loss_functions=soft_dice_balanced
```

---

## Performance Characteristics

### Inference Speed (Single Model)

| Image Size | GPU         | Inference Time | Throughput |
|------------|-------------|----------------|------------|
| 512×512    | A100 40GB   | ~50ms          | 20 img/s   |
| 1024×1024  | A100 40GB   | ~200ms         | 5 img/s    |
| 2048×2048  | A100 40GB   | ~800ms (SWI)   | 1.25 img/s |

**Ensemble:** Multiply by 3 for 3-model ensemble

### Memory Requirements

| Batch Size | Image Size | GPU Memory |
|------------|------------|------------|
| 4          | 512×512    | ~6GB       |
| 8          | 512×512    | ~10GB      |
| 16         | 512×512    | ~18GB      |
| 4          | 1024×1024  | ~20GB      |

**Note:** EfficientNet-B4 encoder has moderate memory footprint

---

## Next Steps

- [02_DATA_PIPELINE.md](02_DATA_PIPELINE.md) - Understanding data loading and preprocessing
- [03_TRAINING.md](03_TRAINING.md) - Training procedures and loss functions
- [05_INFERENCE.md](05_INFERENCE.md) - Running inference with ensemble

---

## References

1. **U-Net:** Ronneberger et al., "U-Net: Convolutional Networks for Biomedical Image Segmentation" (2015)
2. **EfficientNet:** Tan & Le, "EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks" (2019)
3. **SMP Library:** https://github.com/qubvel/segmentation_models.pytorch
4. **MONAI:** https://monai.io/
