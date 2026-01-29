# GleasonXAI Data Pipeline

This document provides a comprehensive guide to the data loading, preprocessing, and augmentation pipeline in GleasonXAI.

---

## Table of Contents

1. [Dataset Overview](#dataset-overview)
2. [GleasonX Dataset Class](#gleasonx-dataset-class)
3. [Data Sources](#data-sources)
4. [Annotation System](#annotation-system)
5. [Label Hierarchy](#label-hierarchy)
6. [Soft Label Generation](#soft-label-generation)
7. [Data Preprocessing](#data-preprocessing)
8. [Data Augmentation](#data-augmentation)
9. [Train/Val/Test Splitting](#trainvaltest-splitting)
10. [DataLoader Configuration](#dataloader-configuration)

---

## Dataset Overview

### Statistics

- **Total images:** 1,015 tissue microarray (TMA) core images
- **Annotators:** 54 international pathologists
- **Annotations per image:** Variable (typically 3-5)
- **Image resolution:** Variable (original and microns-calibrated versions)
- **Label levels:** 3 hierarchical levels
- **Classes (level 1):** 10 (background, benign, 8 Gleason explanations)

### Data Sources

GleasonXAI integrates **three datasets**:

1. **Tissue Microarray (TMA):** Primary dataset with multi-pathologist annotations
2. **Gleason2019 Challenge:** Public dataset from grand-challenge.org
3. **Harvard Arvaniti et al.:** Public dataset from dataverse

---

## GleasonX Dataset Class

**Location:** [src/gleasonxai/gleason_data.py](../src/gleasonxai/gleason_data.py:606)

The `GleasonX` class is a PyTorch `Dataset` subclass that handles all data loading and preprocessing.

### Class Signature

```python
class GleasonX(torch.utils.data.Dataset):
    def __init__(
        self,
        path: Union[str, Path],
        label_level: int = 1,
        scaling: Literal["MicronsCalibrated", "original"] = "MicronsCalibrated",
        sources: List[str] = ["gleason2019", "tissue_microarrays", "dataverse"],
        transforms: Optional[albumentations.Compose] = None,
        tissue_mask_kwargs: Optional[Dict] = None,
        label_remapping: Optional[Dict] = None,
        seed: int = 42,
    ):
        ...
```

### Key Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `path` | str/Path | Required | Root directory containing images and annotations |
| `label_level` | int | 1 | Hierarchy level (0=Gleason patterns, 1=explanations, 2=sub-explanations) |
| `scaling` | str | "MicronsCalibrated" | Image scaling mode |
| `sources` | List[str] | All 3 | Which datasets to include |
| `transforms` | Compose | None | Albumentations pipeline |
| `tissue_mask_kwargs` | Dict | None | Tissue filtering parameters |
| `label_remapping` | Dict | None | Custom label mappings |
| `seed` | int | 42 | Random seed for reproducibility |

### Usage Example

```python
from gleasonxai.gleason_data import GleasonX
from gleasonxai.augmentations import create_zoom_crop

# Create dataset
dataset = GleasonX(
    path="/path/to/data/GleasonXAI",
    label_level=1,
    scaling="MicronsCalibrated",
    transforms=create_zoom_crop(image_size=512),
    seed=42,
)

# Access sample
image, label, metadata = dataset[0]

print(image.shape)        # torch.Tensor [3, 512, 512]
print(label.shape)        # torch.Tensor [10, 512, 512] (soft labels)
print(metadata.keys())    # dict_keys(['TMA_identifier', 'num_annotators', ...])
```

---

## Data Sources

### 1. Tissue Microarray (TMA)

**Primary dataset** with detailed pathologist annotations.

**Location:** `$DATASET_LOCATION/GleasonXAI/TMA/{original,MicronsCalibrated}/`

**Characteristics:**
- **Images:** ~800-900 cores
- **Format:** JPEG
- **Resolution:** Variable (rescaled to consistent microns/pixel)
- **Annotations:** Multi-pathologist soft labels
- **Naming convention:** `PR482a_A1.jpg`, `PR482a_A2.jpg`, etc.

**File structure:**
```
TMA/
├── original/              # Original image resolution
│   ├── PR482a_A1.jpg
│   ├── PR482a_A2.jpg
│   └── ...
└── MicronsCalibrated/     # Rescaled to consistent microns/pixel
    ├── PR482a_A1.jpg
    ├── PR482a_A2.jpg
    └── ...
```

### 2. Gleason2019 Challenge Dataset

**Public benchmark** from MICCAI Gleason2019 Grand Challenge.

**Location:** Automatically downloaded by `scripts/setup.py` (optional)

**Characteristics:**
- **Images:** ~244 cores (train + test)
- **Format:** PNG
- **Annotations:** Single expert annotations
- **URL:** https://gleason2019.grand-challenge.org/

### 3. Harvard Arvaniti et al. Dataset

**Public research dataset** from Harvard Medical School.

**Location:** Automatically downloaded by `scripts/setup.py`

**Characteristics:**
- **Images:** Additional prostate cancer cores
- **Format:** JPEG
- **Annotations:** Gleason scores
- **Reference:** Arvaniti et al., Scientific Reports (2018)

### Handling Multiple Sources

**Problem:** Different datasets have different file naming conventions and may have duplicate filenames.

**Solution:** [gleason_data.py:115-150](../src/gleasonxai/gleason_data.py#115-150)

```python
def extract_tma_info(tma_filename):
    """Extract TMA identifier, removing dataset-specific suffixes."""
    identifier = Path(tma_filename).stem
    # Remove grade suffix patterns:
    # - Pattern 1: _1 through _5 at the end (tissue microarray)
    # - Pattern 2: _grade1 through _grade5 at the end (Gleason2019)
    identifier = re.sub(r'_grade[12345]$', '', identifier)
    identifier = re.sub(r'_[12345]$', '', identifier)
    return identifier
```

---

## Annotation System

### Annotation Format

Annotations stored in CSV: `$DATASET_LOCATION/GleasonXAI/final_filtered_explanations_df.csv`

**Columns:**
- `TMA`: Image filename
- `explanations`: Text description of pattern (e.g., "individual glands")
- `points`: Polygon coordinates (as string)
- `annotator_id`: Pathologist identifier
- Additional metadata

**Example rows:**
```csv
TMA,explanations,points,annotator_id
PR482a_A1.jpg,individual glands,"[[100,200], [150,200], [150,250], [100,250]]",pathologist_042
PR482a_A1.jpg,cribriform glands,"[[300,400], [350,400], [350,450], [300,450]]",pathologist_042
PR482a_A1.jpg,individual glands,"[[105,205], [145,195], [155,245], [95,255]]",pathologist_017
```

### Multi-Annotator Annotations

**Key insight:** Each image annotated by **multiple pathologists** (typically 3-5).

**Why?** Gleason grading has significant inter-observer variability (~30% disagreement).

**Approach:** Generate **soft labels** by averaging annotations.

---

## Label Hierarchy

GleasonXAI uses a **3-level hierarchical label system**.

### Level 0: Gleason Patterns (Coarse)

```python
{
    "background": 0,
    "Gleason 3": 1,
    "Gleason 4": 2,
    "Gleason 5": 3,
}
```

**Use case:** Direct Gleason grade prediction

### Level 1: Fine-Grained Explanations (Default)

```python
{
    "background": 0,
    "benign": 1,
    # Gleason 3 explanations
    "individual glands": 2,
    "compressed glands": 3,
    # Gleason 4 explanations
    "poorly formed glands": 4,
    "cribriform glands": 5,
    "glomeruloid glands": 6,
    "solid groups of tumor cells": 7,
    "single cells": 8,
    "cords": 9,
    # Gleason 5 explanations
    "comedonecrosis": 10,
}
```

**Use case:** Explainable Gleason grading (recommended)

### Level 2: Sub-Explanations (Fine)

Further subdivisions of level 1 classes for highly detailed analysis.

**Use case:** Research on fine-grained pattern recognition

### Hierarchy Structure

**Implementation:** [src/gleasonxai/tree_loss.py](../src/gleasonxai/tree_loss.py)

```python
label_hierarchy = {
    "background": {
        "benign": None,
    },
    "Gleason 3": {
        "individual glands": None,
        "compressed glands": None,
    },
    "Gleason 4": {
        "poorly formed glands": None,
        "cribriform glands": None,
        "glomeruloid glands": None,
        "solid groups of tumor cells": None,
        "single cells": None,
        "cords": None,
    },
    "Gleason 5": {
        "comedonecrosis": None,
    },
}
```

### Label Remapping

**File:** `$DATASET_LOCATION/GleasonXAI/label_remapping.json`

Maps between:
- German ↔ English terminology
- Free-text annotations → standardized classes
- Legacy class names → current names

**Example:**
```json
{
    "einzelne Drüsen": "individual glands",
    "komprimierte Drüsen": "compressed glands",
    "schlecht ausgebildete Drüsen": "poorly formed glands",
    "free_text_mapping": {
        "single gland": "individual glands",
        "cribiform": "cribriform glands"
    }
}
```

---

## Soft Label Generation

### The Problem

Multiple pathologists annotate the same region differently:

```
Image: PR482a_A1.jpg, Region: [100:200, 100:200]

Pathologist 1: "individual glands" (class 2)
Pathologist 2: "individual glands" (class 2)
Pathologist 3: "compressed glands" (class 3)
Pathologist 4: "individual glands" (class 2)
Pathologist 5: "poorly formed glands" (class 4)
```

**Naive approach:** Use majority vote → class 2

**GleasonXAI approach:** Create **soft label** → probability distribution

### Soft Label Computation

**Implementation:** [gleason_data.py:60-78](../src/gleasonxai/gleason_data.py#L60-L78)

```python
def prepare_torch_inputs(img, seg_masks, num_classes):
    """
    Args:
        img: PIL Image
        seg_masks: List of binary masks (one per annotator)
        num_classes: Number of classes

    Returns:
        img: torch.Tensor [3, H, W]
        label: torch.Tensor [num_classes, H, W] with soft labels
    """
    img = transforms.functional.to_tensor(img)
    _, H, W = img.shape

    # Initialize accumulator
    label = torch.zeros((H, W, num_classes), dtype=torch.int64)

    # Accumulate one-hot encoded masks
    for seg_mask in seg_masks:
        seg_mask = torch.tensor(seg_mask, dtype=torch.int64)
        one_hot = torch.zeros((H, W, num_classes), dtype=torch.int64)
        one_hot.scatter_(2, seg_mask.unsqueeze(2), 1)
        label += one_hot

    # Average across annotators
    label = label / len(seg_masks)

    # Reorder to [C, H, W]
    label = label.permute([2, 0, 1])

    return img, label
```

### Example Soft Label

For the example above (5 annotators):

```python
# Pixel (150, 150) soft label probabilities:
[
    0.0,  # class 0 (background)
    0.0,  # class 1 (benign)
    0.6,  # class 2 (individual glands) ← 3/5 votes
    0.2,  # class 3 (compressed glands) ← 1/5 votes
    0.2,  # class 4 (poorly formed glands) ← 1/5 votes
    0.0,  # class 5
    ...
]
```

**Advantage:** Captures diagnostic uncertainty and disagreement

---

## Data Preprocessing

### 1. Image Loading

**Function:** `load_tmas()` [gleason_data.py](../src/gleasonxai/gleason_data.py)

```python
def load_tmas(path_to_tmas: Path) -> Dict[str, Path]:
    """
    Recursively find all image files in dataset.

    Returns:
        Dict mapping TMA_identifier → relative path
    """
    tma_paths = {}
    for file_path in path_to_tmas.rglob("*"):
        if file_path.suffix in [".jpg", ".jpeg", ".png"]:
            identifier = file_path.stem
            tma_paths[identifier] = file_path.relative_to(path_to_tmas)
    return tma_paths
```

### 2. Microns Calibration

Different scanners produce images with different **microns per pixel** (µm/px) ratios.

**Problem:** A 512×512 image might represent:
- Scanner A: 0.5 µm/px → 256 µm × 256 µm physical size
- Scanner B: 0.25 µm/px → 128 µm × 128 µm physical size

**Solution:** Rescale all images to **consistent microns/pixel ratio**.

**Implementation:** [scripts/create_downscaled_dataset.py](../scripts/create_downscaled_dataset.py)

```bash
# Create microns-calibrated versions
python scripts/setup.py --calibrate
```

**Effect:**
- All images represent the same physical scale
- Model learns scale-invariant features
- Improves generalization across scanners

### 3. Tissue Masking

**Goal:** Remove background (glass slide, mounting medium) and focus on tissue.

**Implementation:** [gleason_utils.py](../src/gleasonxai/gleason_utils.py)

```python
def tissue_filter_image(img, fill_holes=True, min_size=1000):
    """
    Apply morphological operations to extract tissue mask.

    Steps:
        1. Convert to grayscale
        2. Otsu thresholding
        3. Morphological closing (fill small gaps)
        4. Morphological opening (remove small objects)
        5. Flood-fill from borders (remove artifacts)
        6. Optional: Fill holes inside tissue

    Returns:
        Binary mask [H, W] where 1=tissue, 0=background
    """
    ...
```

**Configuration:** [configs/dataset/tissue_mask_kwargs/unfilled_holes.yaml](../configs/dataset/tissue_mask_kwargs/)

```yaml
fill_holes: false
min_size: 1000
```

**Why it matters:**
- Focuses computation on relevant regions
- Reduces false positives from background
- Improves training efficiency

---

## Data Augmentation

GleasonXAI uses **Albumentations** library for augmentation.

**Location:** [src/gleasonxai/augmentations.py](../src/gleasonxai/augmentations.py)

### Augmentation Philosophy

**Goal:** Make model robust to:
- Scanner variability (color, brightness, sharpness)
- Tissue preparation artifacts
- Rotation and orientation
- Scale variations

### Available Pipelines

#### 1. Default Training Pipeline: Tellez Transforms

**Reference:** Tellez et al., "Whole-Slide Mitosis Detection in H&E Breast Histology" (2018)

**Transformations:**
```python
def tellez_transforms_train(image_size=512):
    return albumentations.Compose([
        # Geometric
        albumentations.Rotate(limit=90, p=0.5),
        albumentations.HorizontalFlip(p=0.5),
        albumentations.VerticalFlip(p=0.5),

        # Elastic deformation (tissue folding)
        albumentations.ElasticTransform(
            alpha=120,
            sigma=120 * 0.05,
            alpha_affine=120 * 0.03,
            p=0.3
        ),

        # Color augmentation
        albumentations.HueSaturationValue(
            hue_shift_limit=10,
            sat_shift_limit=20,
            val_shift_limit=10,
            p=0.5
        ),

        # Brightness/contrast
        albumentations.RandomBrightnessContrast(
            brightness_limit=0.1,
            contrast_limit=0.1,
            p=0.5
        ),

        # Blur (out-of-focus regions)
        albumentations.GaussianBlur(
            blur_limit=(3, 7),
            p=0.3
        ),

        # Noise
        albumentations.GaussNoise(
            var_limit=(10.0, 50.0),
            p=0.3
        ),

        # Resize
        albumentations.Resize(image_size, image_size),

        # Normalization (ImageNet stats)
        albumentations.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])
```

**Why ImageNet normalization?** EfficientNet encoder pretrained on ImageNet.

#### 2. Scale-Crop Augmentation

**Purpose:** Multi-scale training for better generalization.

**Implementation:**
```python
def create_scaling_crop(image_size=512, zoom_factors=[0.8, 1.0, 1.2]):
    return albumentations.Compose([
        # Resize to largest dimension
        albumentations.SmallestMaxSize(max_size=int(image_size * 1.5)),

        # Random scale
        albumentations.RandomScale(
            scale_limit=(zoom_factors[0] - 1, zoom_factors[-1] - 1),
            p=1.0
        ),

        # Random crop to target size
        albumentations.RandomCrop(height=image_size, width=image_size),

        # ... (color/geometric augmentations)
    ])
```

**Effect:** Model sees tissue at different magnifications.

#### 3. Zoom Crop

**Purpose:** Random zoom and crop.

```python
def create_zoom_crop(image_size=512, zoom_range=(0.8, 1.2)):
    return albumentations.Compose([
        albumentations.RandomScale(scale_limit=zoom_range, p=1.0),
        albumentations.RandomCrop(height=image_size, width=image_size),
        # ... (other transforms)
    ])
```

#### 4. Validation/Test Pipeline

**No augmentation** (except normalization):

```python
def normalize_only_transform(image_size=512):
    return albumentations.Compose([
        albumentations.Resize(image_size, image_size),
        albumentations.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])
```

### Configuration Files

**Location:** [configs/augmentations/](../configs/augmentations/)

Available configs:
- `microns_calibrated_sw.yaml` - With sliding window support (default)
- `microns_calibrated.yaml` - Standard augmentation
- `no_zoom.yaml` - Augmentation without zoom
- `light_augs.yaml` - Minimal augmentation
- `default.yaml` - Legacy variant

**Example:** [configs/augmentations/microns_calibrated_sw.yaml](../configs/augmentations/microns_calibrated_sw.yaml)

```yaml
train:
  _target_: gleasonxai.augmentations.tellez_transforms_train
  image_size: ${dataset.image_size}

val:
  _target_: gleasonxai.augmentations.normalize_only_transform
  image_size: ${dataset.image_size}

test:
  _target_: gleasonxai.augmentations.normalize_only_transform
  image_size: ${dataset.image_size}

sliding_window_inference:
  enable: true
  roi_size: [512, 512]
  sw_batch_size: 4
  overlap: 0.5
  mode: gaussian
```

---

## Train/Val/Test Splitting

### Split Ratios

**Default:** 70% train, 15% val, 15% test

**Configuration:** [configs/dataset/segmentation_microns_calibrated.yaml](../configs/dataset/segmentation_microns_calibrated.yaml)

```yaml
data_split: [0.7, 0.15, 0.15]
```

### Splitting Strategy

**Method:** PyTorch `random_split` with **deterministic seeding**.

**Implementation:** [gleason_data.py](../src/gleasonxai/gleason_data.py)

```python
from torch.utils.data import random_split

# Set seed for reproducibility
generator = torch.Generator().manual_seed(seed)

# Split dataset
train_size = int(len(dataset) * 0.7)
val_size = int(len(dataset) * 0.15)
test_size = len(dataset) - train_size - val_size

train_dataset, val_dataset, test_dataset = random_split(
    dataset,
    [train_size, val_size, test_size],
    generator=generator
)
```

**Key property:** Same seed → same split (reproducibility)

### Stratification

**Note:** Current implementation uses **random splitting**, not stratified splitting.

**Why?** With 10 classes and multi-label soft annotations, stratification is complex.

**Alternative approaches:**
- Stratify by dominant Gleason pattern
- Stratify by image source (TMA, Gleason2019, etc.)
- Custom stratification by patient ID (not currently implemented)

---

## DataLoader Configuration

### PyTorch DataLoader

**Configuration:** [configs/dataloader/](../configs/dataloader/)

#### Default Configuration

[configs/dataloader/default.yaml](../configs/dataloader/default.yaml)

```yaml
batch_size: 4
num_workers: 4
pin_memory: true
persistent_workers: true
```

#### Large Batch Size Configuration

[configs/dataloader/large_batch_size.yaml](../configs/dataloader/large_batch_size.yaml)

```yaml
batch_size: 8
num_workers: 8
pin_memory: true
persistent_workers: true
```

### Parameter Explanations

| Parameter | Value | Explanation |
|-----------|-------|-------------|
| `batch_size` | 4-8 | Number of images per batch. Limited by GPU memory. |
| `num_workers` | 4-8 | Number of CPU processes for data loading. Set to number of CPU cores. |
| `pin_memory` | true | Pin memory for faster CPU→GPU transfer. Always use for GPU training. |
| `persistent_workers` | true | Keep workers alive between epochs. Faster for multi-epoch training. |

### Memory Considerations

**Rule of thumb:** Batch size × image size determines GPU memory usage.

| Batch Size | Image Size | GPU Memory | Recommended GPU |
|------------|------------|------------|-----------------|
| 4 | 512×512 | ~6GB | RTX 3090, A100 |
| 8 | 512×512 | ~10GB | A100 |
| 16 | 512×512 | ~18GB | A100 40GB |
| 4 | 1024×1024 | ~20GB | A100 40GB |

**Tip:** Use `batch_size=4` as default, increase if you have more GPU memory.

---

## Complete Data Pipeline Flow

```
┌─────────────────────────────────────────────────────────────┐
│                    Raw Data Files                           │
│  • Images: TMA/*.jpg, Gleason2019/*.png                     │
│  • Annotations: final_filtered_explanations_df.csv          │
│  • Label mapping: label_remapping.json                      │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ↓
┌─────────────────────────────────────────────────────────────┐
│             GleasonX.__init__() - Setup                     │
│  1. Load TMA file paths (load_tmas)                         │
│  2. Load annotations CSV (load_explanations)                │
│  3. Match annotations to images (postprocess_df)            │
│  4. Apply label hierarchy mapping                           │
│  5. Filter by data sources                                  │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ↓
┌─────────────────────────────────────────────────────────────┐
│         random_split() - Train/Val/Test Split               │
│  • Train: 70% (deterministic seed)                          │
│  • Val: 15%                                                 │
│  • Test: 15%                                                │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ↓
┌─────────────────────────────────────────────────────────────┐
│             DataLoader - Batch Creation                     │
│  • Shuffle (train only)                                     │
│  • Multi-worker loading                                     │
│  • Prefetching                                              │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ↓ (per batch)
┌─────────────────────────────────────────────────────────────┐
│          GleasonX.__getitem__() - Load Sample               │
│  1. Load image from disk (PIL.Image.open)                   │
│  2. Load annotations for this image                         │
│  3. Create segmentation masks (create_segmentation_masks)   │
│  4. Generate soft labels (average across annotators)        │
│  5. Apply tissue masking (tissue_filter_image)              │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ↓
┌─────────────────────────────────────────────────────────────┐
│          Albumentations - Data Augmentation                 │
│  • Geometric: rotate, flip, elastic transform               │
│  • Color: hue/saturation, brightness/contrast               │
│  • Noise: Gaussian noise, blur                              │
│  • Crop/scale: random crop, zoom                            │
│  • Normalize: ImageNet mean/std                             │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ↓
┌─────────────────────────────────────────────────────────────┐
│               prepare_torch_inputs()                        │
│  • Convert PIL Image → torch.Tensor [3, H, W]               │
│  • Soft labels → torch.Tensor [num_classes, H, W]           │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ↓
┌─────────────────────────────────────────────────────────────┐
│                  Batch Collation                            │
│  • Stack samples: [B, 3, H, W]                              │
│  • Stack labels: [B, num_classes, H, W]                     │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ↓
                   Model Training
```

---

## Debugging Tips

### 1. Visualize Loaded Data

```python
from gleasonxai.gleason_data import GleasonX
import matplotlib.pyplot as plt

dataset = GleasonX(path="/path/to/data", label_level=1)
img, label, metadata = dataset[0]

# Visualize image
img_np = img.permute(1, 2, 0).numpy()  # [H, W, 3]
plt.imshow(img_np)
plt.title(metadata['TMA_identifier'])
plt.show()

# Visualize label (argmax for visualization)
pred_class = label.argmax(dim=0)  # [H, W]
plt.imshow(pred_class, cmap='tab10')
plt.title('Ground Truth Annotations')
plt.show()
```

### 2. Check Data Statistics

```python
# Print dataset info
print(f"Dataset size: {len(dataset)}")
print(f"Label level: {dataset.label_level}")
print(f"Number of classes: {dataset.num_classes}")
print(f"Sources: {dataset.sources}")

# Check class distribution
class_counts = dataset.get_class_distribution()
print(class_counts)
```

### 3. Verify Augmentations

```python
from gleasonxai.augmentations import tellez_transforms_train

# Apply augmentation multiple times to same image
transform = tellez_transforms_train(image_size=512)

for i in range(5):
    augmented = transform(image=img_np, mask=label_np)
    plt.subplot(1, 5, i+1)
    plt.imshow(augmented['image'])
    plt.title(f'Aug {i+1}')
plt.show()
```

---

## Next Steps

- [01_ARCHITECTURE.md](01_ARCHITECTURE.md) - Understanding the model architecture
- [03_TRAINING.md](03_TRAINING.md) - Training procedures and loss functions
- [06_EVALUATION.md](06_EVALUATION.md) - Evaluation metrics and visualization

---

## References

1. **Albumentations:** https://albumentations.ai/
2. **PyTorch Dataset:** https://pytorch.org/docs/stable/data.html#torch.utils.data.Dataset
3. **Tellez et al.:** "Whole-Slide Mitosis Detection in H&E Breast Histology" (2018)
