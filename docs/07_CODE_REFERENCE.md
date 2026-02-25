# GleasonXAI Code Reference

Comprehensive API reference for all modules, classes, and functions in GleasonXAI.

---

## Table of Contents

1. [Module Overview](#module-overview)
2. [gleason_data.py](#gleason_datapy)
3. [gleason_utils.py](#gleason_utilspy)
4. [lightning_modul.py](#lightning_modulpy)
5. [loss_functions.py](#loss_functionspy)
6. [model_utils.py](#model_utilspy)
7. [tree_loss.py](#tree_losspy)
8. [augmentations.py](#augmentationspy)
9. [jdt_losses.py](#jdt_lossespy)
10. [Scripts Reference](#scripts-reference)

---

## Module Overview

| Module | Lines | Purpose | Key Classes/Functions |
|--------|-------|---------|----------------------|
| [gleason_data.py](../src/gleasonxai/gleason_data.py) | 606 | Dataset loading | `GleasonX`, `load_tmas`, `prepare_torch_inputs` |
| [gleason_utils.py](../src/gleasonxai/gleason_utils.py) | 226 | Utilities | `tissue_filter_image`, `create_composite_plot` |
| [lightning_modul.py](../src/gleasonxai/lightning_modul.py) | 594 | Training module | `LitSegmenter`, `LitClassifier` |
| [loss_functions.py](../src/gleasonxai/loss_functions.py) | 33 | Loss wrappers | `OneHotCE`, `OneHotDICE` |
| [model_utils.py](../src/gleasonxai/model_utils.py) | 182 | Model utilities | `SoftDiceLoss`, `LabelRemapper` |
| [tree_loss.py](../src/gleasonxai/tree_loss.py) | 180 | Hierarchical loss | `TreeLoss`, `parse_label_hierarchy` |
| [augmentations.py](../src/gleasonxai/augmentations.py) | 306 | Augmentation | `tellez_transforms_train`, `create_zoom_crop` |
| [jdt_losses.py](../src/gleasonxai/jdt_losses.py) | 437 | Research losses | `JDTLoss`, `SoftDICEMetric` |

---

## gleason_data.py

**Location:** [src/gleasonxai/gleason_data.py](../src/gleasonxai/gleason_data.py)

**Purpose:** Core dataset class for loading Gleason grading data with multi-annotator soft labels.

### Classes

#### `GleasonX(torch.utils.data.Dataset)`

**Primary dataset class** for GleasonXAI.

**Constructor:**
```python
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
```

**Parameters:**
- `path` (str/Path): Root directory containing `GleasonXAI/` folder
- `label_level` (int): Hierarchy level (0=patterns, 1=explanations, 2=sub-explanations)
- `scaling` (str): Image scaling mode ("MicronsCalibrated" or "original")
- `sources` (List[str]): Datasets to include
- `transforms` (Compose): Albumentations pipeline
- `tissue_mask_kwargs` (Dict): Tissue filtering parameters
- `label_remapping` (Dict): Custom label mappings
- `seed` (int): Random seed for reproducibility

**Key Attributes:**
- `self.df`: Pandas DataFrame with all annotations
- `self.tma_paths`: Dict mapping TMA identifiers to file paths
- `self.num_classes`: Number of classes for current label level
- `self.class_names`: List of class names
- `self.label_hierarchy`: Nested dict of class relationships

**Methods:**

##### `__len__() -> int`
Returns number of unique images in dataset.

##### `__getitem__(idx: int) -> Tuple[Tensor, Tensor, Dict]`
Loads and processes a single sample.

**Returns:**
- `image`: Tensor [3, H, W] - RGB image (normalized)
- `label`: Tensor [num_classes, H, W] - Soft label probabilities
- `metadata`: Dict with:
  - `TMA_identifier`: Image filename
  - `num_annotators`: Number of pathologists who annotated this image
  - `label_level`: Current label hierarchy level

**Example:**
```python
dataset = GleasonX(path="/path/to/data", label_level=1)
image, label, metadata = dataset[0]

print(image.shape)  # torch.Size([3, 512, 512])
print(label.shape)  # torch.Size([10, 512, 512])
print(metadata['TMA_identifier'])  # 'PR482a_A1'
```

### Functions

#### `load_tmas(path_to_tmas: Union[str, Path]) -> Dict[str, Path]`

**Purpose:** Recursively find all image files in dataset directory.

**Parameters:**
- `path_to_tmas`: Root directory to search

**Returns:**
- Dict mapping TMA identifier → relative file path

**Example:**
```python
tma_paths = load_tmas(Path("/data/GleasonXAI/TMA/MicronsCalibrated"))
# {
#   'PR482a_A1': PosixPath('PR482a_A1.jpg'),
#   'PR482a_A2': PosixPath('PR482a_A2.jpg'),
#   ...
# }
```

#### `load_explanations(path: Path, explanation_file: str = "explanations_df.csv") -> pd.DataFrame`

**Purpose:** Load pathologist annotations from CSV.

**Parameters:**
- `path`: Directory containing CSV file
- `explanation_file`: CSV filename

**Returns:**
- Pandas DataFrame with columns:
  - `TMA`: Image filename
  - `explanations`: Text description (normalized to lowercase)
  - `points`: Polygon coordinates (as string)
  - `annotator_id`: Pathologist identifier

**Example:**
```python
df = load_explanations(Path("/data/GleasonXAI"))
print(df.head())
#   TMA              explanations         points                    annotator_id
# 0 PR482a_A1.jpg    individual glands    [[100,200], [150,200]...] pathologist_042
```

#### `postprocess_df(...) -> pd.DataFrame`

**Purpose:** Process annotations DataFrame to match with TMA files and apply label hierarchy.

**Parameters:**
- `df`: Raw annotations DataFrame
- `tma_paths`: Dict from `load_tmas()`
- `exp_lvl_remapping`: Label hierarchy mapping
- `label_level`: Target level (0, 1, or 2)
- `german_to_english_mapping`: Optional translation dict
- `free_text_mapping`: Optional free-text normalization dict

**Returns:**
- Processed DataFrame with:
  - `TMA_identifier`: Standardized identifier (without grade suffix)
  - `explanation_class`: Integer class label
  - Filtered to match available TMA files

**Implementation details:**
- Removes grade suffixes (`_1`, `_grade3`, etc.) from filenames
- Maps text explanations to integer class labels
- Validates all annotations have corresponding images
- Applies German→English translation if needed
- Normalizes free-text annotations to standard classes

#### `prepare_torch_inputs(img: Image, seg_masks: List[np.ndarray], num_classes: int) -> Tuple[Tensor, Tensor]`

**Purpose:** Convert PIL image and segmentation masks to PyTorch tensors with soft labels.

**Parameters:**
- `img`: PIL Image (RGB)
- `seg_masks`: List of binary masks (one per annotator), each [H, W]
- `num_classes`: Number of segmentation classes

**Returns:**
- `img`: Tensor [3, H, W] (normalized to [0, 1])
- `label`: Tensor [num_classes, H, W] (soft labels, values in [0, 1])

**Algorithm:**
```python
# Pseudocode
for each annotator's mask:
    Create one-hot encoding [H, W, num_classes]
    Accumulate in label tensor

label = label / num_annotators  # Average across annotators
label = permute to [num_classes, H, W]
```

**Example:**
```python
img = Image.open("PR482a_A1.jpg")
masks = [mask1, mask2, mask3]  # 3 annotators
img_tensor, label_tensor = prepare_torch_inputs(img, masks, num_classes=10)

# Soft label at pixel (100, 100)
print(label_tensor[:, 100, 100])
# tensor([0.0, 0.0, 0.67, 0.33, 0.0, ...])  # 67% voted class 2, 33% voted class 3
```

#### `get_class_colormaps(num_classes_per_grade: Dict, min: float = 0.1, max: float = 0.9) -> ListedColormap`

**Purpose:** Generate color palettes for visualization (green=Grade 3, blue=Grade 4, red=Grade 5).

**Parameters:**
- `num_classes_per_grade`: Dict with keys "3", "4", "5" → number of classes per grade
- `min`, `max`: Color intensity range (avoid too light/dark)

**Returns:**
- Matplotlib `ListedColormap` with colors for all classes

**Example:**
```python
num_classes_per_grade = {"3": 2, "4": 6, "5": 1}
cmap = get_class_colormaps(num_classes_per_grade)

# Use for visualization
plt.imshow(prediction, cmap=cmap)
```

#### `reformat_dataset_to_flat_structure(path_to_tmas: Path, new_path_to_tmas: Path, file_format: str)`

**Purpose:** Reorganize nested dataset into flat directory structure.

**Parameters:**
- `path_to_tmas`: Source directory (possibly nested)
- `new_path_to_tmas`: Destination directory (flat)
- `file_format`: Target file extension (e.g., ".jpg")

**Effect:**
```
Before:
data/
├── subfolder1/
│   ├── PR482a_A1.png
│   └── PR482a_A2.png
└── subfolder2/
    └── PR482a_A3.png

After:
data_flat/
├── PR482a_A1.jpg
├── PR482a_A2.jpg
└── PR482a_A3.jpg
```

---

## gleason_utils.py

**Location:** [src/gleasonxai/gleason_utils.py](../src/gleasonxai/gleason_utils.py)

**Purpose:** Visualization and image processing utilities.

### Functions

#### `tissue_filter_image(img: np.ndarray, fill_holes: bool = True, min_size: int = 1000) -> np.ndarray`

**Purpose:** Extract tissue mask from H&E stained image using morphological operations.

**Parameters:**
- `img`: RGB image [H, W, 3] (uint8)
- `fill_holes`: Whether to fill holes inside tissue
- `min_size`: Minimum object size (pixels) to keep

**Returns:**
- Binary mask [H, W] (0=background, 1=tissue)

**Algorithm:**
```
1. Convert RGB → Grayscale
2. Otsu thresholding
3. Morphological closing (fill small gaps)
4. Morphological opening (remove small objects)
5. Flood-fill from borders (remove edge artifacts)
6. [Optional] Fill interior holes
7. Remove small connected components (< min_size)
```

**Example:**
```python
import cv2
img = cv2.imread("tissue.jpg")
tissue_mask = tissue_filter_image(img, fill_holes=True, min_size=1000)

# Apply mask
tissue_only = img * tissue_mask[:, :, None]
```

#### `create_composite_plot(...) -> Figure`

**Purpose:** Create visualization of multi-annotator segmentation masks.

**Parameters:**
- `image`: RGB image
- `masks`: List of masks (one per annotator)
- `class_names`: List of class labels
- `colormap`: Matplotlib colormap

**Returns:**
- Matplotlib Figure with:
  - Original image
  - Individual annotator masks
  - Composite (averaged) mask

**Use case:** Visualizing inter-annotator agreement/disagreement

#### `create_single_annotator_segmentation_plot(...) -> Figure`

**Purpose:** Overlay single annotator's segmentation on image.

**Parameters:**
- `image`: RGB image
- `mask`: Segmentation mask [H, W]
- `alpha`: Transparency (0=invisible, 1=opaque)

**Returns:**
- Matplotlib Figure with mask overlay

### Class Name Definitions

```python
CLASS_NAMES_LEVEL_1 = [
    "background",
    "benign",
    "individual glands",
    "compressed glands",
    "poorly formed glands",
    "cribriform glands",
    "glomeruloid glands",
    "solid groups of tumor cells",
    "single cells",
    "cords",
    "comedonecrosis",
]
```

---

## lightning_modul.py

**Location:** [src/gleasonxai/lightning_modul.py](../src/gleasonxai/lightning_modul.py)

**Purpose:** PyTorch Lightning modules for training and evaluation.

### Classes

#### `LitSegmenter(LightningModule)`

**Primary training module** for semantic segmentation.

**Constructor:**
```python
def __init__(
    self,
    model: nn.Module,
    loss_fn: nn.Module,
    num_classes: int,
    lr: float = 1e-3,
    weight_decay: float = 0.02,
    metrics: Union[str, List[str]] = "all",
    enable_swi: bool = False,
    swi_roi_size: Tuple[int, int] = (512, 512),
    swi_batch_size: int = 4,
    swi_overlap: float = 0.5,
):
```

**Parameters:**
- `model`: Segmentation model (e.g., SMP Unet)
- `loss_fn`: Loss function (e.g., SoftDiceLoss)
- `num_classes`: Number of segmentation classes
- `lr`: Learning rate
- `weight_decay`: L2 regularization strength
- `metrics`: Which metrics to track ("all" or list of names)
- `enable_swi`: Enable sliding window inference for validation
- `swi_roi_size`: Sliding window patch size
- `swi_batch_size`: Number of patches to process in parallel
- `swi_overlap`: Overlap ratio between patches

**Key Methods:**

##### `forward(x: Tensor) -> Tensor`
Forward pass through model.

##### `training_step(batch, batch_idx) -> Tensor`
Single training iteration.

**Returns:** Loss value (scalar Tensor)

##### `validation_step(batch, batch_idx) -> Tensor`
Single validation iteration.

**Returns:** Loss value

**Side effects:** Updates metrics (accuracy, DICE, etc.)

##### `test_step(batch, batch_idx) -> Tensor`
Single test iteration (same as validation).

##### `configure_optimizers() -> Optimizer`
Setup optimizer (and optionally scheduler).

**Returns:** Adam optimizer with specified lr and weight_decay

##### `on_validation_epoch_end()`
Compute and log epoch-level metrics after all validation batches.

**Example:**
```python
import segmentation_models_pytorch as smp
from gleasonxai.lightning_modul import LitSegmenter
from gleasonxai.model_utils import SoftDiceLoss

# Create model
model = smp.Unet(
    encoder_name="efficientnet-b4",
    encoder_weights="imagenet",
    in_channels=3,
    classes=10,
)

# Create loss
loss_fn = SoftDiceLoss(average='macro')

# Create Lightning module
lit_model = LitSegmenter(
    model=model,
    loss_fn=loss_fn,
    num_classes=10,
    lr=1e-3,
    weight_decay=0.02,
)

# Train
trainer = pl.Trainer(max_epochs=100)
trainer.fit(lit_model, train_dataloader, val_dataloader)
```

#### `LitClassifier(LightningModule)`

**Classification variant** (not used in main workflow).

Similar to `LitSegmenter` but for image-level classification instead of segmentation.

### Functions

#### `initialize_torchmetrics(nn_module, num_classes, max_num_datasets=1, metrics="all")`

**Purpose:** Initialize all torchmetrics for tracking during training.

**Parameters:**
- `nn_module`: LightningModule instance
- `num_classes`: Number of classes
- `max_num_datasets`: Number of datasets to track separately
- `metrics`: "all" or list of specific metrics

**Side effects:** Adds metric attributes to `nn_module`:
- `nn_module.accuracy`: Micro-average accuracy
- `nn_module.b_accuracy`: Macro-average (balanced) accuracy
- `nn_module.DICE`: Micro-average DICE
- `nn_module.b_DICE`: Macro-average DICE
- `nn_module.soft_DICE`: Soft DICE metric
- `nn_module.conf_matrix`: Confusion matrix
- `nn_module.f1_score`: F1 score
- `nn_module.auroc`: AUROC
- `nn_module.avg_prec`: Average precision
- `nn_module.cal_error`: Calibration error (ECE, MCE)
- `nn_module.L1`: L1 calibration metric

**Usage:**
```python
class MyModule(LightningModule):
    def __init__(self, num_classes):
        super().__init__()
        initialize_torchmetrics(self, num_classes, metrics="all")

    def validation_step(self, batch, batch_idx):
        preds, targets = ...
        # Update metrics
        self.accuracy['val_split'][0](preds, targets)
        self.DICE['val_split'][0](preds, targets)
```

#### `log_metrics(nn_module, split, dataset_idx=0)`

**Purpose:** Log all computed metrics for current split.

**Parameters:**
- `nn_module`: LightningModule with metrics
- `split`: "train_split", "val_split", or "test_split"
- `dataset_idx`: Dataset index (for multi-dataset scenarios)

**Side effects:** Logs metrics to Lightning logger (W&B, TensorBoard)

---

## loss_functions.py

**Location:** [src/gleasonxai/loss_functions.py](../src/gleasonxai/loss_functions.py)

**Purpose:** Wrapper classes for standard loss functions to handle one-hot encoded targets.

### Classes

#### `OneHotCE(nn.Module)`

**Purpose:** Cross-entropy loss for one-hot encoded soft labels.

**Constructor:**
```python
def __init__(self, ignore_index: int = -100):
```

**Forward:**
```python
def forward(self, predictions: Tensor, targets: Tensor) -> Tensor:
    """
    Args:
        predictions: [B, C, H, W] logits
        targets: [B, C, H, W] one-hot soft labels

    Returns:
        Scalar loss
    """
```

**Algorithm:**
```python
# Standard CE requires class indices, not one-hot
# Convert one-hot targets to class indices
targets_indices = targets.argmax(dim=1)  # [B, H, W]

# Compute cross-entropy
loss = F.cross_entropy(predictions, targets_indices)
```

**Note:** Converts soft labels to hard labels (argmax), losing uncertainty information.

#### `OneHotDICE(nn.Module)`

**Purpose:** DICE loss from MONAI for one-hot soft labels.

**Constructor:**
```python
def __init__(self, include_background: bool = True, softmax: bool = True):
```

**Parameters:**
- `include_background`: Whether to include background class in loss
- `softmax`: Apply softmax to predictions

**Forward:**
```python
def forward(self, predictions: Tensor, targets: Tensor) -> Tensor:
```

**Uses:** `monai.losses.DiceLoss` internally

---

## model_utils.py

**Location:** [src/gleasonxai/model_utils.py](../src/gleasonxai/model_utils.py)

**Purpose:** Model utilities, custom losses, and metrics.

### Classes

#### `SoftDiceLoss(nn.Module)`

**Primary loss function** for GleasonXAI (recommended).

**Constructor:**
```python
def __init__(self, average: str = 'macro', smooth: float = 1e-6):
```

**Parameters:**
- `average`: 'micro' (pixel-wise) or 'macro' (class-balanced)
- `smooth`: Smoothing constant to avoid division by zero

**Forward:**
```python
def forward(self, predictions: Tensor, targets: Tensor) -> Tensor:
    """
    Args:
        predictions: [B, C, H, W] logits
        targets: [B, C, H, W] soft labels (probabilities)

    Returns:
        Scalar loss (1 - DICE score)
    """
```

**Algorithm:**
```python
# Apply softmax to get probabilities
pred_probs = F.softmax(predictions, dim=1)

if average == 'macro':
    # Compute DICE per class, then average
    dice_per_class = []
    for c in range(C):
        intersection = (pred_probs[:, c] * targets[:, c]).sum()
        denominator = pred_probs[:, c].sum() + targets[:, c].sum()
        dice_c = (2 * intersection + smooth) / (denominator + smooth)
        dice_per_class.append(dice_c)

    dice_score = mean(dice_per_class)

elif average == 'micro':
    # Compute DICE globally
    intersection = (pred_probs * targets).sum()
    denominator = pred_probs.sum() + targets.sum()
    dice_score = (2 * intersection + smooth) / (denominator + smooth)

return 1 - dice_score  # Loss = 1 - DICE
```

**Why macro?** Handles class imbalance (background/benign dominate pixel counts).

#### `MultiLabelLoss(nn.Module)`

**Purpose:** Binary cross-entropy for multi-label classification.

**Use case:** When multiple classes can be present simultaneously (not typical for GleasonXAI).

#### `LabelRemapper(nn.Module)`

**Purpose:** Map predictions between label hierarchy levels.

**Constructor:**
```python
def __init__(self, label_hierarchy: Dict, source_level: int, target_level: int):
```

**Parameters:**
- `label_hierarchy`: Nested dict of class relationships
- `source_level`: Input level (e.g., 1 = explanations)
- `target_level`: Output level (e.g., 0 = Gleason patterns)

**Forward:**
```python
def forward(self, predictions: Tensor) -> Tensor:
    """
    Args:
        predictions: [B, C_src, H, W] predictions at source level

    Returns:
        Tensor [B, C_tgt, H, W] predictions at target level
    """
```

**Algorithm:** Sum probabilities of child classes to get parent class probability.

**Example:**
```python
# Map level 1 (10 classes) → level 0 (4 classes)
remapper = LabelRemapper(label_hierarchy, source_level=1, target_level=0)

predictions_l1 = model(image)  # [1, 10, 512, 512]
predictions_l0 = remapper(predictions_l1)  # [1, 4, 512, 512]

# Gleason 4 probability = sum of all Gleason 4 explanation probabilities
```

#### `L1CalibrationMetric(torchmetrics.Metric)`

**Purpose:** Compute L1 calibration error.

**Calibration:** How well predicted probabilities match actual frequencies.

**Example:** If model predicts 70% confidence for class A, then ~70% of those predictions should be correct.

**Metric:** Average absolute difference between predicted confidence and actual accuracy.

### Functions

#### `dice_score_hard(predictions: Tensor, targets: Tensor) -> Tensor`

**Purpose:** Compute hard DICE score (for binary or multi-class segmentation).

**Parameters:**
- `predictions`: [B, C, H, W] class predictions (not probabilities)
- `targets`: [B, C, H, W] ground truth (one-hot or soft)

**Returns:** DICE score (higher is better, range [0, 1])

#### `dice_score_soft(predictions: Tensor, targets: Tensor) -> Tensor`

**Purpose:** Compute soft DICE score (using probabilities instead of hard predictions).

**Parameters:**
- `predictions`: [B, C, H, W] probabilities
- `targets`: [B, C, H, W] soft labels

**Returns:** Soft DICE score

**Advantage:** Differentiable and works with soft labels.

---

## tree_loss.py

**Location:** [src/gleasonxai/tree_loss.py](../src/gleasonxai/tree_loss.py)

**Purpose:** Hierarchical label management and multi-level loss functions.

### Classes

#### `TreeLoss(nn.Module)`

**Purpose:** Multi-level hierarchical loss function.

**Constructor:**
```python
def __init__(
    self,
    label_hierarchy: Dict,
    loss_weights: Optional[List[float]] = None,
    base_loss: Optional[nn.Module] = None,
):
```

**Parameters:**
- `label_hierarchy`: Nested dict defining class relationships
  ```python
  {
      "Gleason 3": {
          "individual glands": None,
          "compressed glands": None,
      },
      "Gleason 4": { ... },
  }
  ```
- `loss_weights`: Weight per level (e.g., `[0.5, 1.0, 0.5]`)
- `base_loss`: Base loss function (default: SoftDiceLoss)

**Forward:**
```python
def forward(self, predictions: Tensor, targets: Tensor, level: int) -> Tensor:
    """
    Args:
        predictions: [B, C, H, W] predictions at 'level'
        targets: [B, C, H, W] targets at 'level'
        level: Current training level (0, 1, or 2)

    Returns:
        Weighted sum of losses across all levels
    """
```

**Algorithm:**
```python
total_loss = 0
for lvl in range(num_levels):
    # Remap predictions and targets to current level
    pred_lvl = remap_to_level(predictions, src=level, dst=lvl)
    tgt_lvl = remap_to_level(targets, src=level, dst=lvl)

    # Compute base loss
    loss_lvl = base_loss(pred_lvl, tgt_lvl)

    # Weight and accumulate
    total_loss += loss_weights[lvl] * loss_lvl

return total_loss
```

**Benefit:** Jointly optimizes all levels, enforcing consistency across hierarchy.

### Functions

#### `parse_label_hierarchy(label_hierarchy: Dict) -> Tuple[List[str], List[int]]`

**Purpose:** Extract flat list of class names and their levels from nested hierarchy.

**Parameters:**
- `label_hierarchy`: Nested dict

**Returns:**
- `class_names`: Flat list of all class names
- `class_levels`: Level of each class

**Example:**
```python
hierarchy = {
    "Gleason 3": {
        "individual glands": None,
        "compressed glands": None,
    },
    "Gleason 4": {
        "cribriform glands": None,
    },
}

class_names, class_levels = parse_label_hierarchy(hierarchy)
# class_names = ["Gleason 3", "individual glands", "compressed glands", "Gleason 4", "cribriform glands"]
# class_levels = [0, 1, 1, 0, 1]
```

#### `generate_label_hierarchy(targets: Tensor, label_hierarchy: Dict, current_level: int) -> Dict[int, Tensor]`

**Purpose:** Generate target tensors for all hierarchy levels.

**Parameters:**
- `targets`: [B, C, H, W] targets at `current_level`
- `label_hierarchy`: Hierarchy structure
- `current_level`: Level of input targets

**Returns:**
- Dict mapping level → remapped targets
  ```python
  {
      0: Tensor[B, C0, H, W],  # Gleason patterns
      1: Tensor[B, C1, H, W],  # Explanations
      2: Tensor[B, C2, H, W],  # Sub-explanations
  }
  ```

#### `get_explanation_level_mapping(label_hierarchy: Dict, source_level: int, target_level: int) -> Dict[int, int]`

**Purpose:** Create mapping from source level class indices to target level.

**Returns:**
- Dict mapping source_class_idx → target_class_idx

**Example:**
```python
# Map level 1 → level 0
mapping = get_explanation_level_mapping(hierarchy, source_level=1, target_level=0)
# {
#   2: 1,  # "individual glands" → "Gleason 3"
#   3: 1,  # "compressed glands" → "Gleason 3"
#   4: 2,  # "poorly formed glands" → "Gleason 4"
#   ...
# }
```

---

## augmentations.py

**Location:** [src/gleasonxai/augmentations.py](../src/gleasonxai/augmentations.py)

**Purpose:** Data augmentation pipelines using Albumentations.

### Functions

#### `tellez_transforms_train(image_size: int = 512) -> Compose`

**Purpose:** Primary training augmentation pipeline (Tellez et al., 2018).

**Returns:** Albumentations Compose object

**Transformations:**
- Random rotation (±90°)
- Horizontal/vertical flips
- Elastic deformation
- Hue/saturation/value shifts
- Brightness/contrast adjustments
- Gaussian blur
- Gaussian noise
- Resize to `image_size × image_size`
- Normalize (ImageNet stats)

**Example:**
```python
transform = tellez_transforms_train(image_size=512)

augmented = transform(image=img, mask=mask)
aug_img = augmented['image']
aug_mask = augmented['mask']
```

#### `create_zoom_crop(image_size: int = 512, zoom_factors: Tuple[float, float] = (0.8, 1.2)) -> Compose`

**Purpose:** Random zoom and crop augmentation.

**Parameters:**
- `image_size`: Target size after crop
- `zoom_factors`: (min_zoom, max_zoom)

**Returns:** Albumentations pipeline

**Example:**
```python
transform = create_zoom_crop(image_size=512, zoom_factors=(0.8, 1.2))
```

#### `create_scaling_crop(image_size: int = 512, zoom_factors: List[float] = [0.8, 1.0, 1.2]) -> Compose`

**Purpose:** Multi-scale random crop.

**Algorithm:**
1. Resize to largest dimension
2. Apply random scaling
3. Random crop to `image_size`

**Use case:** Training on multiple scales improves scale invariance.

#### `normalize_only_transform(image_size: int = 512) -> Compose`

**Purpose:** Validation/test pipeline (no augmentation).

**Transformations:**
- Resize to `image_size × image_size`
- Normalize (ImageNet stats)

**Example:**
```python
transform = normalize_only_transform(512)
normalized = transform(image=img, mask=mask)
```

### Normalization Constants

```python
IMAGENET_MEAN = [0.485, 0.456, 0.406]  # RGB
IMAGENET_STD = [0.229, 0.224, 0.225]   # RGB
```

**Why ImageNet?** EfficientNet encoder pretrained on ImageNet.

---

## jdt_losses.py

**Location:** [src/gleasonxai/jdt_losses.py](../src/gleasonxai/jdt_losses.py)

**Purpose:** Experimental loss functions (Jaccard-Dice-Tversky family).

**Note:** Not used in default GleasonXAI training. For research purposes.

### Classes

#### `JDTLoss(nn.Module)`

**Purpose:** Configurable loss combining Jaccard, DICE, and Tversky metrics.

**Constructor:**
```python
def __init__(
    self,
    metric_type: str = "dice",  # "dice", "jaccard", "tversky"
    average: str = "macro",
    smooth: float = 1e-6,
):
```

#### `SoftDICEMetric(nn.Module)`

**Purpose:** Soft DICE metric for evaluation (not loss).

**Similar to SoftDiceLoss** but returns DICE score instead of (1 - DICE).

---

## Scripts Reference

### run_training.py

**Location:** [scripts/run_training.py](../scripts/run_training.py)

**Purpose:** Main entry point for training.

**Usage:**
```bash
python scripts/run_training.py \
    dataset.label_level=1 \
    loss_functions=soft_dice_balanced \
    experiment=MyExperiment
```

**Key features:**
- Hydra configuration management
- Automatic logging setup
- Checkpoint management

### test.py

**Location:** [scripts/test.py](../scripts/test.py)

**Purpose:** Evaluate trained models on test set.

**Usage:**
```bash
python scripts/test.py \
    --experiment_path /path/to/experiment \
    --checkpoint model/checkpoints/best_model.ckpt
```

**Outputs:**
- Test metrics (accuracy, DICE, etc.)
- Confusion matrices
- Per-class performance

### run_gleasonXAI.py

**Location:** [scripts/run_gleasonXAI.py](../scripts/run_gleasonXAI.py)

**Purpose:** Run inference with ensemble of 3 models.

**Usage:**
```bash
python scripts/run_gleasonXAI.py \
    --images /path/to/images \
    --save_path /path/to/output \
    --checkpoint_1 model1.ckpt \
    --checkpoint_2 model2.ckpt \
    --checkpoint_3 model3.ckpt
```

**Features:**
- Ensemble prediction (average of 3 models)
- Sliding window inference for large images
- Color-coded output visualizations

### setup.py

**Location:** [scripts/setup.py](../scripts/setup.py)

**Purpose:** Download and prepare datasets.

**Usage:**
```bash
python scripts/setup.py \
    --gleasonxai_data GleasonXAI_data.zip \
    --download \
    --calibrate
```

**Features:**
- Unzips data archives
- Downloads Gleason2019 dataset (optional)
- Creates microns-calibrated versions
- Extracts model weights

---

## Code Organization Patterns

### Configuration Pattern

All configurable components use:

1. **Hydra dataclass configs** in `configs/`
2. **`_target_`** pointing to class/function
3. **Command-line overrides** via dot notation

**Example:**
```yaml
# configs/model/segmentation_efficientnet.yaml
_target_: segmentation_models_pytorch.Unet
encoder_name: efficientnet-b4
encoder_weights: imagenet
```

### Metric Pattern

All metrics follow:

1. **Initialize** in `__init__` via `initialize_torchmetrics`
2. **Update** in `{train,val,test}_step`
3. **Compute** in `on_{train,val,test}_epoch_end`
4. **Reset** after logging

**Example:**
```python
# Initialize
self.accuracy = MulticlassAccuracy(num_classes)

# Update (in step)
self.accuracy(predictions, targets)

# Compute (in epoch_end)
acc = self.accuracy.compute()
self.log('val_acc', acc)
self.accuracy.reset()
```

### Loss Function Pattern

All losses implement:

```python
class MyLoss(nn.Module):
    def __init__(self, ...):
        super().__init__()
        # Store hyperparameters

    def forward(self, predictions, targets):
        # predictions: [B, C, H, W] logits or probabilities
        # targets: [B, C, H, W] soft labels
        # Returns: scalar loss
        ...
```

---

## Next Steps

- [00_OVERVIEW.md](00_OVERVIEW.md) - High-level project overview
- [01_ARCHITECTURE.md](01_ARCHITECTURE.md) - Model architecture details
- [03_TRAINING.md](03_TRAINING.md) - Training procedures

---

## Further Reading

- **PyTorch Documentation:** https://pytorch.org/docs/
- **PyTorch Lightning:** https://pytorch-lightning.readthedocs.io/
- **Albumentations:** https://albumentations.ai/docs/
- **Segmentation Models PyTorch:** https://smp.readthedocs.io/
