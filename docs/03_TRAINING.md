# GleasonXAI Training Guide

Complete guide to training GleasonXAI models, including loss functions, optimization, and best practices.

---

## Table of Contents

1. [Training Overview](#training-overview)
2. [Quick Start](#quick-start)
3. [Loss Functions](#loss-functions)
4. [Optimization](#optimization)
5. [PyTorch Lightning Module](#pytorch-lightning-module)
6. [Training Configuration](#training-configuration)
7. [Metrics and Logging](#metrics-and-logging)
8. [Hyperparameter Tuning](#hyperparameter-tuning)
9. [Training Best Practices](#training-best-practices)
10. [Troubleshooting](#troubleshooting)

---

## Training Overview

GleasonXAI uses **PyTorch Lightning** for training, providing:
- Automatic GPU handling
- Distributed training support
- Checkpointing and early stopping
- Comprehensive logging (W&B, TensorBoard)
- Reproducible experiments

### Training Script Locations

- **Entry point:** [scripts/run_training.py](../scripts/run_training.py)
- **Implementation:** [scripts/train.py](../scripts/train.py)
- **Lightning module:** [src/gleasonxai/lightning_modul.py](../src/gleasonxai/lightning_modul.py)

---

## Quick Start

### Basic Training Command

```bash
# Train on fine-grained explanations (level 1) with SoftDiceLoss
python scripts/run_training.py \
    dataset.label_level=1 \
    loss_functions=soft_dice_balanced \
    experiment=MyFirstExperiment
```

### Training with uv

```bash
# Recommended: Use uv for dependency management
uv run --env-file=.env scripts/run_training.py \
    dataset.label_level=1 \
    loss_functions=soft_dice_balanced \
    experiment=MyExperiment
```

### Output Location

Results saved to: `$EXPERIMENT_LOCATION/MyExperiment/`

```
$EXPERIMENT_LOCATION/MyExperiment/
├── checkpoints/
│   ├── best_model.ckpt          # Best model (val_loss)
│   └── last.ckpt                # Last epoch
├── version_0/                   # Lightning logs
│   ├── hparams.yaml             # Hyperparameters
│   ├── metrics.csv              # Training metrics
│   └── events.out.tfevents.*    # TensorBoard events
└── wandb/                       # W&B logs (if enabled)
```

---

## Loss Functions

GleasonXAI provides **multiple loss functions** optimized for semantic segmentation with soft labels.

### 1. SoftDiceLoss (Recommended)

**Default choice** for GleasonXAI training.

**Implementation:** [src/gleasonxai/model_utils.py:182](../src/gleasonxai/model_utils.py)

```python
class SoftDiceLoss(nn.Module):
    """
    Soft DICE loss for soft label training.

    Args:
        average: 'micro' (pixel-wise) or 'macro' (class-balanced)
        smooth: Smoothing constant to avoid division by zero
    """
    def __init__(self, average='macro', smooth=1e-6):
        super().__init__()
        self.average = average
        self.smooth = smooth

    def forward(self, predictions, targets):
        # predictions: [B, C, H, W] (logits)
        # targets: [B, C, H, W] (soft labels)

        # Softmax to probabilities
        pred_probs = F.softmax(predictions, dim=1)

        if self.average == 'macro':
            # Compute DICE per class, then average
            dice_per_class = []
            for c in range(predictions.shape[1]):
                intersection = (pred_probs[:, c] * targets[:, c]).sum()
                denominator = pred_probs[:, c].sum() + targets[:, c].sum()
                dice = (2 * intersection + self.smooth) / (denominator + self.smooth)
                dice_per_class.append(dice)
            return 1 - torch.stack(dice_per_class).mean()

        elif self.average == 'micro':
            # Compute DICE globally across all pixels
            intersection = (pred_probs * targets).sum()
            denominator = pred_probs.sum() + targets.sum()
            dice = (2 * intersection + self.smooth) / (denominator + self.smooth)
            return 1 - dice
```

**Why macro averaging?**
- **Class imbalance:** Background and benign dominate pixel count
- **Macro averaging:** Treats all classes equally
- **Effect:** Model learns rare classes (e.g., comedonecrosis)

**Configuration:** [configs/loss_functions/soft_dice_balanced.yaml](../configs/loss_functions/soft_dice_balanced.yaml)

```yaml
_target_: gleasonxai.model_utils.SoftDiceLoss
average: macro
smooth: 1.0e-6
```

**Usage:**
```bash
python scripts/run_training.py loss_functions=soft_dice_balanced
```

### 2. TreeLoss (Multi-Level)

**Hierarchical loss** for training on multiple label levels simultaneously.

**Implementation:** [src/gleasonxai/tree_loss.py:180](../src/gleasonxai/tree_loss.py)

```python
class TreeLoss(nn.Module):
    """
    Multi-level hierarchical loss.

    Jointly optimizes all levels of the label hierarchy with configurable weights.

    Args:
        label_hierarchy: Nested dict defining class relationships
        loss_weights: List of weights per level (e.g., [0.5, 1.0, 0.5])
        base_loss: Base loss function (e.g., SoftDiceLoss)
    """
    def __init__(self, label_hierarchy, loss_weights=None, base_loss=None):
        super().__init__()
        self.label_hierarchy = label_hierarchy
        self.loss_weights = loss_weights or [1.0] * len(levels)
        self.base_loss = base_loss or SoftDiceLoss(average='macro')

    def forward(self, predictions, targets, level=1):
        total_loss = 0.0

        # Compute loss for each level
        for lvl in range(len(self.loss_weights)):
            # Remap predictions and targets to current level
            pred_remapped = self.remap_to_level(predictions, level_src=level, level_dst=lvl)
            target_remapped = self.remap_to_level(targets, level_src=level, level_dst=lvl)

            # Compute base loss
            loss_lvl = self.base_loss(pred_remapped, target_remapped)

            # Weight and accumulate
            total_loss += self.loss_weights[lvl] * loss_lvl

        return total_loss
```

**Configuration:** [configs/loss_functions/soft_dice_balanced_multilevel.yaml](../configs/loss_functions/soft_dice_balanced_multilevel.yaml)

```yaml
_target_: gleasonxai.tree_loss.TreeLoss
label_hierarchy: ${dataset.label_hierarchy}
loss_weights: [0.5, 1.0, 0.5]  # [level0, level1, level2]
base_loss:
  _target_: gleasonxai.model_utils.SoftDiceLoss
  average: macro
```

**Why multi-level?**
- **Regularization:** Enforces consistency across levels
- **Better generalization:** Learns shared features
- **Flexible inference:** Can predict at any level

**Usage:**
```bash
python scripts/run_training.py loss_functions=soft_dice_balanced_multilevel
```

### 3. Cross-Entropy Loss

**Standard classification loss**.

**Configuration:** [configs/loss_functions/one_hot_ce.yaml](../configs/loss_functions/one_hot_ce.yaml)

```yaml
_target_: gleasonxai.loss_functions.OneHotCE
```

**When to use:** Baseline comparison (not recommended for soft labels)

### 4. MONAI DiceLoss

**Alternative DICE implementation** from MONAI library.

**Configuration:** [configs/loss_functions/dice_loss.yaml](../configs/loss_functions/dice_loss.yaml)

```yaml
_target_: gleasonxai.loss_functions.OneHotDICE
```

**Difference from SoftDiceLoss:** Uses MONAI's implementation (minor numerical differences)

### 5. JDT Losses (Experimental)

**Jaccard-Dice-Tversky loss family** for research.

**Implementation:** [src/gleasonxai/jdt_losses.py:437](../src/gleasonxai/jdt_losses.py)

**Reference papers:**
- Jaccard Metric Losses (arXiv:2302.05666)
- Dice Semimetric Losses (arXiv:2303.16296)

**Not recommended** for standard use.

### Loss Function Comparison

| Loss Function | Soft Labels | Class Balance | Multi-Level | Recommended |
|---------------|-------------|---------------|-------------|-------------|
| **SoftDiceLoss (macro)** | ✅ | ✅ | ❌ | ✅ Best default |
| TreeLoss | ✅ | ✅ | ✅ | ✅ For multi-level |
| Cross-Entropy | ❌ | ❌ | ❌ | ❌ Baseline only |
| MONAI DiceLoss | ✅ | ✅ | ❌ | ⚠️ Alternative |
| JDT Losses | ✅ | ✅ | ❌ | ❌ Experimental |

---

## Optimization

### Optimizer: Adam

**Default:** Adam optimizer with weight decay (AdamW variant)

**Configuration:** [configs/optimization/optuna_tuned.yaml](../configs/optimization/optuna_tuned.yaml)

```yaml
optimizer:
  _target_: torch.optim.Adam
  lr: 1.0e-3
  weight_decay: 0.02

scheduler: null  # No learning rate scheduler

early_stopping:
  patience: 3
  mode: min
  monitor: val_loss_epoch
```

### Hyperparameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| Learning rate | 1e-3 | Base learning rate |
| Weight decay | 0.02 | L2 regularization (prevents overfitting) |
| Batch size | 4-8 | Limited by GPU memory |
| Max epochs | 400 | Early stopping typically stops before this |
| Patience | 3 | Stop if no improvement for 3 epochs |

### Learning Rate Scheduling

**Current:** No learning rate scheduler (constant lr)

**Alternative (not default):** Cosine annealing, ReduceLROnPlateau

To add scheduler, modify [configs/optimization/](../configs/optimization/):

```yaml
scheduler:
  _target_: torch.optim.lr_scheduler.CosineAnnealingLR
  T_max: ${trainer.max_epochs}
  eta_min: 1.0e-6
```

### Early Stopping

**Strategy:** Monitor `val_loss_epoch`, stop if no improvement for 3 consecutive epochs

**Why patience=3?**
- Too small (1): May stop too early
- Too large (10): Wastes compute
- 3: Good balance

**Configuration:** [src/gleasonxai/lightning_modul.py](../src/gleasonxai/lightning_modul.py)

```python
from pytorch_lightning.callbacks import EarlyStopping

early_stop_callback = EarlyStopping(
    monitor='val_loss_epoch',
    patience=3,
    mode='min',
    verbose=True,
)
```

### Gradient Accumulation

**Not used by default**, but can enable for effectively larger batch sizes.

**Usage:**
```yaml
# In trainer config
accumulate_grad_batches: 4  # Effective batch size = 4 * 4 = 16
```

**When to use:**
- GPU memory limited
- Want larger effective batch size for stability

---

## PyTorch Lightning Module

**Class:** `LitSegmenter`

**Location:** [src/gleasonxai/lightning_modul.py:594](../src/gleasonxai/lightning_modul.py)

### Key Methods

```python
class LitSegmenter(LightningModule):
    def __init__(self, model, loss_fn, num_classes, lr=1e-3, ...):
        super().__init__()
        self.model = model
        self.loss_fn = loss_fn
        self.lr = lr
        # Initialize metrics
        initialize_torchmetrics(self, num_classes)

    def forward(self, x):
        """Forward pass through model."""
        return self.model(x)

    def training_step(self, batch, batch_idx):
        """Single training step."""
        images, labels, metadata = batch

        # Forward pass
        predictions = self(images)

        # Compute loss
        loss = self.loss_fn(predictions, labels)

        # Log metrics
        self.log('train_loss', loss, on_step=True, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):
        """Single validation step."""
        images, labels, metadata = batch

        # Forward pass
        predictions = self(images)

        # Compute loss
        loss = self.loss_fn(predictions, labels)

        # Update metrics
        pred_classes = predictions.argmax(dim=1)
        self.accuracy['val_split'][0](pred_classes, labels.argmax(dim=1))

        # Log
        self.log('val_loss', loss)

        return loss

    def configure_optimizers(self):
        """Setup optimizer and scheduler."""
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay
        )
        return optimizer

    def on_validation_epoch_end(self):
        """Compute and log epoch-level metrics."""
        val_acc = self.accuracy['val_split'][0].compute()
        self.log('val_acc_epoch', val_acc)
        self.accuracy['val_split'][0].reset()
```

### Sliding Window Inference Support

For large images, use sliding window inference in validation/test:

```python
class LitSegmenter(LightningModule):
    def __init__(self, ..., enable_swi=False, roi_size=(512, 512)):
        ...
        if enable_swi:
            from monai.inferers import SlidingWindowInferer
            self.inferer = SlidingWindowInferer(
                roi_size=roi_size,
                sw_batch_size=4,
                overlap=0.5,
                mode='gaussian'
            )

    def validation_step(self, batch, batch_idx):
        images, labels, metadata = batch

        if self.enable_swi:
            # Use sliding window for large images
            predictions = self.inferer(images, self.model)
        else:
            # Standard forward pass
            predictions = self(images)

        # ... rest of validation logic
```

---

## Training Configuration

### Master Config

**Location:** [configs/config.yaml](../configs/config.yaml)

```yaml
save_metric: val_loss_epoch
metric_direction: minimize
seed: null
task: segmentation
experiment: ???  # MUST specify via command line

defaults:
  - _self_
  - dataset: segmentation_microns_calibrated
  - dataloader: large_batch_size
  - model: segmentation_efficientnet
  - optimization: optuna_tuned
  - trainer: small_epochs
  - logger: default
  - loss_functions: ???  # MUST specify via command line
  - augmentations: microns_calibrated_sw
  - hparam_search: null
  - optuna: null
```

### Overriding Defaults

**Command line override:**
```bash
python scripts/run_training.py \
    dataset.label_level=0 \
    model=segmentation_deeplabv3p_efficientnet \
    optimization.optimizer.lr=5e-4 \
    trainer.max_epochs=200 \
    experiment=CustomExperiment
```

**Hydra syntax:**
- Dot notation: `dataset.label_level=0`
- Config group: `model=segmentation_resnet`
- Nested params: `optimization.optimizer.lr=5e-4`

---

## Metrics and Logging

### Tracked Metrics

**Per-batch metrics:**
- `train_loss` (logged every step)

**Per-epoch metrics:**
- `train_loss_epoch`
- `val_loss_epoch` (monitored for early stopping)
- `val_acc_epoch` (micro-average accuracy)
- `val_b_acc_epoch` (macro-average accuracy, class-balanced)
- `val_DICE_epoch` (micro-average DICE)
- `val_b_DICE_epoch` (macro-average DICE)
- `val_soft_DICE_epoch` (soft DICE metric)
- `val_f1_epoch` (F1 score)
- `val_auroc_epoch` (AUROC)
- `val_L1_epoch` (L1 calibration error)

**Test metrics:**
- Same as validation, prefixed with `test_`

### Metric Initialization

**Implementation:** [src/gleasonxai/lightning_modul.py:27](../src/gleasonxai/lightning_modul.py#L27)

```python
def initialize_torchmetrics(nn_module, num_classes, metrics="all"):
    """
    Initialize all torchmetrics for tracking.

    Args:
        nn_module: LightningModule instance
        num_classes: Number of segmentation classes
        metrics: 'all' or list of metric names
    """
    # Accuracy (micro and macro)
    nn_module.accuracy = nn.ModuleDict({
        step: nn.ModuleList([
            MulticlassAccuracy(num_classes=num_classes, average='micro')
        ])
        for step in ['train_split', 'val_split', 'test_split']
    })

    nn_module.b_accuracy = nn.ModuleDict({
        step: nn.ModuleList([
            MulticlassAccuracy(num_classes=num_classes, average='macro')
        ])
        for step in ['train_split', 'val_split', 'test_split']
    })

    # DICE score
    nn_module.DICE = nn.ModuleDict({
        step: nn.ModuleList([
            Dice(num_classes=num_classes, average='micro')
        ])
        for step in ['train_split', 'val_split', 'test_split']
    })

    # ... (additional metrics)
```

### Weights & Biases Integration

**Configuration:** [configs/logger/default.yaml](../configs/logger/default.yaml)

```yaml
wandb:
  _target_: pytorch_lightning.loggers.WandbLogger
  project: GleasonXAI
  name: ${experiment}
  save_dir: ${experiment_location}
  log_model: false

tensorboard:
  _target_: pytorch_lightning.loggers.TensorBoardLogger
  save_dir: ${experiment_location}
  name: ${experiment}
```

**Enable W&B logging:**
```bash
# Set W&B API key
export WANDB_API_KEY=your_api_key

# Train (W&B logging automatic)
python scripts/run_training.py ...
```

**Disable W&B logging:**
```bash
export WANDB_MODE=disabled
python scripts/run_training.py ...
```

---

## Hyperparameter Tuning

### Optuna Integration

**Configuration:** [configs/hparam_search/optuna.yaml](../configs/hparam_search/optuna.yaml)

```yaml
_target_: optuna.create_study
direction: minimize
study_name: gleason_hparam_search

# Parameter search space
params_to_search:
  - name: lr
    type: loguniform
    low: 1.0e-5
    high: 1.0e-2

  - name: weight_decay
    type: loguniform
    low: 1.0e-4
    high: 1.0e-1

  - name: batch_size
    type: categorical
    choices: [4, 8, 16]

n_trials: 50
```

**Run hyperparameter search:**
```bash
python scripts/run_training.py \
    hparam_search=optuna \
    loss_functions=soft_dice_balanced \
    experiment=OptunaSearch
```

**Result:** Best hyperparameters saved to [configs/optimization/optuna_tuned.yaml](../configs/optimization/optuna_tuned.yaml)

### Manual Hyperparameter Tuning

**Grid search example:**
```bash
# Loop over learning rates
for lr in 1e-4 5e-4 1e-3 5e-3; do
    python scripts/run_training.py \
        optimization.optimizer.lr=$lr \
        experiment=lr_search/lr_${lr}
done
```

---

## Training Best Practices

### 1. Start with Default Config

**Recommendation:** Use `soft_dice_balanced` on `label_level=1`

```bash
python scripts/run_training.py \
    dataset.label_level=1 \
    loss_functions=soft_dice_balanced \
    experiment=baseline
```

**Why?**
- Proven configuration from paper
- Macro-averaging handles class imbalance
- Fine-grained explanations provide best interpretability

### 2. Monitor Validation Loss

**Early stopping monitors `val_loss_epoch`**. Check it's decreasing:

```bash
# View TensorBoard logs
tensorboard --logdir $EXPERIMENT_LOCATION/baseline
```

### 3. Check for Overfitting

**Signs of overfitting:**
- Train loss decreasing, val loss increasing
- Large gap between train and val accuracy

**Solutions:**
- Increase weight decay (0.02 → 0.05)
- Add more augmentation
- Reduce model capacity (EfficientNet-B4 → B3)

### 4. Class Imbalance Handling

**Use macro-averaging losses:**
- `soft_dice_balanced` (macro)
- NOT `soft_dice` (micro)

**Why?** Background and benign classes dominate pixel count.

### 5. Reproducibility

**Set random seed:**
```bash
python scripts/run_training.py \
    seed=42 \
    experiment=reproducible
```

**Effect:** Deterministic data splitting and weight initialization

### 6. Save Best Model

**Automatic:** Best model saved to `checkpoints/best_model.ckpt`

**Metric:** Determined by `save_metric: val_loss_epoch` in config

### 7. Multi-GPU Training

**Enable distributed training:**
```bash
python scripts/run_training.py \
    trainer.devices=2 \
    trainer.strategy=ddp \
    experiment=multi_gpu
```

**Effective batch size:** `batch_size * num_gpus * accumulate_grad_batches`

---

## Troubleshooting

### Issue: Out of Memory (OOM)

**Error:** `RuntimeError: CUDA out of memory`

**Solutions:**
1. Reduce batch size:
   ```bash
   python scripts/run_training.py dataloader.batch_size=2
   ```

2. Reduce image size:
   ```bash
   python scripts/run_training.py dataset.image_size=256
   ```

3. Use gradient checkpointing (not implemented by default)

### Issue: Training Loss Not Decreasing

**Possible causes:**
1. **Learning rate too high:** Try `lr=1e-4`
2. **Loss function mismatch:** Ensure soft labels + soft loss
3. **Data loading error:** Visualize batch to verify labels

**Debug:**
```python
# Add to training script
batch = next(iter(train_loader))
images, labels, metadata = batch
print(images.shape, labels.shape)
print(labels.min(), labels.max())  # Should be [0, 1] for soft labels
```

### Issue: Validation Loss Fluctuating

**Cause:** Small validation set or high variability

**Solutions:**
- Increase validation set size (adjust `data_split`)
- Use more stable metric for early stopping (`val_acc_epoch`)

### Issue: Slow Training

**Causes:**
1. **Slow data loading:** Increase `num_workers`
2. **Small batch size:** Increase if GPU allows
3. **Augmentation overhead:** Reduce augmentation complexity

**Benchmark data loading:**
```python
import time
for i, batch in enumerate(train_loader):
    start = time.time()
    images, labels, metadata = batch
    print(f"Batch {i}: {time.time() - start:.3f}s")
    if i > 10: break
```

### Issue: Checkpoints Not Saving

**Check:**
1. `$EXPERIMENT_LOCATION` environment variable set
2. Write permissions on directory
3. Disk space available

**Debug:**
```bash
echo $EXPERIMENT_LOCATION
ls -lh $EXPERIMENT_LOCATION
```

---

## Example Training Workflows

### Workflow 1: Reproduce Paper Results

```bash
# Train 3 models with different seeds
for seed in 1 2 3; do
    python scripts/run_training.py \
        dataset.label_level=1 \
        loss_functions=soft_dice_balanced \
        seed=$seed \
        experiment=paper_reproduction/model_${seed}
done
```

### Workflow 2: Compare Loss Functions

```bash
# Try different losses
for loss in soft_dice_balanced dice_loss one_hot_ce; do
    python scripts/run_training.py \
        dataset.label_level=1 \
        loss_functions=$loss \
        experiment=loss_comparison/$loss
done
```

### Workflow 3: Multi-Level Training

```bash
# Train on all three label levels
for level in 0 1 2; do
    python scripts/run_training.py \
        dataset.label_level=$level \
        loss_functions=soft_dice_balanced \
        experiment=level_comparison/level_${level}
done
```

---

## Next Steps

- [04_CONFIGURATION.md](04_CONFIGURATION.md) - Deep dive into Hydra configuration system
- [05_INFERENCE.md](05_INFERENCE.md) - Running inference on trained models
- [06_EVALUATION.md](06_EVALUATION.md) - Evaluating model performance

---

## References

1. **PyTorch Lightning:** https://pytorch-lightning.readthedocs.io/
2. **Optuna:** https://optuna.org/
3. **DICE Loss:** Milletari et al., "V-Net" (2016)
4. **Soft Labels:** Hinton et al., "Distilling Knowledge in Neural Networks" (2015)
