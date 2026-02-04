"""
Script to evaluate level 0 (Gleason pattern) metrics by remapping level 1 predictions.

This script:
1. Loads existing level 1 predictions (logits)
2. Converts to probabilities using softmax
3. Remaps from level 1 (10 classes) to level 0 (4 classes) by summing probabilities
4. Evaluates with default 0.5 threshold (for comparison with paper)
5. Optionally optimizes thresholds on level 0 for best performance
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Tuple

import hydra
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from tqdm import tqdm

from gleasonxai.tree_loss import generate_label_hierarchy
from gleasonxai.gleason_data import prepare_torch_inputs
from PIL import Image

# Helper to flush stdout immediately for cluster logs
def print_flush(*args, **kwargs):
    print(*args, **kwargs)
    sys.stdout.flush()


def load_labels_only(dataset, indices: list) -> list:
    """
    Load labels directly from seg masks without loading TMA images.
    This is much more memory efficient (~90GB → ~5GB).
    """
    labels = []

    for idx in tqdm(indices, desc="Loading labels (memory-efficient)"):
        img_name = dataset.used_slides[idx]

        # Validate that segmentation masks exist
        if img_name not in dataset.seg_paths:
            raise RuntimeError(
                f"Sample {idx} ({img_name}) has no pre-computed segmentation masks. "
                f"Dataset split: {dataset.split}. Available samples: {len(dataset.seg_paths)}. "
                f"This might indicate a preprocessing issue or data split mismatch."
            )

        # Load segmentation masks (small PNGs, not huge TMA images)
        seg_paths = dataset.seg_paths[img_name]
        seg_images = [
            np.array(Image.open(dataset.segmentation_masks_base_path / seg_path))
            for seg_path in seg_paths
        ]

        # Convert seg_images to label tensor (one-hot encoded)
        # We need a dummy image placeholder for prepare_torch_inputs
        H, W = seg_images[0].shape
        img_placeholder = np.zeros((H, W, 3), dtype=np.uint8)

        _, label = prepare_torch_inputs(img_placeholder, seg_images, dataset.num_classes)
        labels.append(label)

    return labels


def load_predictions(pred_path: Path) -> list:
    """Load predictions from .pt file."""
    print(f"Loading predictions from {pred_path}")
    # Force CPU to avoid device mismatch with labels loaded from PNGs
    preds_list = torch.load(pred_path, map_location='cpu')
    # Remove batch dimension: each tensor shape (num_classes, H, W)
    preds_list = [p.squeeze(0) for p in preds_list]
    print(f"  Loaded {len(preds_list)} samples, shape: {preds_list[0].shape}")
    return preds_list


def convert_to_probabilities(logits_list: list) -> list:
    """Convert logits to probabilities using softmax."""
    return [F.softmax(logits, dim=0) for logits in logits_list]


def remap_to_level0(probs_list: list, remapping_dict: Dict[int, list]) -> list:
    """
    Remap level 1 probabilities to level 0 by summing probabilities.

    Args:
        probs_list: List of level 1 probabilities, each shape (10, H, W)
        remapping_dict: Dictionary mapping level 0 class -> list of level 1 classes
                       e.g., {0: [0], 1: [1, 2], 2: [3, 4, 5, 6], 3: [7, 8, 9]}

    Returns:
        List of level 0 probabilities, each shape (4, H, W)
    """
    level0_probs = []

    for probs in tqdm(probs_list, desc="Remapping level 1 → level 0"):
        num_level0_classes = len(remapping_dict)
        C, H, W = probs.shape
        level0_prob = torch.zeros((num_level0_classes, H, W), dtype=probs.dtype, device=probs.device)

        for level0_class, level1_classes in remapping_dict.items():
            # Sum probabilities from all level 1 classes that map to this level 0 class
            for level1_class in level1_classes:
                level0_prob[level0_class] += probs[level1_class]

        level0_probs.append(level0_prob)

    return level0_probs


def apply_thresholds(probs_list: list, thresholds: Dict[int, float]) -> list:
    """Apply per-class thresholds to probabilities."""
    binary_preds_list = []

    for probs in probs_list:
        binary_preds = torch.zeros_like(probs)
        for class_idx, threshold in thresholds.items():
            binary_preds[class_idx, :, :] = (probs[class_idx, :, :] >= threshold).float()
        binary_preds_list.append(binary_preds)

    return binary_preds_list


def evaluate_metrics(preds_list: list, labels_list: list, num_classes: int) -> Dict[str, float]:
    """Compute evaluation metrics."""
    metrics = {}

    # Per-class metrics
    for class_idx in range(num_classes):
        # Concatenate all samples for this class
        class_preds = torch.cat([preds[class_idx, :, :].flatten() for preds in preds_list])
        class_labels_raw = torch.cat([labels[class_idx, :, :].flatten() for labels in labels_list])

        # Binarize labels
        class_labels = (class_labels_raw >= 0.5).float()

        # Dice coefficient
        intersection = (class_preds * class_labels).sum()
        union = class_preds.sum() + class_labels.sum()
        dice = (2 * intersection) / (union + 1e-8) if union > 0 else 0

        # IoU
        intersection = (class_preds * class_labels).sum()
        union = ((class_preds + class_labels) > 0).sum()
        iou = intersection / (union + 1e-8) if union > 0 else 0

        # Precision and Recall
        tp = (class_preds * class_labels).sum()
        fp = (class_preds * (1 - class_labels)).sum()
        fn = ((1 - class_preds) * class_labels).sum()

        precision = tp / (tp + fp + 1e-8) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn + 1e-8) if (tp + fn) > 0 else 0

        metrics[f"class_{class_idx}_dice"] = dice.item() if torch.is_tensor(dice) else dice
        metrics[f"class_{class_idx}_iou"] = iou.item() if torch.is_tensor(iou) else iou
        metrics[f"class_{class_idx}_precision"] = precision.item() if torch.is_tensor(precision) else precision
        metrics[f"class_{class_idx}_recall"] = recall.item() if torch.is_tensor(recall) else recall

    # Macro-averaged metrics
    metrics["macro_dice"] = np.mean([metrics[f"class_{i}_dice"] for i in range(num_classes)])
    metrics["macro_iou"] = np.mean([metrics[f"class_{i}_iou"] for i in range(num_classes)])
    metrics["macro_precision"] = np.mean([metrics[f"class_{i}_precision"] for i in range(num_classes)])
    metrics["macro_recall"] = np.mean([metrics[f"class_{i}_recall"] for i in range(num_classes)])

    return metrics


def optimize_thresholds_per_class(
    probs_list: list,
    labels_list: list,
    metric: str = "dice"
) -> Dict[int, float]:
    """Optimize threshold for each class independently."""
    num_classes = probs_list[0].shape[0]
    optimal_thresholds = {}

    print(f"\nOptimizing level 0 thresholds using {metric} metric...")

    for class_idx in range(num_classes):
        # Extract class-specific probabilities and labels from all samples
        class_probs = torch.cat([probs[class_idx, :, :].flatten() for probs in probs_list]).cpu().numpy()
        class_labels = torch.cat([labels[class_idx, :, :].flatten() for labels in labels_list]).cpu().numpy()

        # Binarize labels if they're soft
        class_labels_binary = (class_labels >= 0.5).astype(int)

        # Skip if class has no positive samples
        if class_labels_binary.sum() == 0:
            print(f"  Level 0 class {class_idx}: No positive samples, using default threshold 0.5")
            optimal_thresholds[class_idx] = 0.5
            continue

        # Try different thresholds
        thresholds = np.linspace(0.01, 0.99, 99)
        best_score = -1
        best_threshold = 0.5

        for threshold in thresholds:
            preds_binary = (class_probs >= threshold).astype(int)

            if metric == "dice":
                # Compute Dice coefficient
                intersection = (preds_binary * class_labels_binary).sum()
                union = preds_binary.sum() + class_labels_binary.sum()
                score = (2 * intersection) / (union + 1e-8) if union > 0 else 0
            else:
                raise ValueError(f"Unknown metric: {metric}")

            if score > best_score:
                best_score = score
                best_threshold = threshold

        optimal_thresholds[class_idx] = best_threshold
        print(f"  Level 0 class {class_idx}: threshold={best_threshold:.3f}, {metric}={best_score:.4f}")

    return optimal_thresholds


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate level 0 (Gleason pattern) metrics from level 1 predictions"
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="/home/f049r/checkpoints/full_data_test/version_8",
        help="Path to checkpoint directory with level 1 predictions"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Path to save results (default: checkpoint_dir/level0_evaluation)"
    )
    parser.add_argument(
        "--optimize_thresholds",
        action="store_true",
        help="Whether to re-optimize thresholds on level 0 (in addition to default 0.5)"
    )
    parser.add_argument(
        "--skip_train",
        action="store_true",
        help="Skip train set evaluation (saves memory - only evaluate test set)"
    )

    args = parser.parse_args()

    checkpoint_dir = Path(args.checkpoint_dir)
    train_pred_path = checkpoint_dir / "preds" / "pred_train.pt"
    test_pred_path = checkpoint_dir / "preds" / "pred_test.pt"
    config_path = checkpoint_dir / "logs" / "config.yaml"

    # Set output directory
    if args.output_dir is None:
        output_dir = checkpoint_dir / "level0_evaluation"
    else:
        output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("LEVEL 0 (GLEASON PATTERN) EVALUATION")
    print("=" * 80)
    print(f"Checkpoint directory: {checkpoint_dir}")
    print(f"Config path: {config_path}")
    print(f"Output directory: {output_dir}")
    print()

    # Load config to get dataset remapping
    print("=" * 80)
    print("LOADING DATASET TO GET REMAPPING")
    print("=" * 80)
    cfg = OmegaConf.load(config_path)

    # Instantiate dataset to get the remapping dictionary
    import gleasonxai.augmentations as augmentations
    AUGMENTATION_TEST = getattr(cfg.augmentations, "test", cfg.augmentations.eval)
    transforms_test = augmentations.AUGMENTATIONS[AUGMENTATION_TEST]

    dataset = hydra.utils.instantiate(
        cfg.dataset,
        split="train",
        transforms=transforms_test
    )

    # Get the remapping dictionary from dataset
    # The dataset stores exp_numbered_lvl_remapping which is a list of dictionaries
    # We need the first one (level 0 remapping from level 1)
    remapping_dict = dataset.exp_numbered_lvl_remapping[0]
    print(f"Remapping dictionary (level 0 ← level 1):")
    for level0_cls, level1_classes in remapping_dict.items():
        print(f"  Level 0 class {level0_cls} ← Level 1 classes {level1_classes}")
    print()

    # Load training predictions (skip if only evaluating test set)
    if not args.skip_train:
        print("=" * 80)
        print("LOADING AND PROCESSING TRAINING DATA")
        print("=" * 80)
        train_logits = load_predictions(train_pred_path)
        train_probs_l1 = convert_to_probabilities(train_logits)
        train_probs_l0 = remap_to_level0(train_probs_l1, remapping_dict)
        print(f"✓ Remapped to level 0: {len(train_probs_l0)} samples, shape: {train_probs_l0[0].shape}")
        print()
    else:
        print("=" * 80)
        print("SKIPPING TRAINING DATA (--skip_train enabled)")
        print("=" * 80)
        print()

    # Load test predictions
    print("=" * 80)
    print("LOADING AND PROCESSING TEST DATA")
    print("=" * 80)
    test_logits = load_predictions(test_pred_path)
    test_probs_l1 = convert_to_probabilities(test_logits)
    test_probs_l0 = remap_to_level0(test_probs_l1, remapping_dict)
    print(f"✓ Remapped to level 0: {len(test_probs_l0)} samples, shape: {test_probs_l0[0].shape}")

    # Validate probability sums (should be ~1.0 after remapping)
    sample_prob_sums = test_probs_l0[0].sum(dim=0)
    print(f"✓ Validation: Probability sums range [{sample_prob_sums.min():.4f}, {sample_prob_sums.max():.4f}] (should be ~1.0)")
    print()

    # Load labels (we need to remap them too)
    print("=" * 80)
    print("LOADING AND REMAPPING LABELS")
    print("=" * 80)

    # Load train labels only if not skipping train evaluation
    if not args.skip_train:
        dataset_train = hydra.utils.instantiate(
            cfg.dataset,
            split="train",
            transforms=None,  # Don't need transforms for labels
            create_seg_masks=False  # Need this to access pre-computed segmentation masks
        )

        print_flush(f"Loading train labels from {len(dataset_train)} samples (memory-efficient mode)...")
        train_labels_l1 = load_labels_only(dataset_train, list(range(len(dataset_train))))

        # Remap train labels to level 0
        train_labels_l0 = remap_to_level0(train_labels_l1, remapping_dict)
        print_flush(f"✓ Train labels remapped: {len(train_labels_l0)} samples")

        # Free memory
        del train_labels_l1
        import gc
        gc.collect()
    else:
        print_flush("✓ Skipping train labels (--skip_train enabled)")
        train_labels_l0 = None

    # Load test labels
    dataset_test = hydra.utils.instantiate(
        cfg.dataset,
        split="test",
        transforms=None,  # Don't need transforms for labels
        create_seg_masks=False  # Need this to access pre-computed segmentation masks
    )

    print_flush(f"Loading test labels from {len(dataset_test)} samples (memory-efficient mode)...")
    test_labels_l1 = load_labels_only(dataset_test, list(range(len(dataset_test))))

    # Remap test labels to level 0
    test_labels_l0 = remap_to_level0(test_labels_l1, remapping_dict)
    print_flush(f"✓ Test labels remapped: {len(test_labels_l0)} samples")

    # Free memory
    del test_labels_l1
    import gc
    gc.collect()
    print_flush("")
    sys.stdout.flush()

    num_level0_classes = len(remapping_dict)
    default_thresholds = {i: 0.5 for i in range(num_level0_classes)}

    # ============================================================================
    # EVALUATE WITH DEFAULT THRESHOLD (0.5) - For comparison with paper
    # ============================================================================
    print_flush("=" * 80)
    print_flush("EVALUATING WITH DEFAULT THRESHOLD (0.5)")
    print_flush("=" * 80)

    # Evaluate train set (if not skipped)
    if not args.skip_train:
        print_flush("\n[1/2] Train set with default threshold...")
        sys.stdout.flush()
        try:
            train_default_preds = apply_thresholds(train_probs_l0, default_thresholds)
            train_default_metrics = evaluate_metrics(train_default_preds, train_labels_l0, num_level0_classes)
        except Exception as e:
            print_flush(f"ERROR during train evaluation: {e}")
            import traceback
            traceback.print_exc()
            raise

        print("  Results:")
        print(f"    Macro Dice:      {train_default_metrics['macro_dice']:.4f}")
        print(f"    Macro IoU:       {train_default_metrics['macro_iou']:.4f}")
        print(f"    Macro Precision: {train_default_metrics['macro_precision']:.4f}")
        print(f"    Macro Recall:    {train_default_metrics['macro_recall']:.4f}")
    else:
        print_flush("\nSkipping train set evaluation (--skip_train enabled)")
        train_default_metrics = None

    print("\n[1/1] Test set with default threshold..." if args.skip_train else "\n[2/2] Test set with default threshold...")
    test_default_preds = apply_thresholds(test_probs_l0, default_thresholds)
    test_default_metrics = evaluate_metrics(test_default_preds, test_labels_l0, num_level0_classes)

    print("  Results:")
    print(f"    Macro Dice:      {test_default_metrics['macro_dice']:.4f}")
    print(f"    Macro IoU:       {test_default_metrics['macro_iou']:.4f}")
    print(f"    Macro Precision: {test_default_metrics['macro_precision']:.4f}")
    print(f"    Macro Recall:    {test_default_metrics['macro_recall']:.4f}")

    results = {
        'remapping': {k: v for k, v in remapping_dict.items()},
        'default_thresholds': default_thresholds,
        'test_default_metrics': test_default_metrics,
    }

    if not args.skip_train:
        results['train_default_metrics'] = train_default_metrics

    # ============================================================================
    # OPTIONALLY OPTIMIZE THRESHOLDS ON LEVEL 0
    # ============================================================================
    if args.optimize_thresholds:
        if args.skip_train:
            print_flush("\n⚠ WARNING: Cannot optimize thresholds with --skip_train enabled (need train set)")
            print_flush("           Skipping threshold optimization...")
        else:
            print("\n" + "=" * 80)
            print("OPTIMIZING THRESHOLDS ON LEVEL 0 TRAIN SET")
            print("=" * 80)

            optimal_thresholds = optimize_thresholds_per_class(
                train_probs_l0,
                train_labels_l0,
                metric="dice"
            )

            print("\n" + "=" * 80)
            print("EVALUATING WITH OPTIMIZED THRESHOLDS")
            print("=" * 80)

            print("\n[1/2] Train set with optimized thresholds...")
            train_optimized_preds = apply_thresholds(train_probs_l0, optimal_thresholds)
            train_optimized_metrics = evaluate_metrics(train_optimized_preds, train_labels_l0, num_level0_classes)

            print("  Results:")
            print(f"    Macro Dice:      {train_optimized_metrics['macro_dice']:.4f}")
            print(f"    Macro IoU:       {train_optimized_metrics['macro_iou']:.4f}")
            print(f"    Macro Precision: {train_optimized_metrics['macro_precision']:.4f}")
            print(f"    Macro Recall:    {train_optimized_metrics['macro_recall']:.4f}")

            print("\n[2/2] Test set with optimized thresholds...")
            test_optimized_preds = apply_thresholds(test_probs_l0, optimal_thresholds)
            test_optimized_metrics = evaluate_metrics(test_optimized_preds, test_labels_l0, num_level0_classes)

            print("  Results:")
            print(f"    Macro Dice:      {test_optimized_metrics['macro_dice']:.4f}")
            print(f"    Macro IoU:       {test_optimized_metrics['macro_iou']:.4f}")
            print(f"    Macro Precision: {test_optimized_metrics['macro_precision']:.4f}")
            print(f"    Macro Recall:    {test_optimized_metrics['macro_recall']:.4f}")

            # Add optimized results
            results['optimal_thresholds'] = optimal_thresholds
            results['train_optimized_metrics'] = train_optimized_metrics
            results['test_optimized_metrics'] = test_optimized_metrics

            # Compute improvements
            train_improvements = {
                'dice': train_optimized_metrics['macro_dice'] - train_default_metrics['macro_dice'],
                'iou': train_optimized_metrics['macro_iou'] - train_default_metrics['macro_iou'],
                'precision': train_optimized_metrics['macro_precision'] - train_default_metrics['macro_precision'],
                'recall': train_optimized_metrics['macro_recall'] - train_default_metrics['macro_recall']
            }
            test_improvements = {
                'dice': test_optimized_metrics['macro_dice'] - test_default_metrics['macro_dice'],
                'iou': test_optimized_metrics['macro_iou'] - test_default_metrics['macro_iou'],
                'precision': test_optimized_metrics['macro_precision'] - test_default_metrics['macro_precision'],
                'recall': test_optimized_metrics['macro_recall'] - test_default_metrics['macro_recall']
            }

            results['train_improvements'] = train_improvements
            results['test_improvements'] = test_improvements

    # Save results
    results_path = output_dir / "level0_results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n✓ Results saved to {results_path}")

    # ============================================================================
    # SUMMARY
    # ============================================================================
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    class_names = ["Benign", "Gleason Pattern 3", "Gleason Pattern 4", "Gleason Pattern 5"]

    print("\nPer-class results (TEST SET, default threshold 0.5):")
    print("  Class                  Dice    IoU     Precision  Recall")
    print("  " + "-" * 60)
    for i, name in enumerate(class_names):
        print(f"  {name:20s}  {test_default_metrics[f'class_{i}_dice']:.3f}   {test_default_metrics[f'class_{i}_iou']:.3f}   {test_default_metrics[f'class_{i}_precision']:.3f}      {test_default_metrics[f'class_{i}_recall']:.3f}")
    print(f"\n  Macro Average:         {test_default_metrics['macro_dice']:.3f}   {test_default_metrics['macro_iou']:.3f}   {test_default_metrics['macro_precision']:.3f}      {test_default_metrics['macro_recall']:.3f}")

    print("\n" + "=" * 80)
    print(f"COMPARISON WITH PAPER (paper reports: 0.713 Dice)")
    print("=" * 80)
    print(f"Your test macro Dice (default 0.5): {test_default_metrics['macro_dice']:.4f}")
    print(f"Paper test macro Dice:               0.7113 (reported in notebook)")
    print(f"Difference:                          {test_default_metrics['macro_dice'] - 0.7113:.4f} ({(test_default_metrics['macro_dice'] - 0.7113) * 100:.1f}%)")

    print("\n" + "=" * 80)
    print("DONE!")
    print("=" * 80)
    print(f"\nResults saved to: {output_dir}")


if __name__ == "__main__":
    main()
