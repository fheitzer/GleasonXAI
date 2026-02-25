"""
Script to evaluate per-class thresholds for binarizing soft predictions.

This script:
1. Loads the saved predictions from both train and test splits
2. Loads the datasets with the same data split
3. Optimizes per-class thresholds using training set
4. Evaluates the optimized thresholds on both training and test sets
5. Saves and visualizes the results
"""

import argparse
import json
from pathlib import Path
from typing import Dict, Tuple

import hydra
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from sklearn.metrics import f1_score, precision_recall_curve
from torch.utils.data import DataLoader
from tqdm import tqdm

import gleasonxai.augmentations as augmentations


def load_predictions_and_labels(
    pred_path: Path,
    config_path: Path,
    split: str = "test",
    data_split_override: list = None,
    num_workers: int = 4
) -> Tuple[list, list]:
    """
    Load predictions and ground truth labels.

    Args:
        pred_path: Path to saved predictions (.pt file)
        config_path: Path to config.yaml
        split: Dataset split to load ('train', 'test', or 'val')
        data_split_override: Optional override for data_split [train, val, test]
        num_workers: Number of DataLoader workers (default: 4)

    Returns:
        Tuple of (predictions_list, labels_list) as lists of tensors
    """
    # Load predictions
    preds_list = torch.load(pred_path)

    # Remove batch dimension: each tensor shape (num_classes, H, W)
    preds_list = [p.squeeze(0) for p in preds_list]

    # Load config
    cfg = OmegaConf.load(config_path)

    # Override data_split if provided
    if data_split_override is not None:
        print(f"  Overriding data_split: {cfg.dataset.data_split} -> {data_split_override}")
        cfg.dataset.data_split = data_split_override

    # Load dataset with same configuration
    AUGMENTATION_TEST = getattr(cfg.augmentations, "test", cfg.augmentations.eval)
    transforms_test = augmentations.AUGMENTATIONS[AUGMENTATION_TEST]

    dataset = hydra.utils.instantiate(
        cfg.dataset,
        split=split,
        transforms=transforms_test
    )

    # Load all labels
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False
    )

    labels_list = []
    for batch in tqdm(dataloader, desc=f"Loading {split} labels"):
        # Dataset returns [image, label, background_mask]
        _, label, _ = batch
        labels_list.append(label.squeeze(0))

    assert len(preds_list) == len(labels_list), f"Length mismatch: {len(preds_list)} preds vs {len(labels_list)} labels"

    # Verify shapes match for each sample
    for i, (pred, label) in enumerate(zip(preds_list, labels_list)):
        assert pred.shape == label.shape, f"Sample {i}: shape mismatch pred {pred.shape} vs label {label.shape}"

    return preds_list, labels_list


def compute_pixel_probabilities(logits_list: list) -> list:
    """
    Convert logits to probabilities using softmax.

    Args:
        logits_list: List of raw model outputs, each shape (C, H, W)

    Returns:
        List of probabilities, each shape (C, H, W)
    """
    return [F.softmax(logits, dim=0) for logits in logits_list]


def optimize_thresholds_per_class(
    probs_list: list,
    labels_list: list,
    metric: str = "dice"
) -> Dict[int, float]:
    """
    Optimize threshold for each class independently.

    Args:
        probs_list: List of class probabilities, each shape (C, H, W)
        labels_list: List of ground truth labels (soft or hard), each shape (C, H, W)
        metric: Metric to optimize ('dice', 'f1', 'iou')

    Returns:
        Dictionary mapping class_idx -> optimal_threshold
    """
    num_classes = probs_list[0].shape[0]
    optimal_thresholds = {}

    print(f"\nOptimizing thresholds using {metric} metric...")

    for class_idx in range(num_classes):
        # Extract class-specific probabilities and labels from all samples
        class_probs = torch.cat([probs[class_idx, :, :].flatten() for probs in probs_list]).cpu().numpy()
        class_labels = torch.cat([labels[class_idx, :, :].flatten() for labels in labels_list]).cpu().numpy()

        # Binarize labels if they're soft
        class_labels_binary = (class_labels >= 0.5).astype(int)

        # Skip if class has no positive samples
        if class_labels_binary.sum() == 0:
            print(f"  Class {class_idx}: No positive samples, using default threshold 0.5")
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
            elif metric == "f1":
                # F1 score
                score = f1_score(class_labels_binary, preds_binary, zero_division=0)
            elif metric == "iou":
                # IoU
                intersection = (preds_binary * class_labels_binary).sum()
                union = ((preds_binary + class_labels_binary) > 0).sum()
                score = intersection / (union + 1e-8) if union > 0 else 0
            else:
                raise ValueError(f"Unknown metric: {metric}")

            if score > best_score:
                best_score = score
                best_threshold = threshold

        optimal_thresholds[class_idx] = best_threshold
        print(f"  Class {class_idx}: threshold={best_threshold:.3f}, {metric}={best_score:.4f}")

    return optimal_thresholds


def apply_thresholds(
    probs_list: list,
    thresholds: Dict[int, float]
) -> list:
    """
    Apply per-class thresholds to probabilities.

    Args:
        probs_list: List of class probabilities, each shape (C, H, W)
        thresholds: Dictionary mapping class_idx -> threshold

    Returns:
        List of binary predictions, each shape (C, H, W)
    """
    binary_preds_list = []

    for probs in probs_list:
        binary_preds = torch.zeros_like(probs)
        for class_idx, threshold in thresholds.items():
            binary_preds[class_idx, :, :] = (probs[class_idx, :, :] >= threshold).float()
        binary_preds_list.append(binary_preds)

    return binary_preds_list


def evaluate_metrics(
    preds_list: list,
    labels_list: list,
    num_classes: int
) -> Dict[str, float]:
    """
    Compute evaluation metrics.

    Args:
        preds_list: List of binary predictions, each shape (C, H, W)
        labels_list: List of ground truth labels, each shape (C, H, W)
        num_classes: Number of classes

    Returns:
        Dictionary of metrics
    """
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


def visualize_thresholds(
    thresholds: Dict[int, float],
    save_path: Path
):
    """
    Visualize the optimized thresholds per class.

    Args:
        thresholds: Dictionary mapping class_idx -> threshold
        save_path: Path to save the visualization
    """
    classes = sorted(thresholds.keys())
    threshold_values = [thresholds[c] for c in classes]

    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.bar(classes, threshold_values, color='steelblue', alpha=0.7)

    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}',
                ha='center', va='bottom', fontsize=10)

    ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='Default threshold (0.5)')
    ax.set_xlabel('Class Index', fontsize=12)
    ax.set_ylabel('Optimal Threshold', fontsize=12)
    ax.set_title('Per-Class Optimal Thresholds (Optimized on Training Set)', fontsize=14, fontweight='bold')
    ax.set_ylim([0, 1])
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Threshold visualization saved to {save_path}")


def compare_metrics(
    train_default_metrics: Dict[str, float],
    train_optimized_metrics: Dict[str, float],
    test_default_metrics: Dict[str, float],
    test_optimized_metrics: Dict[str, float],
    save_path: Path
):
    """
    Compare metrics between default and optimized thresholds for both train and test.

    Args:
        train_default_metrics: Train metrics with default threshold (0.5)
        train_optimized_metrics: Train metrics with optimized thresholds
        test_default_metrics: Test metrics with default threshold (0.5)
        test_optimized_metrics: Test metrics with optimized thresholds
        save_path: Path to save comparison plot
    """
    metrics_to_compare = ['macro_dice', 'macro_iou', 'macro_precision', 'macro_recall']
    metric_labels = ['Dice', 'IoU', 'Precision', 'Recall']

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Train set comparison
    ax = axes[0]
    train_default_values = [train_default_metrics[m] for m in metrics_to_compare]
    train_optimized_values = [train_optimized_metrics[m] for m in metrics_to_compare]

    x = np.arange(len(metrics_to_compare))
    width = 0.35

    bars1 = ax.bar(x - width/2, train_default_values, width, label='Default (0.5)', color='coral', alpha=0.7)
    bars2 = ax.bar(x + width/2, train_optimized_values, width, label='Optimized', color='seagreen', alpha=0.7)

    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}',
                    ha='center', va='bottom', fontsize=9)

    ax.set_xlabel('Metric', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Training Set: Default vs Optimized Thresholds', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels)
    ax.legend()
    ax.set_ylim([0, 1])
    ax.grid(axis='y', alpha=0.3)

    # Test set comparison
    ax = axes[1]
    test_default_values = [test_default_metrics[m] for m in metrics_to_compare]
    test_optimized_values = [test_optimized_metrics[m] for m in metrics_to_compare]

    bars1 = ax.bar(x - width/2, test_default_values, width, label='Default (0.5)', color='coral', alpha=0.7)
    bars2 = ax.bar(x + width/2, test_optimized_values, width, label='Optimized', color='seagreen', alpha=0.7)

    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}',
                    ha='center', va='bottom', fontsize=9)

    ax.set_xlabel('Metric', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Test Set: Default vs Optimized Thresholds', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels)
    ax.legend()
    ax.set_ylim([0, 1])
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Metrics comparison saved to {save_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate per-class thresholds: optimize on train, evaluate on test"
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="/home/f049r/checkpoints/full_data_test/version_8",
        help="Path to checkpoint directory"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Path to save results (default: checkpoint_dir/threshold_evaluation)"
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="dice",
        choices=["dice", "f1", "iou"],
        help="Metric to optimize thresholds"
    )
    parser.add_argument(
        "--data_split",
        type=float,
        nargs=3,
        default=None,
        metavar=("TRAIN", "VAL", "TEST"),
        help="Override data split ratios [train, val, test] (e.g., 0.9 0.01 0.09)"
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of DataLoader workers (default: 4, use 0 to disable multiprocessing)"
    )

    args = parser.parse_args()

    checkpoint_dir = Path(args.checkpoint_dir)
    train_pred_path = checkpoint_dir / "preds" / "pred_train.pt"
    test_pred_path = checkpoint_dir / "preds" / "pred_test.pt"
    config_path = checkpoint_dir / "logs" / "config.yaml"

    # Set output directory
    if args.output_dir is None:
        output_dir = checkpoint_dir / "threshold_evaluation"
    else:
        output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("="*80)
    print("PER-CLASS THRESHOLD OPTIMIZATION")
    print("="*80)
    print(f"Checkpoint directory: {checkpoint_dir}")
    print(f"Config path: {config_path}")
    print(f"Output directory: {output_dir}")
    print(f"Optimization metric: {args.metric}")
    if args.data_split is not None:
        print(f"Data split override: {args.data_split} (train={args.data_split[0]:.2f}, val={args.data_split[1]:.2f}, test={args.data_split[2]:.2f})")
    print()

    # Load training predictions and labels
    print("="*80)
    print("LOADING TRAINING DATA (for threshold optimization)")
    print("="*80)
    if not train_pred_path.exists():
        raise FileNotFoundError(
            f"Training predictions not found at {train_pred_path}\n"
            f"Please generate training predictions first using the test script."
        )
    train_preds, train_labels = load_predictions_and_labels(
        train_pred_path,
        config_path,
        split="train",
        data_split_override=args.data_split,
        num_workers=args.num_workers
    )
    print(f"✓ Loaded {len(train_preds)} training samples")
    print(f"  Sample shape: {train_preds[0].shape}")

    # Load test predictions and labels
    print("\n" + "="*80)
    print("LOADING TEST DATA (for evaluation)")
    print("="*80)
    if not test_pred_path.exists():
        raise FileNotFoundError(
            f"Test predictions not found at {test_pred_path}\n"
            f"Please generate test predictions first using the test script."
        )
    test_preds, test_labels = load_predictions_and_labels(
        test_pred_path,
        config_path,
        split="test",
        data_split_override=args.data_split,
        num_workers=args.num_workers
    )
    print(f"✓ Loaded {len(test_preds)} test samples")
    print(f"  Sample shape: {test_preds[0].shape}")

    # Convert logits to probabilities
    print("\n" + "="*80)
    print("CONVERTING LOGITS TO PROBABILITIES")
    print("="*80)
    train_probs = compute_pixel_probabilities(train_preds)
    test_probs = compute_pixel_probabilities(test_preds)
    print(f"✓ Converted using softmax")

    num_classes = train_probs[0].shape[0]
    print(f"  Number of classes: {num_classes}")

    # Optimize thresholds on training set
    print("\n" + "="*80)
    print("OPTIMIZING THRESHOLDS ON TRAINING SET")
    print("="*80)
    print(f"Strategy: Optimize per-class thresholds on training data")
    print(f"Metric: {args.metric}")

    optimal_thresholds = optimize_thresholds_per_class(
        train_probs,
        train_labels,
        metric=args.metric
    )

    # Default thresholds for comparison
    default_thresholds = {i: 0.5 for i in range(num_classes)}

    # ============================================================================
    # EVALUATE ON TRAINING SET
    # ============================================================================
    print("\n" + "="*80)
    print("EVALUATING ON TRAINING SET")
    print("="*80)

    # Default threshold (0.5) on train
    print("\n[1/2] Default threshold (0.5)...")
    train_default_preds = apply_thresholds(train_probs, default_thresholds)
    train_default_metrics = evaluate_metrics(train_default_preds, train_labels, num_classes)

    print("  Results:")
    print(f"    Macro Dice:      {train_default_metrics['macro_dice']:.4f}")
    print(f"    Macro IoU:       {train_default_metrics['macro_iou']:.4f}")
    print(f"    Macro Precision: {train_default_metrics['macro_precision']:.4f}")
    print(f"    Macro Recall:    {train_default_metrics['macro_recall']:.4f}")

    # Optimized thresholds on train
    print("\n[2/2] Optimized thresholds...")
    train_optimized_preds = apply_thresholds(train_probs, optimal_thresholds)
    train_optimized_metrics = evaluate_metrics(train_optimized_preds, train_labels, num_classes)

    print("  Results:")
    print(f"    Macro Dice:      {train_optimized_metrics['macro_dice']:.4f}")
    print(f"    Macro IoU:       {train_optimized_metrics['macro_iou']:.4f}")
    print(f"    Macro Precision: {train_optimized_metrics['macro_precision']:.4f}")
    print(f"    Macro Recall:    {train_optimized_metrics['macro_recall']:.4f}")

    # Compute improvement on training set
    train_improvements = {
        'dice': train_optimized_metrics['macro_dice'] - train_default_metrics['macro_dice'],
        'iou': train_optimized_metrics['macro_iou'] - train_default_metrics['macro_iou'],
        'precision': train_optimized_metrics['macro_precision'] - train_default_metrics['macro_precision'],
        'recall': train_optimized_metrics['macro_recall'] - train_default_metrics['macro_recall']
    }

    print("\n  Improvement on TRAINING set:")
    for metric_name, improvement in train_improvements.items():
        sign = "+" if improvement >= 0 else ""
        print(f"    {metric_name.capitalize():12s}: {sign}{improvement:.4f} ({improvement*100:+.2f}%)")

    # ============================================================================
    # EVALUATE ON TEST SET
    # ============================================================================
    print("\n" + "="*80)
    print("EVALUATING ON TEST SET (unseen data)")
    print("="*80)

    # Default threshold (0.5) on test
    print("\n[1/2] Default threshold (0.5)...")
    test_default_preds = apply_thresholds(test_probs, default_thresholds)
    test_default_metrics = evaluate_metrics(test_default_preds, test_labels, num_classes)

    print("  Results:")
    print(f"    Macro Dice:      {test_default_metrics['macro_dice']:.4f}")
    print(f"    Macro IoU:       {test_default_metrics['macro_iou']:.4f}")
    print(f"    Macro Precision: {test_default_metrics['macro_precision']:.4f}")
    print(f"    Macro Recall:    {test_default_metrics['macro_recall']:.4f}")

    # Optimized thresholds on test
    print("\n[2/2] Optimized thresholds (from training set)...")
    test_optimized_preds = apply_thresholds(test_probs, optimal_thresholds)
    test_optimized_metrics = evaluate_metrics(test_optimized_preds, test_labels, num_classes)

    print("  Results:")
    print(f"    Macro Dice:      {test_optimized_metrics['macro_dice']:.4f}")
    print(f"    Macro IoU:       {test_optimized_metrics['macro_iou']:.4f}")
    print(f"    Macro Precision: {test_optimized_metrics['macro_precision']:.4f}")
    print(f"    Macro Recall:    {test_optimized_metrics['macro_recall']:.4f}")

    # Compute improvement on test set
    test_improvements = {
        'dice': test_optimized_metrics['macro_dice'] - test_default_metrics['macro_dice'],
        'iou': test_optimized_metrics['macro_iou'] - test_default_metrics['macro_iou'],
        'precision': test_optimized_metrics['macro_precision'] - test_default_metrics['macro_precision'],
        'recall': test_optimized_metrics['macro_recall'] - test_default_metrics['macro_recall']
    }

    print("\n  Improvement on TEST set:")
    for metric_name, improvement in test_improvements.items():
        sign = "+" if improvement >= 0 else ""
        print(f"    {metric_name.capitalize():12s}: {sign}{improvement:.4f} ({improvement*100:+.2f}%)")

    # ============================================================================
    # SUMMARY
    # ============================================================================
    print("\n" + "="*80)
    print("SUMMARY OF IMPROVEMENTS")
    print("="*80)

    print("\nTraining Set (where thresholds were optimized):")
    print("  Metric       Default    Optimized   Improvement")
    print("  " + "-"*54)
    for metric_name in ['dice', 'iou', 'precision', 'recall']:
        default_val = train_default_metrics[f'macro_{metric_name}']
        optimized_val = train_optimized_metrics[f'macro_{metric_name}']
        improvement = train_improvements[metric_name]
        sign = "+" if improvement >= 0 else ""
        print(f"  {metric_name.capitalize():11s}  {default_val:.4f}     {optimized_val:.4f}      {sign}{improvement:.4f}")

    print("\nTest Set (unseen data):")
    print("  Metric       Default    Optimized   Improvement")
    print("  " + "-"*54)
    for metric_name in ['dice', 'iou', 'precision', 'recall']:
        default_val = test_default_metrics[f'macro_{metric_name}']
        optimized_val = test_optimized_metrics[f'macro_{metric_name}']
        improvement = test_improvements[metric_name]
        sign = "+" if improvement >= 0 else ""
        print(f"  {metric_name.capitalize():11s}  {default_val:.4f}     {optimized_val:.4f}      {sign}{improvement:.4f}")

    # Save results
    results = {
        'optimal_thresholds': optimal_thresholds,
        'default_thresholds': default_thresholds,
        'train': {
            'default_metrics': train_default_metrics,
            'optimized_metrics': train_optimized_metrics,
            'improvements': train_improvements
        },
        'test': {
            'default_metrics': test_default_metrics,
            'optimized_metrics': test_optimized_metrics,
            'improvements': test_improvements
        },
        'optimization_metric': args.metric,
        'optimization_set': 'train',
        'data_split': args.data_split if args.data_split is not None else 'from_config'
    }

    results_path = output_dir / "threshold_results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n✓ Results saved to {results_path}")

    # Visualize
    print("\n" + "="*80)
    print("GENERATING VISUALIZATIONS")
    print("="*80)
    visualize_thresholds(optimal_thresholds, output_dir / "optimal_thresholds.png")
    compare_metrics(
        train_default_metrics,
        train_optimized_metrics,
        test_default_metrics,
        test_optimized_metrics,
        output_dir / "metrics_comparison.png"
    )

    print("\n" + "="*80)
    print("DONE!")
    print("="*80)
    print(f"\nResults saved to: {output_dir}")
    print("  - threshold_results.json: Complete results with metrics")
    print("  - optimal_thresholds.png: Visualization of optimized thresholds")
    print("  - metrics_comparison.png: Train vs Test comparison")


if __name__ == "__main__":
    main()
