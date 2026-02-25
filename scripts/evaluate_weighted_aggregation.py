"""
Comprehensive evaluation of weighted probability aggregation strategies.

This script compares three strategies for aggregating level 1 probabilities to level 0:
1. Equal weights (baseline) - current approach
2. Threshold-based weights (Option A) - derived from level 1 threshold optimization
3. Optimized weights (Option B) - directly optimized for level 0 Dice score

Usage:
    python scripts/evaluate_weighted_aggregation.py \
        --checkpoint_dir /path/to/checkpoint \
        --threshold_results_path threshold_evaluation/threshold_results.json \
        --output_dir weighted_aggregation_evaluation
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

import hydra
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

# Import from existing scripts
from evaluate_level0_metrics import (
    load_predictions,
    convert_to_probabilities,
    remap_to_level0,
    weighted_remap_to_level0,
    apply_thresholds,
    evaluate_metrics
)

from optimize_aggregation_weights import (
    compute_threshold_based_weights,
    optimize_aggregation_weights
)

import gleasonxai.augmentations as augmentations


def print_flush(*args, **kwargs):
    """Print and flush for cluster logs."""
    print(*args, **kwargs)
    sys.stdout.flush()


def create_comparison_table(
    strategies_results: Dict[str, Tuple[Dict, Dict]],
    remapping_dict: Dict[int, list]
) -> str:
    """
    Create a formatted comparison table for all strategies.

    Args:
        strategies_results: {strategy_name: (train_metrics, test_metrics)}
        remapping_dict: Level 0 to level 1 class remapping

    Returns:
        Formatted table string
    """
    table_lines = []
    table_lines.append("\n" + "="*100)
    table_lines.append("COMPARISON OF AGGREGATION STRATEGIES")
    table_lines.append("="*100)

    # Header
    table_lines.append("\nOVERALL PERFORMANCE:")
    table_lines.append(f"{'Strategy':<25} | {'Train Dice':>11} | {'Test Dice':>11} | {'Improvement':>12}")
    table_lines.append("-" * 100)

    # Get baseline (equal weights)
    baseline_test_dice = None
    for strategy_name, results in strategies_results.items():
        if results is None:
            continue
        train_metrics, test_metrics = results
        if strategy_name.lower().startswith("equal"):
            baseline_test_dice = test_metrics['macro_dice']
            break

    # Print each strategy
    for strategy_name, results in strategies_results.items():
        if results is None:
            table_lines.append(f"{strategy_name:<25} | {'N/A':>11} | {'N/A':>11} | {'N/A':>12}")
            continue

        train_metrics, test_metrics = results
        train_dice = train_metrics['macro_dice']
        test_dice = test_metrics['macro_dice']

        if baseline_test_dice is not None:
            improvement = test_dice - baseline_test_dice
            sign = "+" if improvement >= 0 else ""
            improvement_str = f"{sign}{improvement:.4f}"
        else:
            improvement_str = "N/A"

        table_lines.append(
            f"{strategy_name:<25} | {train_dice:>11.4f} | {test_dice:>11.4f} | {improvement_str:>12}"
        )

    # Per-class breakdown for test set
    table_lines.append("\n\nPER-CLASS PERFORMANCE (TEST SET):")
    class_names = ["Background", "Benign/GP3", "Gleason P. 4", "Gleason P. 5"]

    for class_idx, class_name in enumerate(class_names):
        table_lines.append(f"\n{class_name}:")
        table_lines.append(f"{'Strategy':<25} | {'Dice':>7} | {'IoU':>7} | {'Precision':>10} | {'Recall':>7}")
        table_lines.append("-" * 80)

        for strategy_name, results in strategies_results.items():
            if results is None:
                continue

            _, test_metrics = results
            dice = test_metrics[f'class_{class_idx}_dice']
            iou = test_metrics[f'class_{class_idx}_iou']
            precision = test_metrics[f'class_{class_idx}_precision']
            recall = test_metrics[f'class_{class_idx}_recall']

            table_lines.append(
                f"{strategy_name:<25} | {dice:>7.3f} | {iou:>7.3f} | {precision:>10.3f} | {recall:>7.3f}"
            )

    return "\n".join(table_lines)


def visualize_weights_comparison(
    equal_weights: Dict[int, list],
    threshold_weights: Optional[Dict[int, list]],
    optimized_weights: Dict[int, list],
    remapping_dict: Dict[int, list],
    save_path: Path
):
    """
    Visualize weight distributions for each strategy.

    Creates subplots for each level 0 class showing weight distributions.
    """
    # Only plot multi-child classes
    multi_child_classes = [
        (level0_class, level1_classes)
        for level0_class, level1_classes in remapping_dict.items()
        if len(level1_classes) > 1
    ]

    n_classes = len(multi_child_classes)
    fig, axes = plt.subplots(1, n_classes, figsize=(6 * n_classes, 6))

    if n_classes == 1:
        axes = [axes]

    class_names = {0: "Background", 1: "Benign/GP3", 2: "Gleason Pattern 4", 3: "Gleason Pattern 5"}

    for idx, (level0_class, level1_classes) in enumerate(multi_child_classes):
        ax = axes[idx]

        n_children = len(level1_classes)
        x = np.arange(n_children)
        width = 0.25

        # Plot equal weights
        ax.bar(x - width, equal_weights[level0_class], width,
               label='Equal (baseline)', color='coral', alpha=0.7)

        # Plot threshold-based weights if available
        if threshold_weights is not None:
            ax.bar(x, threshold_weights[level0_class], width,
                   label='Threshold-based', color='steelblue', alpha=0.7)

        # Plot optimized weights
        ax.bar(x + width, optimized_weights[level0_class], width,
               label='Optimized', color='seagreen', alpha=0.7)

        ax.set_xlabel('Level 1 Subtype Index', fontsize=12)
        ax.set_ylabel('Weight', fontsize=12)
        ax.set_title(f'{class_names.get(level0_class, f"Class {level0_class}")}', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([str(c1) for c1 in level1_classes])
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        ax.set_ylim([0, 1])

        # Add value labels on bars
        for bars in ax.containers:
            ax.bar_label(bars, fmt='%.2f', fontsize=9)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print_flush(f"✓ Weights comparison saved to {save_path}")


def visualize_performance_comparison(
    strategies_results: Dict[str, Tuple[Dict, Dict]],
    save_path: Path
):
    """
    Visualize performance comparison across strategies.

    Creates bar charts comparing Dice, IoU, Precision, and Recall.
    """
    metrics_to_compare = ['macro_dice', 'macro_iou', 'macro_precision', 'macro_recall']
    metric_labels = ['Dice', 'IoU', 'Precision', 'Recall']

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    for split_idx, split_name in enumerate(['Train', 'Test']):
        ax = axes[split_idx]

        # Collect data
        strategy_names = []
        strategy_values = {metric: [] for metric in metrics_to_compare}

        for strategy_name, results in strategies_results.items():
            if results is None:
                continue

            train_metrics, test_metrics = results
            metrics = train_metrics if split_name == 'Train' else test_metrics

            strategy_names.append(strategy_name)
            for metric in metrics_to_compare:
                strategy_values[metric].append(metrics[metric])

        # Plot grouped bars
        x = np.arange(len(metrics_to_compare))
        width = 0.25
        colors = ['coral', 'steelblue', 'seagreen']

        for strategy_idx, strategy_name in enumerate(strategy_names):
            offset = (strategy_idx - 1) * width
            values = [strategy_values[metric][strategy_idx] for metric in metrics_to_compare]

            bars = ax.bar(x + offset, values, width,
                         label=strategy_name, color=colors[strategy_idx % len(colors)], alpha=0.7)

            # Add value labels
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.3f}', ha='center', va='bottom', fontsize=9)

        ax.set_xlabel('Metric', fontsize=12)
        ax.set_ylabel('Score', fontsize=12)
        ax.set_title(f'{split_name} Set Performance', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(metric_labels)
        ax.legend()
        ax.set_ylim([0, 1])
        ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print_flush(f"✓ Performance comparison saved to {save_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate weighted probability aggregation strategies"
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
        help="Path to save results (default: checkpoint_dir/weighted_aggregation_evaluation)"
    )
    parser.add_argument(
        "--threshold_results_path",
        type=str,
        default=None,
        help="Path to threshold_results.json from evaluate_per_class_thresholds.py (for Option A)"
    )
    parser.add_argument(
        "--skip_train",
        action="store_true",
        help="Skip training set evaluation (saves memory, but cannot optimize weights)"
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
        output_dir = checkpoint_dir / "weighted_aggregation_evaluation"
    else:
        output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Set default threshold results path if not provided
    if args.threshold_results_path is None:
        args.threshold_results_path = checkpoint_dir / "threshold_evaluation" / "threshold_results.json"
    else:
        args.threshold_results_path = Path(args.threshold_results_path)

    print_flush("="*100)
    print_flush("WEIGHTED PROBABILITY AGGREGATION EVALUATION")
    print_flush("="*100)
    print_flush(f"Checkpoint directory: {checkpoint_dir}")
    print_flush(f"Config path: {config_path}")
    print_flush(f"Output directory: {output_dir}")
    print_flush(f"Threshold results: {args.threshold_results_path}")
    if args.data_split is not None:
        print_flush(f"Data split override: {args.data_split}")
    print_flush()

    # ==================================================================
    # LOAD DATA
    # ==================================================================
    print_flush("="*100)
    print_flush("LOADING DATA")
    print_flush("="*100)

    # Load config to get remapping
    cfg = OmegaConf.load(config_path)

    # Override data_split if provided
    if args.data_split is not None:
        print_flush(f"  Overriding data_split: {cfg.dataset.data_split} -> {args.data_split}")
        cfg.dataset.data_split = list(args.data_split)

    AUGMENTATION_TEST = getattr(cfg.augmentations, "test", cfg.augmentations.eval)
    transforms_test = augmentations.AUGMENTATIONS[AUGMENTATION_TEST]

    dataset = hydra.utils.instantiate(
        cfg.dataset,
        split="train",
        transforms=transforms_test
    )

    remapping_dict = dataset.exp_numbered_lvl_remapping[0]
    print_flush(f"Remapping dictionary (level 0 ← level 1):")
    for level0_cls, level1_classes in remapping_dict.items():
        print_flush(f"  Level 0 class {level0_cls} ← Level 1 classes {level1_classes}")
    print_flush()

    # Load predictions
    if not args.skip_train:
        print_flush("Loading training predictions...")
        train_logits = load_predictions(train_pred_path)
        train_probs_l1 = convert_to_probabilities(train_logits)
        print_flush(f"✓ Loaded {len(train_probs_l1)} training samples")

    print_flush("Loading test predictions...")
    test_logits = load_predictions(test_pred_path)
    test_probs_l1 = convert_to_probabilities(test_logits)
    print_flush(f"✓ Loaded {len(test_probs_l1)} test samples")
    print_flush()

    # Load labels via DataLoader (same approach as evaluate_per_class_thresholds.py,
    # avoids requiring pre-computed segmentation masks on disk)
    def load_labels_via_dataloader(split: str) -> list:
        ds = hydra.utils.instantiate(cfg.dataset, split=split, transforms=transforms_test)
        
        # Use batch_size=1 because samples may have different dimensions
        # (batching fails with variable-sized tensors)
        dl = DataLoader(ds, batch_size=1, shuffle=False,
                        num_workers=args.num_workers, pin_memory=False)
        labels = []
        for batch in tqdm(dl, desc=f"Loading {split} labels"):
            _, label, _ = batch
            labels.append(label.squeeze(0))
        return labels

    if not args.skip_train:
        print_flush("Loading train labels...")
        train_labels_l1 = load_labels_via_dataloader("train")
        train_labels_l0 = remap_to_level0(train_labels_l1, remapping_dict)
        print_flush(f"✓ Train labels remapped: {len(train_labels_l0)} samples")
        
        # Free memory: delete level 1 labels (no longer needed)
        del train_labels_l1
        import gc
        gc.collect()

    print_flush("Loading test labels...")
    test_labels_l1 = load_labels_via_dataloader("test")
    test_labels_l0 = remap_to_level0(test_labels_l1, remapping_dict)
    print_flush(f"✓ Test labels remapped: {len(test_labels_l0)} samples")
    
    # Free memory: delete level 1 labels (no longer needed)
    del test_labels_l1
    import gc
    gc.collect()
    
    print_flush()

    num_level0_classes = len(remapping_dict)
    default_thresholds = {i: 0.5 for i in range(num_level0_classes)}

    # ==================================================================
    # STRATEGY 1: EQUAL WEIGHTS (BASELINE)
    # ==================================================================
    print_flush("="*100)
    print_flush("STRATEGY 1: EQUAL WEIGHTS (BASELINE)")
    print_flush("="*100)

    equal_weights = {
        level0_class: [1.0] * len(level1_classes)
        for level0_class, level1_classes in remapping_dict.items()
    }
    print_flush(f"Equal weights: {equal_weights}")

    if not args.skip_train:
        train_probs_l0_equal = weighted_remap_to_level0(train_probs_l1, remapping_dict, equal_weights)
        train_preds_l0_equal = apply_thresholds(train_probs_l0_equal, default_thresholds)
        train_metrics_equal = evaluate_metrics(train_preds_l0_equal, train_labels_l0, num_level0_classes)
        print_flush(f"Train macro Dice: {train_metrics_equal['macro_dice']:.4f}")

    test_probs_l0_equal = weighted_remap_to_level0(test_probs_l1, remapping_dict, equal_weights)
    test_preds_l0_equal = apply_thresholds(test_probs_l0_equal, default_thresholds)
    test_metrics_equal = evaluate_metrics(test_preds_l0_equal, test_labels_l0, num_level0_classes)
    print_flush(f"Test macro Dice:  {test_metrics_equal['macro_dice']:.4f}")
    print_flush()
    
    # Free memory: delete intermediate tensors
    if not args.skip_train:
        del train_probs_l0_equal, train_preds_l0_equal
    del test_probs_l0_equal, test_preds_l0_equal
    import gc
    gc.collect()

    # ==================================================================
    # STRATEGY 2: THRESHOLD-BASED WEIGHTS (OPTION A)
    # ==================================================================
    print_flush("="*100)
    print_flush("STRATEGY 2: THRESHOLD-BASED WEIGHTS (OPTION A)")
    print_flush("="*100)

    threshold_weights = None
    train_metrics_threshold = None
    test_metrics_threshold = None

    if args.threshold_results_path.exists():
        print_flush(f"Loading threshold results from {args.threshold_results_path}")
        with open(args.threshold_results_path) as f:
            threshold_results = json.load(f)

        optimal_thresholds_l1 = threshold_results['optimal_thresholds']
        print_flush(f"Optimal level 1 thresholds: {optimal_thresholds_l1}")

        threshold_weights = compute_threshold_based_weights(optimal_thresholds_l1, remapping_dict)
        print_flush(f"Threshold-based weights: {threshold_weights}")

        if not args.skip_train:
            train_probs_l0_threshold = weighted_remap_to_level0(train_probs_l1, remapping_dict, threshold_weights)
            train_preds_l0_threshold = apply_thresholds(train_probs_l0_threshold, default_thresholds)
            train_metrics_threshold = evaluate_metrics(train_preds_l0_threshold, train_labels_l0, num_level0_classes)
            print_flush(f"Train macro Dice: {train_metrics_threshold['macro_dice']:.4f}")

        test_probs_l0_threshold = weighted_remap_to_level0(test_probs_l1, remapping_dict, threshold_weights)
        test_preds_l0_threshold = apply_thresholds(test_probs_l0_threshold, default_thresholds)
        test_metrics_threshold = evaluate_metrics(test_preds_l0_threshold, test_labels_l0, num_level0_classes)
        print_flush(f"Test macro Dice:  {test_metrics_threshold['macro_dice']:.4f}")
    else:
        print_flush(f"⚠ Threshold results not found at {args.threshold_results_path}")
        print_flush("  Skipping threshold-based weights evaluation")

    print_flush()

    # ==================================================================
    # STRATEGY 3: OPTIMIZED WEIGHTS (OPTION B)
    # ==================================================================
    print_flush("="*100)
    print_flush("STRATEGY 3: OPTIMIZED WEIGHTS (OPTION B)")
    print_flush("="*100)

    if args.skip_train:
        print_flush("⚠ Cannot optimize weights with --skip_train (need training set)")
        print_flush("  Skipping optimized weights evaluation")
        optimized_weights = None
        train_metrics_optimized = None
        test_metrics_optimized = None
    else:
        optimized_weights, opt_info = optimize_aggregation_weights(
            train_probs_l1,
            train_labels_l0,
            remapping_dict,
            method='Nelder-Mead',
            initial_weights='equal',
            verbose=True
        )

        print_flush(f"\nOptimized weights: {optimized_weights}")

        train_probs_l0_optimized = weighted_remap_to_level0(train_probs_l1, remapping_dict, optimized_weights)
        train_preds_l0_optimized = apply_thresholds(train_probs_l0_optimized, default_thresholds)
        train_metrics_optimized = evaluate_metrics(train_preds_l0_optimized, train_labels_l0, num_level0_classes)
        print_flush(f"Train macro Dice: {train_metrics_optimized['macro_dice']:.4f}")

        test_probs_l0_optimized = weighted_remap_to_level0(test_probs_l1, remapping_dict, optimized_weights)
        test_preds_l0_optimized = apply_thresholds(test_probs_l0_optimized, default_thresholds)
        test_metrics_optimized = evaluate_metrics(test_preds_l0_optimized, test_labels_l0, num_level0_classes)
        print_flush(f"Test macro Dice:  {test_metrics_optimized['macro_dice']:.4f}")

    print_flush()

    # ==================================================================
    # COMPARISON & VISUALIZATION
    # ==================================================================
    strategies_results = {
        'Equal Weights (Baseline)': (
            train_metrics_equal if not args.skip_train else None,
            test_metrics_equal
        ),
        'Threshold-based': (
            train_metrics_threshold,
            test_metrics_threshold
        ) if threshold_weights is not None else None,
        'Optimized': (
            train_metrics_optimized,
            test_metrics_optimized
        ) if optimized_weights is not None else None
    }

    # Print comparison table
    comparison_table = create_comparison_table(strategies_results, remapping_dict)
    print_flush(comparison_table)

    # Save results
    results = {
        'remapping_dict': {k: v for k, v in remapping_dict.items()},
        'strategies': {
            'equal_weights': {
                'weights': equal_weights,
                'train_metrics': train_metrics_equal if not args.skip_train else None,
                'test_metrics': test_metrics_equal
            },
            'threshold_based': {
                'weights': threshold_weights,
                'train_metrics': train_metrics_threshold,
                'test_metrics': test_metrics_threshold
            } if threshold_weights is not None else None,
            'optimized': {
                'weights': optimized_weights,
                'train_metrics': train_metrics_optimized,
                'test_metrics': test_metrics_optimized,
                'optimization_info': opt_info if optimized_weights is not None else None
            } if optimized_weights is not None else None
        }
    }

    results_path = output_dir / "weighted_aggregation_results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print_flush(f"\n✓ Results saved to {results_path}")

    # Generate visualizations
    print_flush("\n" + "="*100)
    print_flush("GENERATING VISUALIZATIONS")
    print_flush("="*100)

    if optimized_weights is not None:
        visualize_weights_comparison(
            equal_weights,
            threshold_weights,
            optimized_weights,
            remapping_dict,
            output_dir / "weights_comparison.png"
        )

    visualize_performance_comparison(
        strategies_results,
        output_dir / "performance_comparison.png"
    )

    print_flush("\n" + "="*100)
    print_flush("DONE!")
    print_flush("="*100)
    print_flush(f"\nResults saved to: {output_dir}")
    print_flush("  - weighted_aggregation_results.json: Complete results")
    print_flush("  - weights_comparison.png: Weight distributions")
    print_flush("  - performance_comparison.png: Performance comparison")


if __name__ == "__main__":
    main()
