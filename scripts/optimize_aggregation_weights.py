"""
Weight optimization module for aggregating level 1 probabilities to level 0.

This module provides utilities to:
1. Convert weights between dictionary and flat array formats
2. Compute threshold-based weights (Option A)
3. Optimize weights directly to maximize level 0 Dice score (Option B)
"""

import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from scipy.optimize import minimize
from tqdm import tqdm


def flatten_weights(weights_dict: Dict[int, List[float]], remapping_dict: Dict[int, List[int]]) -> np.ndarray:
    """
    Convert weights dictionary to flat array for optimization.
    Only includes weights that need optimization (multi-child classes).

    Args:
        weights_dict: Dictionary mapping level0_class -> list of weights
        remapping_dict: Dictionary mapping level0_class -> list of level1_classes

    Returns:
        Flat numpy array of weights to optimize

    Example:
        >>> remapping_dict = {0: [0], 1: [1, 2], 2: [3, 4, 5, 6, 7, 8], 3: [9]}
        >>> weights_dict = {0: [1.0], 1: [0.6, 0.4], 2: [0.2, 0.15, 0.25, 0.1, 0.2, 0.1], 3: [1.0]}
        >>> flat = flatten_weights(weights_dict, remapping_dict)
        >>> flat.shape
        (10,)  # 1 weight for class 0 + 2 for class 1 + 6 for class 2 + 1 for class 3
    """
    flat_weights = []

    for level0_class in sorted(remapping_dict.keys()):
        level1_classes = remapping_dict[level0_class]

        # Include ALL weights (single and multi-child classes)
        # Single-child weights act as implicit threshold adjustments
        weights = weights_dict[level0_class]
        flat_weights.extend(weights)

    return np.array(flat_weights)


def unflatten_weights(
    flat_weights: np.ndarray,
    remapping_dict: Dict[int, List[int]]
) -> Dict[int, List[float]]:
    """
    Convert flat array back to weights dictionary structure.

    Weights are kept as unconstrained positive values (weighted sum, not
    weighted average). This preserves the same probability scale as the
    original remap_to_level0() which also summed probabilities, ensuring
    the default threshold of 0.5 remains meaningful.

    Equal weights (all = 1.0) reproduce the original equal-sum baseline.
    The optimizer can deviate to emphasize subtypes that are more reliable.

    Args:
        flat_weights: Flat numpy array of weights
        remapping_dict: Dictionary mapping level0_class -> list of level1_classes

    Returns:
        Dictionary mapping level0_class -> list of positive weights (unconstrained sum)

    Example:
        >>> remapping_dict = {0: [0], 1: [1, 2], 2: [3, 4, 5, 6, 7, 8], 3: [9]}
        >>> flat = np.array([1.5, 1.2, 0.8, 0.9, 1.1, 1.0, 0.7, 1.3, 0.6, 0.9])
        >>> weights_dict = unflatten_weights(flat, remapping_dict)
        >>> weights_dict[0]  # [1.5] - optimized weight for single-child class
        >>> weights_dict[1]  # [1.2, 0.8] - weights for multi-child class
    """
    weights_dict = {}
    idx = 0

    for level0_class in sorted(remapping_dict.keys()):
        level1_classes = remapping_dict[level0_class]

        # Extract weights for ALL classes (single and multi-child)
        # Single-child weights act as implicit threshold adjustments
        n_children = len(level1_classes)
        raw_weights = flat_weights[idx:idx + n_children]

        # Ensure positive weights via absolute value
        weights_dict[level0_class] = np.abs(raw_weights).tolist()

        idx += n_children

    return weights_dict


def compute_threshold_based_weights(
    optimal_thresholds_l1: Dict[int, float],
    remapping_dict: Dict[int, List[int]]
) -> Dict[int, List[float]]:
    """
    Convert optimized level 1 thresholds to aggregation weights (Option A).

    Strategy: Use inverse threshold weighting
    Intuition: Classes with lower thresholds (more sensitive) should have higher weight
    in the aggregation, as they indicate stronger subtype presence.

    Formula: w_i = (1 - threshold_i) / Σ(1 - threshold_j)

    Args:
        optimal_thresholds_l1: Per-class thresholds from level 1 optimization
                              e.g., {0: 0.5, 1: 0.42, 2: 0.58, ...}
        remapping_dict: Mapping from level 0 to level 1 classes
                       e.g., {0: [0], 1: [1, 2], 2: [3, 4, 5, 6, 7, 8], 3: [9]}

    Returns:
        Weights dictionary for aggregation
        e.g., {0: [1.0], 1: [0.58, 0.42], 2: [...], 3: [1.0]}

    Example:
        If threshold_individual_glands = 0.42 and threshold_compressed_glands = 0.58:
        - w_individual = (1 - 0.42) / (1 - 0.42 + 1 - 0.58) = 0.58 / 1.00 = 0.58
        - w_compressed = (1 - 0.58) / (1 - 0.42 + 1 - 0.58) = 0.42 / 1.00 = 0.42

        Lower threshold → higher weight (more sensitive → more important)
    """
    weights_dict = {}

    for level0_class, level1_classes in remapping_dict.items():
        if len(level1_classes) == 1:
            # Single child: weight = 1.0
            weights_dict[level0_class] = [1.0]
        else:
            # Multiple children: inverse threshold weighting
            # Convert threshold keys to integers if they're strings
            thresholds_list = [
                optimal_thresholds_l1[str(c1)] if str(c1) in optimal_thresholds_l1
                else optimal_thresholds_l1[c1]
                for c1 in level1_classes
            ]

            # Compute inverse thresholds (1 - t) as raw sensitivity scores
            inverse_thresholds = [1.0 - t for t in thresholds_list]

            # Normalize so the mean weight = 1.0, preserving the same probability
            # scale as the original equal-sum baseline (all weights = 1.0)
            n = len(inverse_thresholds)
            mean_inv = sum(inverse_thresholds) / n
            if mean_inv < 1e-10:
                weights = [1.0] * n
            else:
                weights = [inv_t / mean_inv for inv_t in inverse_thresholds]

            weights_dict[level0_class] = weights

    return weights_dict


def optimize_aggregation_weights(
    probs_l1_train: list,
    labels_l0_train: list,
    remapping_dict: Dict[int, list],
    method: str = "Nelder-Mead",
    initial_weights: str = "equal",
    verbose: bool = True
) -> Tuple[Dict[int, List[float]], Dict[str, float]]:
    """
    Optimize weights to maximize macro Dice score on level 0 (Option B).

    This function finds the optimal weights for aggregating level 1 probabilities
    into level 0 predictions by maximizing the macro Dice score on the training set.

    Optimization details:
    - Objective: Maximize level 0 macro Dice on training set
    - Method: Nelder-Mead (derivative-free simplex) — required because Dice is
      non-differentiable (hard threshold). Gradient-based methods (L-BFGS-B) see
      zero gradient everywhere via finite differences and exit in 0 iterations.
    - Parameters: 10 weights (1 for class 0, 2 for class 1, 3 for class 2, 4 for class 3)
    - Initial point: all weights = 1.0 (reproduces original equal-sum baseline)

    Args:
        probs_l1_train: Level 1 probabilities (training set), list of tensors (11, H, W)
        labels_l0_train: Level 0 labels (training set), list of tensors (4, H, W)
        remapping_dict: Mapping from level 0 to level 1 classes
        method: Optimization method ('L-BFGS-B', 'SLSQP', 'trust-constr')
        initial_weights: 'equal', 'random', or dict of initial weights
        verbose: Print optimization progress

    Returns:
        Tuple of:
        - optimal_weights_dict: Optimized weights {level0_class: [weights]}
        - optimization_info: Dictionary with optimization metadata

    Example:
        >>> optimal_weights, info = optimize_aggregation_weights(
        ...     train_probs_l1, train_labels_l0, remapping_dict, verbose=True
        ... )
        >>> print(f"Optimal Dice: {info['final_dice']:.4f}")
        >>> print(f"Optimal weights: {optimal_weights}")
    """
    # Import here to avoid circular dependency
    sys.path.insert(0, str(Path(__file__).parent))
    from evaluate_level0_metrics import weighted_remap_to_level0, apply_thresholds, evaluate_metrics

    # Initialize weights
    # Use 1.0 per child as the "equal sum" baseline (reproduces original remap_to_level0)
    if initial_weights == "equal":
        weights_dict_init = {
            level0_class: [1.0] * len(level1_classes)
            for level0_class, level1_classes in remapping_dict.items()
        }
    elif initial_weights == "random":
        np.random.seed(42)
        weights_dict_init = {}
        for level0_class, level1_classes in remapping_dict.items():
            # Random positive weights centered around 1.0
            weights_dict_init[level0_class] = (0.5 + np.random.rand(len(level1_classes))).tolist()
    else:
        weights_dict_init = initial_weights

    # Flatten for optimization
    x0 = flatten_weights(weights_dict_init, remapping_dict)
    n_params = len(x0)

    if verbose:
        print(f"\nOptimizing {n_params} weight parameters...")
        print(f"Initial weights: {weights_dict_init}")
        print(f"Optimization method: {method}")

    # Define objective function
    iteration_count = [0]  # Use list to allow modification in nested function

    def objective(x):
        weights_dict = unflatten_weights(x, remapping_dict)

        # Weighted remapping
        probs_l0 = weighted_remap_to_level0(probs_l1_train, remapping_dict, weights_dict)

        # Binarize with default threshold
        default_thresholds = {i: 0.5 for i in range(len(remapping_dict))}
        preds_l0 = apply_thresholds(probs_l0, default_thresholds)

        # Evaluate
        metrics = evaluate_metrics(preds_l0, labels_l0_train, num_classes=len(remapping_dict))

        # Return negative macro Dice (we minimize)
        loss = -metrics['macro_dice']

        iteration_count[0] += 1
        if verbose and iteration_count[0] % 10 == 0:
            print(f"  Iteration {iteration_count[0]}: Dice={-loss:.4f}, weights={weights_dict}")

        return loss

    if verbose:
        print(f"\nRunning {method} optimization...")

    # Nelder-Mead is derivative-free (no bounds parameter supported).
    # Negative weights are clipped to positive inside unflatten_weights via abs().
    # L-BFGS-B is kept as an option but will fail on this non-smooth objective.
    if method == "Nelder-Mead":
        nm_options = {
            'maxiter': 1500,
            'maxfev': 4000,
            'xatol': 1e-4,
            'fatol': 1e-4,
            'disp': verbose,
            'adaptive': True,   # Adaptive simplex parameters for higher-dimensional problems
        }
        result = minimize(objective, x0, method="Nelder-Mead", options=nm_options)
    else:
        bounds = [(0.01, 5.0)] * n_params
        result = minimize(
            objective, x0, method=method, bounds=bounds,
            options={'maxiter': 200, 'disp': verbose}
        )

    if not result.success and verbose:
        print(f"Warning: Optimization did not converge. Message: {result.message}")

    # Extract optimal weights
    optimal_weights_dict = unflatten_weights(result.x, remapping_dict)

    # Compute final metrics
    if verbose:
        print(f"\nComputing final metrics with optimal weights...")

    probs_l0_optimal = weighted_remap_to_level0(probs_l1_train, remapping_dict, optimal_weights_dict)
    default_thresholds = {i: 0.5 for i in range(len(remapping_dict))}
    preds_l0_optimal = apply_thresholds(probs_l0_optimal, default_thresholds)
    final_metrics = evaluate_metrics(preds_l0_optimal, labels_l0_train, num_classes=len(remapping_dict))

    optimization_info = {
        'method': method,
        'success': result.success,
        'n_iterations': result.nit if hasattr(result, 'nit') else result.nfev,
        'final_dice': final_metrics['macro_dice'],
        'final_loss': result.fun,
        'message': result.message
    }

    if verbose:
        print(f"\nOptimization complete!")
        print(f"  Success: {result.success}")
        print(f"  Iterations: {optimization_info['n_iterations']}")
        print(f"  Final macro Dice: {final_metrics['macro_dice']:.4f}")
        print(f"  Optimal weights:")
        for level0_class, weights in optimal_weights_dict.items():
            print(f"    Level 0 class {level0_class}: {[f'{w:.4f}' for w in weights]}")

    return optimal_weights_dict, optimization_info


if __name__ == "__main__":
    # Simple test of weight utilities
    print("Testing weight utilities...")

    # Test data
    remapping_dict = {0: [0], 1: [1, 2], 2: [3, 4, 5, 6, 7, 8], 3: [9]}
    weights_dict = {
        0: [1.0],
        1: [0.6, 0.4],
        2: [0.2, 0.15, 0.25, 0.1, 0.2, 0.1],
        3: [1.0]
    }

    print(f"\nOriginal weights: {weights_dict}")

    # Test flatten
    flat = flatten_weights(weights_dict, remapping_dict)
    print(f"Flattened: {flat} (shape: {flat.shape})")

    # Test unflatten
    reconstructed = unflatten_weights(flat, remapping_dict)
    print(f"Reconstructed: {reconstructed}")

    # Verify normalization
    for level0_class, weights in reconstructed.items():
        weight_sum = sum(weights)
        print(f"  Level 0 class {level0_class}: sum(weights) = {weight_sum:.6f}")

    print("\n✓ Weight utilities working correctly!")
