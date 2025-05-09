import pytest
import torch
import torch.nn.functional as F

from gleasonxai.model_utils import (
    dice_score_soft,
    dice_score_hard,
    dice_loss_soft,
    SoftDiceLoss,
    MultiLabelLoss,
    SoftDICEMetric,
    L1CalibrationMetric,
)


# ---------------------------------------------------------
# dice_score_* utility functions
# ---------------------------------------------------------


def test_dice_score_soft_perfect() -> None:
    """Soft DICE should be exactly 1 for identical prediction & target."""
    logits = torch.tensor([[[10.0, 0.0]]]).permute(2, 0, 1).unsqueeze(0)  # shape (C, H, W)
    probs = F.softmax(logits, dim=1)
    target = torch.tensor([[0]])  # class index 0
    target_oh = F.one_hot(target, num_classes=2).permute(2, 0, 1)

    score = dice_score_soft(probs, target_oh, average="micro")
    assert score.item() == pytest.approx(1.0, abs=1e-3)


def test_dice_score_soft_zero() -> None:
    """Soft DICE should be near 0 when prediction and target do not overlap."""
    logits = torch.tensor([[[0.0, 10.0]]]).permute(2, 0, 1).unsqueeze(0)  # shape (C, H, W)
    probs = F.softmax(logits, dim=1)
    target = torch.tensor([[0]])  # class index 0
    target_oh = F.one_hot(target, num_classes=2).permute(2, 0, 1)

    score = dice_score_soft(probs, target_oh, average="micro")
    assert score.item() == pytest.approx(0.0, abs=1e-3)


def test_dice_score_hard() -> None:
    logits = torch.tensor([[[10.0, 0.0]]]).permute(2, 0, 1).unsqueeze(0)  # shape (C, H, W)
    probs = F.softmax(logits, dim=1)
    target = torch.tensor([[0]])  # class index 0
    target_oh = F.one_hot(target, num_classes=2).permute(2, 0, 1)

    score = dice_score_hard(probs, target_oh, average="micro")
    assert score.item() == pytest.approx(1.0, abs=1e-3)


def test_soft_dice_loss_module_matches_function() -> None:
    logits = torch.randn(4, 3)  # random mini‑batch
    target_idx = torch.randint(0, 3, (4,))
    target_oh = F.one_hot(target_idx, 3)

    loss_fn = SoftDiceLoss(average="micro")
    loss_module_val = loss_fn(logits, target_oh)
    loss_function_val = dice_loss_soft(logits, target_oh, average="micro")
    assert loss_module_val.item() == pytest.approx(loss_function_val.item(), rel=1e-6)


# ---------------------------------------------------------
# MultiLabelLoss
# ---------------------------------------------------------


def test_multilabel_loss_matches_bce() -> None:
    torch.manual_seed(0)
    logits = torch.randn(5, 4)  # 5 samples, 4 labels
    soft_targets = torch.rand(5, 4)
    ml_loss_soft = MultiLabelLoss(hard_targets=False)
    expected_soft = F.binary_cross_entropy_with_logits(logits, soft_targets)
    assert ml_loss_soft(logits, soft_targets).item() == pytest.approx(expected_soft.item(), rel=1e-6)

    # hard mask should threshold targets to 0/1.
    # If one annotator voted for a label consider it set.
    hard_targets = (soft_targets > 0).float()
    ml_loss_hard = MultiLabelLoss(hard_targets=True)
    expected_hard = F.binary_cross_entropy_with_logits(logits, hard_targets)
    assert ml_loss_hard(logits, soft_targets).item() == pytest.approx(expected_hard.item(), rel=1e-6)


# ---------------------------------------------------------
# SoftDICEMetric
# ---------------------------------------------------------


def test_softdice_metric_accumulates_batches() -> None:
    metric = SoftDICEMetric(average="micro")

    # Batch 1 – perfect prediction
    logits1 = torch.zeros(1, 2, 2, 2)  # shape (B=1, C=2, H=2, W=2)
    logits1[0, 0, 0, 0] = 10.0
    logits1[0, 0, 1, 1] = 10.0

    logits1[0, 1, 1, 0] = 10.0
    logits1[0, 1, 0, 1] = 10.0

    probs1 = F.softmax(logits1, dim=1)

    target1_idx = torch.tensor([[0, 1], [1, 0]]).unsqueeze(0)  # Add batch dimension
    target1_oh = F.one_hot(target1_idx, 2).permute(0, 3, 1, 2)  # shape (B=1, C=2, H=2, W=2)
    metric.update(probs1, target1_oh)

    assert metric.compute().item() == pytest.approx(1.0, abs=0.001)

    metric.reset()

    # Batch 2 – completely wrong prediction
    logits2 = torch.zeros(1, 2, 2, 2)  # shape (B=1, C=2, H=2, W=2)
    logits2[0, 1, 0, 0] = 10.0
    logits2[0, 1, 1, 1] = 10.0

    logits2[0, 0, 1, 0] = 10.0
    logits2[0, 0, 0, 1] = 10.0

    probs2 = F.softmax(logits2, dim=1)

    target2_idx = torch.tensor([[0, 1], [1, 0]]).unsqueeze(0)  # Add batch dimension
    target2_oh = F.one_hot(target2_idx, 2).permute(0, 3, 1, 2)  # shape (B=1, C=2, H=2, W=2)
    metric.update(probs2, target2_oh)

    assert metric.compute().item() == pytest.approx(0.0, abs=0.001)
    metric.reset()

    metric.update(probs1, target1_oh)
    metric.update(probs2, target2_oh)

    # Expected average DICE: (1 + 0) / 2 = 0.5
    assert metric.compute().item() == pytest.approx(0.5, abs=1e-6)

    # Soft target test
    target_3 = torch.zeros(1, 2, 2, 2)  # shape (B=1, C=2, H=2, W=2)
    target_3[0, 1, 0, 0] = 3.0
    target_3[0, 1, 1, 1] = 3.0

    target_3[0, 0, 1, 0] = 0.0
    target_3[0, 0, 0, 1] = 0.0

    target_3 = F.softmax(target_3, dim=1)

    metric.reset()
    metric.update(probs1, target_3)

    c = metric.compute().item()
    assert 0.0 < c < 1.0, f"Soft DICE should be between 0 and 1, got {c}"


# ---------------------------------------------------------
# L1CalibrationMetric
# ---------------------------------------------------------


def test_l1_calibration_metric() -> None:
    metric = L1CalibrationMetric()

    preds = torch.tensor([[0.7, 0.3], [0.2, 0.8]])
    target = torch.tensor([[1.0, 0.0], [0.0, 1.0]])

    metric.update(preds, target)
    # Manual computation:
    # sample 1: (|0.7‑1| + |0.3‑0|)/2 = 0.3
    # sample 2: (|0.2‑0| + |0.8‑1|)/2 = 0.2
    expected = (0.3 + 0.2) / 2
    assert metric.compute().item() == pytest.approx(expected, abs=1e-6)
