import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import torch

# -----------------------------------------------------------------------------
# Ensure the project root is on the PYTHONPATH so that ``lightning_modul`` can
# be imported when the tests are run from any location (e.g. via ``pytest`` at
# the project root).
# -----------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# The module name is *lightning_modul.py* (without the trailing "e")
from gleasonxai.lightning_modul import LitClassifier, LitSegmenter

# -----------------------------------------------------------------------------
# Dummy models
# -----------------------------------------------------------------------------


class DummySegModel(torch.nn.Module):
    """Very small convolutional net that keeps the spatial resolution and
    changes only the channel dimension. Useful for lightning module smoke
    tests where we do *not* want to spend time or memory on a large network.
    """

    def __init__(self, num_classes: int = 3):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, 3, H, W) -> (B, C, H, W)
        return self.conv(x)


class DummyClsModel(torch.nn.Module):
    """Tiny classifier: a *1×1* conv followed by global average pooling."""

    def __init__(self, num_classes: int = 3):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, num_classes, kernel_size=1)
        self.pool = torch.nn.AdaptiveAvgPool2d(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, 3, H, W) -> (B, C)
        x = self.conv(x)
        x = self.pool(x)
        return x.view(x.size(0), -1)


# -----------------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------------


@pytest.fixture(scope="module")
def device():
    """CPU for the test‑suite; CUDA is optional but not required."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def seg_model(device):
    model = DummySegModel(num_classes=3).to(device)
    lit = LitSegmenter(model, num_classes=3, metrics_to_track=["loss"]).to(device)
    return lit


@pytest.fixture
def cls_model(device):
    model = DummyClsModel(num_classes=3).to(device)
    lit = LitClassifier(model, num_classes=3, metrics_to_track=["loss"]).to(device)
    return lit


@pytest.fixture
def seg_batch(device):
    """Batch for segmentation: (x, y, ignore_mask)."""
    B, C, H, W = 2, 3, 8, 8
    x = torch.randn(B, 3, H, W, device=device)
    y = torch.nn.functional.softmax(torch.randn(B, C, H, W, device=device), dim=1)
    ignore = torch.zeros(B, H, W, dtype=torch.bool, device=device)
    return x, y, ignore


@pytest.fixture
def seg_model_sw(device):
    """Segmentation module with *sliding‑window* inferer enabled."""
    model = DummySegModel(num_classes=3).to(device)
    return LitSegmenter(
        model,
        num_classes=3,
        metrics_to_track=["loss"],
        sliding_window_in_test=True,
    ).to(device)


@pytest.fixture
def cls_batch(device):
    """Batch for classification: (x, y, dummy_ignore)."""
    B, C, H, W = 2, 3, 8, 8
    x = torch.randn(B, 3, H, W, device=device)
    y = torch.nn.functional.softmax(torch.randn(B, C, device=device), dim=1)
    dummy_ignore = torch.zeros(B, H, W, dtype=torch.bool, device=device)
    return x, y, dummy_ignore


# -----------------------------------------------------------------------------
# Tests for the segmentation LightningModule
# -----------------------------------------------------------------------------


def test_forward_shapes_seg(seg_model, seg_batch):
    x, *_ = seg_batch
    out = seg_model(x)
    B, _, H, W = x.shape
    assert out.shape == (B, seg_model.num_classes, H, W)


def test_evaluate_seg(seg_model, seg_batch):
    losses, logits, labels, out_org = seg_model.evaluate(seg_batch)

    # 1) returned objects have expected types / keys
    assert isinstance(losses, dict) and "loss" in losses

    # 2) shapes agree (after flattening the spatial dims)
    assert logits.shape == labels.shape
    assert logits.shape[1] == seg_model.num_classes

    # 3) original (un‑flattened) logits keep spatial dims
    x, *_ = seg_batch
    _, _, H, W = x.shape
    assert out_org.shape[2:] == (H, W)


def test_configure_optimizers_seg(seg_model):
    optim_conf = seg_model.configure_optimizers()

    # ``configure_optimizers`` may return an ``Optimizer`` *or* a tuple that
    # contains one. Handle both to keep the assertion simple.
    if isinstance(optim_conf, tuple):
        optim = optim_conf[0][0]  # (optimizers, schedulers)
    else:
        optim = optim_conf

    assert hasattr(optim, "state_dict")  # basic sanity check


# -----------------------------------------------------------------------------
# LightningModule smoke‑tests (classification)
# -----------------------------------------------------------------------------


def test_forward_shapes_cls(cls_model, cls_batch):
    x, *_ = cls_batch
    out = cls_model(x)
    assert out.shape == (x.size(0), cls_model.num_classes)


def test_evaluate_cls(cls_model, cls_batch):
    losses, logits, labels = cls_model.evaluate(cls_batch)
    assert isinstance(losses, dict) and "loss" in losses
    assert logits.shape == labels.shape == (cls_batch[0].size(0), cls_model.num_classes)


# -----------------------------------------------------------------------------
# Ignore‑mask behaviour (segmentation)
# -----------------------------------------------------------------------------


def test_ignore_mask(seg_model, device):
    B, C, H, W = 1, 3, 4, 4
    x = torch.randn(B, 3, H, W, device=device)
    y_idx = torch.randint(C, (B, H, W), device=device)
    y = torch.nn.functional.one_hot(y_idx, C).permute(0, 3, 1, 2).float()

    ignore = torch.zeros(B, H, W, dtype=torch.bool, device=device)
    ignore[:, 0:2, 0:2] = True  # ignore upper‑left quarter of each image
    n_valid = (~ignore).sum().item()

    losses, logits, labels, _ = seg_model.evaluate((x, y, ignore), loss=False)

    assert logits.size(0) == labels.size(0) == n_valid, "Logit/label count must equal non‑ignored pixels"
    assert logits.size(1) == C, "Channel dimension should stay the same"


# -----------------------------------------------------------------------------
# Logging behaviour – ensure ``training_step`` propagates to ``self.log``
# -----------------------------------------------------------------------------


def test_logging_train_step(seg_model, seg_batch):
    # Wrap the real ``log`` so that we spy on the calls while preserving behaviour
    seg_model.log = MagicMock(wraps=seg_model.log)
    seg_model.training_step(seg_batch, 0)

    logged_names = [c.args[0] for c in seg_model.log.call_args_list]
    assert any(name.startswith("train_loss") for name in logged_names), "Training loss wasn’t logged"


# -----------------------------------------------------------------------------
# Sliding‑window inference correctness
# -----------------------------------------------------------------------------


def test_sliding_window_inference(seg_model_sw, device):
    """The *SlidingWindowInferer* must reproduce the *direct* model output."""
    B, H, W = 1, 1024, 1024  # Larger than the 512×512 ROI used in the inferer
    x = torch.randn(B, 3, H, W, device=device)

    # Direct (whole‑image) prediction
    direct_out = seg_model_sw.forward(x)

    # Sliding‑window prediction via MONAI inferer
    sw_out = seg_model_sw.sw_inferer(x, seg_model_sw)

    assert sw_out.shape == direct_out.shape
    # Numerical equivalence (allow tiny FP tolerance)
    assert torch.allclose(sw_out, direct_out, atol=1e-5, rtol=1e-3)
