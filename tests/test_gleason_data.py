from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

# ---------------------------------------------------------------------------
# Imports from the data module under test
# ---------------------------------------------------------------------------
from gleasonxai.gleason_data import (
    get_class_colormaps,
    prepare_torch_inputs,
    reformat_dataset_to_flat_structure,
)

# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def rng() -> np.random.Generator:  # pragma: no cover
    """Shared PRNG to make images deterministic."""
    return np.random.default_rng(seed=42)


@pytest.fixture()
def sample_image(rng: np.random.Generator) -> Image.Image:
    """Return a deterministic 32×32 RGB PIL image."""
    arr = rng.integers(0, 255, size=(32, 32, 3), dtype=np.uint8)
    return Image.fromarray(arr)


# ---------------------------------------------------------------------------
# Unit‑tests for *stand‑alone* helper functions (no dataset required)
# ---------------------------------------------------------------------------


def test_prepare_torch_inputs_shapes(sample_image):
    """Verify tensor and label channel shapes returned by `prepare_torch_inputs`."""
    seg_mask = np.zeros((32, 32), dtype=np.int64)  # background only
    img_t, label = prepare_torch_inputs(sample_image, [seg_mask], num_classes=1)

    assert img_t.shape == (3, 32, 32)
    assert label.shape == (1, 32, 32)
    assert torch.all((label == 0) | (label == 1))  # label must be 0/1 only


def test_get_class_colormaps_counts():
    """The colormap must contain 1 background + class counts per grade."""
    cmap = get_class_colormaps({"3": 2, "4": 3, "5": 4})
    expected = 1 + 2 + 3 + 4
    assert cmap.N == expected


@pytest.mark.parametrize("file_format", [".png"])
def test_reformat_dataset_to_flat_structure(tmp_path: Path, sample_image: Image.Image, monkeypatch, file_format):
    """Images should be copied/converted into the new flat folder."""
    src_dir = tmp_path / "src"
    nested_dir = src_dir / "nested"
    dst_dir = tmp_path / "dst"
    nested_dir.mkdir(parents=True)

    orig_name = f"sample{file_format}"
    img_path = nested_dir / orig_name
    sample_image.save(img_path)

    # Monkey‑patch load_tmas so the function under test finds our file only
    def _fake_load_tmas(path: Path):
        return {"sample": Path("nested") / orig_name}

    monkeypatch.setitem(reformat_dataset_to_flat_structure.__globals__, "load_tmas", _fake_load_tmas)

    reformat_dataset_to_flat_structure(src_dir, dst_dir, ".png")

    assert (dst_dir / "sample.png").exists()

