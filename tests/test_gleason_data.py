"""tests/test_gleasonx.py
================================
Comprehensive PyTest suite for `gleasonxai.gleason_data` helpers and dataset
classes.

All tests are designed to run quickly in CI.  Lightweight unit–tests that do
not touch the real dataset are grouped first; heavier integration tests which
rely on the dataset present at
`Path(os.environ["DATASET_LOCATION"])/"GleasonXAI"` follow and are skipped if
the data folder is absent.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image

# ---------------------------------------------------------------------------
# Imports from the data module under test
# ---------------------------------------------------------------------------
from gleasonxai.gleason_data import (
    GleasonX,
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


@pytest.mark.parametrize("file_format", [".png", "jpg"])
def test_reformat_dataset_to_flat_structure(tmp_path: Path, sample_image: Image.Image, monkeypatch, file_format):
    """Images should be copied/converted into the new flat folder."""
    # Arrange – fake nested folder with a single file
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

    # Act
    reformat_dataset_to_flat_structure(src_dir, dst_dir, ".png")

    # Assert
    assert (dst_dir / "sample.png").exists()


# ---------------------------------------------------------------------------
# Integration tests – require the *real* dataset on disk
# ---------------------------------------------------------------------------

DATA_ROOT = Path(os.environ.get("DATASET_LOCATION", "")) / "GleasonXAI"

skip_data = pytest.mark.skipif(
    not DATA_ROOT.exists(),
    reason="GleasonXAI dataset not available in CI env",
)


def create_gleasxonx_dataset(split: str):
    return GleasonX(
        DATA_ROOT,
        split=split,
        scaling="MicronsCalibrated",
        create_seg_masks=True,
        drawing_order="grade_frame_order",
        explanation_file="final_filtered_explanations_df.csv",
        data_split=[0.7, 0.15, 0.15],
        tissue_mask_kwargs={"open": False, "close": False, "flood": False},
    )


@pytest.fixture(scope="module")
def gleasonx_dataset():
    return (create_gleasxonx_dataset("train"), create_gleasxonx_dataset("val"), create_gleasxonx_dataset("test"))


def test_dataset_exists():
    """The dataset root must exist."""
    assert DATA_ROOT.exists(), f"GleasonXAI dataset not found at {DATA_ROOT}"
    assert DATA_ROOT.is_dir(), f"GleasonXAI dataset root is not a directory: {DATA_ROOT}"


@skip_data
def test_dataset_basic_properties(gleasonx_dataset):
    """Dataset splits should be non‑empty and mutually exclusive."""
    # ds_train = GleasonX(DATA_ROOT, split="train", scaling="MicronsCalibrated", create_seg_masks=True, drawing_order="grade_frame_order", explanation_file="final_filtered_explanations_df.csv", data_split=[0.7, 0.15, 0.15], tissue_mask_kwargs={"open":False, "close":False, "flood":False})
    # ds_val = GleasonX(DATA_ROOT, split="val", scaling="MicronsCalibrated", create_seg_masks=True, drawing_order="grade_frame_order", explanation_file="final_filtered_explanations_df.csv", data_split=[0.7, 0.15, 0.15], tissue_mask_kwargs={"open":False, "close":False, "flood":False})
    # ds_test = GleasonX(DATA_ROOT, split="test", scaling="MicronsCalibrated", create_seg_masks=True, drawing_order="grade_frame_order", explanation_file="final_filtered_explanations_df.csv", data_split=[0.7, 0.15, 0.15], tissue_mask_kwargs={"open":False, "close":False, "flood":False})

    ds_train, ds_val, ds_test = gleasonx_dataset

    # Basic lengths
    assert isinstance(ds_train, GleasonX)
    assert len(ds_train) > 0
    assert len(ds_val) > 0
    assert len(ds_test) > 0

    # Ensure the slide identifiers do not overlap across splits
    ids_train = set(ds_train.used_slides)
    ids_val = set(ds_val.used_slides)
    ids_test = set(ds_test.used_slides)
    assert ids_train.isdisjoint(ids_val)
    assert ids_train.isdisjoint(ids_test)
    assert ids_val.isdisjoint(ids_test)


@skip_data
@pytest.mark.parametrize("idx", [0, 1])
def test_dataset_item_shapes(gleasonx_dataset, idx):
    """`GleasonX.__getitem__` should return well‑formed tensors."""
    ds = gleasonx_dataset[1] # val split
    img, mask, bg = ds[idx]

    C, H, W = img.shape
    assert C == 3
    assert mask.shape == (ds.num_classes, H, W)
    assert bg.shape == (H, W)

    # Mask channels must be 0/1 – allow soft‑labels when multiple annotators
    assert torch.le(mask, 1).all() and torch.ge(mask, 0).all()


@skip_data
def test_get_raw_image_and_slide_df(gleasonx_dataset):
    """Utility accessors should fetch a PIL image and DataFrame."""
    ds = gleasonx_dataset[1]

    pil_img = ds.get_raw_image(0)
    assert isinstance(pil_img, Image.Image)

    df = ds.get_slide_df(0)
    assert isinstance(df, pd.DataFrame)
    # Must contain at minimum these columns
    assert {"coords", "explanations"}.issubset(df.columns)
