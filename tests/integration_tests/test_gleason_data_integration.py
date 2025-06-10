from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import pytest
import torch
from PIL import Image


# ---------------------------------------------------------------------------
# Imports from the data module under test
# ---------------------------------------------------------------------------
from gleasonxai.gleason_data import (
    GleasonX,
)

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

    all_len = len(ds_train) + len(ds_val) + len(ds_test)
    assert pytest.approx(0.7, abs=0.1) == len(ds_train) / all_len
    assert pytest.approx(0.15, abs=0.1) == len(ds_val) / all_len
    assert pytest.approx(0.15, abs=0.1) == len(ds_test) / all_len

    ds_train_2 = create_gleasxonx_dataset("train")
    # Make sure the same images land in train
    assert set(ds_train.used_slides) == set(ds_train_2.used_slides)


@skip_data
@pytest.mark.parametrize("idx", [0, 1])
def test_dataset_item_shapes(gleasonx_dataset, idx):
    """`GleasonX.__getitem__` should return well‑formed tensors."""
    ds = gleasonx_dataset[1]  # val split
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


@skip_data
def test_label_level_shapes_and_values_df(gleasonx_dataset):
    """Utility accessors should fetch a PIL image and DataFrame."""
    ds_0 = GleasonX(
        DATA_ROOT,
        split="test",
        label_level=0,
        scaling="MicronsCalibrated",
        create_seg_masks=True,
        drawing_order="grade_frame_order",
        explanation_file="final_filtered_explanations_df.csv",
        data_split=[0.7, 0.15, 0.15],
        tissue_mask_kwargs={"open": False, "close": False, "flood": False},
    )

    ds_1 = GleasonX(
        DATA_ROOT,
        split="test",
        label_level=1,
        scaling="MicronsCalibrated",
        create_seg_masks=True,
        drawing_order="grade_frame_order",
        explanation_file="final_filtered_explanations_df.csv",
        data_split=[0.7, 0.15, 0.15],
        tissue_mask_kwargs={"open": False, "close": False, "flood": False},
    )

    ds_2 = GleasonX(
        DATA_ROOT,
        split="test",
        label_level=2,
        scaling="MicronsCalibrated",
        create_seg_masks=True,
        drawing_order="grade_frame_order",
        explanation_file="final_filtered_explanations_df.csv",
        data_split=[0.7, 0.15, 0.15],
        tissue_mask_kwargs={"open": False, "close": False, "flood": False},
    )

    assert ds_0.num_classes == 4
    assert ds_1.num_classes == 10
    assert ds_2.num_classes == 30

    _, mask_0, _ = ds_0[0]
    _, mask_1, _ = ds_1[0]
    _, mask_2, _ = ds_2[0]

    _, H, W = mask_0.shape

    assert mask_0.shape == (ds_0.num_classes, H, W)
    assert mask_1.shape == (ds_1.num_classes, H, W)
    assert mask_2.shape == (ds_2.num_classes, H, W)

    assert (mask_0.sum(dim=0) == 1).all()
    assert (mask_1.sum(dim=0) == 1).all()
    assert (mask_2.sum(dim=0) == 1).all()

    assert torch.le(mask_0, 1).all() and torch.ge(mask_0, 0).all()
    assert torch.le(mask_1, 1).all() and torch.ge(mask_1, 0).all()
    assert torch.le(mask_2, 1).all() and torch.ge(mask_2, 0).all()
