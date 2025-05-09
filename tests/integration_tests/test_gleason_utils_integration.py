from __future__ import annotations

import os
from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# Imports from the data module under test
# ---------------------------------------------------------------------------
from gleasonxai.gleason_data import (
    GleasonX,
)
from gleasonxai.gleason_utils import create_segmentation_masks
import numpy as np

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


# ---------------------------------------------------------------------------
# SEGMENTATION‑MASK CREATION TESTS
# ---------------------------------------------------------------------------


@skip_data
@pytest.mark.parametrize("shorter_edge", [None, 512])
def test_create_segmentation_masks_shapes(shorter_edge):
    ds = create_gleasxonx_dataset("train")
    _, annotator_imgs, bg_mask = create_segmentation_masks(
        ds,
        img_index=0,
        shorter_edge_length=shorter_edge,
        img=None,
        background_mask=None,
        tissue_mask_kwargs={"open": False, "close": False, "flood": False},
        drawing_order="grade_frame_order",
    )

    # Return contract
    assert isinstance(annotator_imgs, dict) and len(annotator_imgs) > 0
    assert isinstance(bg_mask, np.ndarray) and bg_mask.ndim == 2

    for name, arr in annotator_imgs.items():
        assert isinstance(arr, np.ndarray) and arr.ndim == 2, f"Annotator {name} returns non‑2D mask"
        assert arr.shape == bg_mask.shape, "Shape mismatch between label & background masks"
        assert arr.dtype.kind in {"i", "u"}
        assert arr.min() >= 0, "Negative label IDs found"
