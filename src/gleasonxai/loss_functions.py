''' This module implements helper classes for the Cross-Entropy and DICE loss functions, to allow passing in the exact same shapes as with the SoftDiceLoss.'''

import torch
import torch.nn as nn
from monai.losses.dice import DiceLoss


class OneHotCE(nn.CrossEntropyLoss):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def forward(self, img: torch.Tensor, mask: torch.Tensor):

        assert img.dim() == mask.dim(), f"Dimension mismatch: {img.dim()} != {mask.dim()}"

        mask = torch.argmax(mask, dim=1)

        return super().forward(img, mask)


class OneHotDICE(DiceLoss):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def forward(self, input: torch.Tensor, target: torch.Tensor):

        assert input.dim() == target.dim(), f"Dimension mismatch: {input.dim()} != {target.dim()}"

        target = torch.argmax(target, dim=1)

        return super().forward(input, target)
