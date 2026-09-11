"""Detection transforms that keep pixels and boxes in step.

The original pipeline applied torchvision image transforms as `img = transforms(img)`, so a
random horizontal flip moved the pixels while the annotation stayed put. Everything here takes
and returns `(image, target)` as a pair, so a geometric change cannot reach one without the other.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from PIL import Image
from torch import Tensor
from torchvision.transforms import functional as TF


def to_float_tensor(image: Image.Image | Tensor) -> Tensor:
    """Return `image` as a float32 CHW tensor scaled to [0, 1], never aliasing the input.

    An integer tensor is divided by the largest value its dtype can hold, so a uint8 tile and the
    same tile widened to int16 land on the same scale. A float tensor is taken to be scaled
    already and is rejected otherwise: letting 0-255 floats through unchanged is the silent
    failure this function exists to prevent.
    """
    if isinstance(image, Image.Image):
        image = TF.pil_to_tensor(image)
    if not isinstance(image, Tensor):
        raise TypeError(f"Expected a PIL image or a tensor, got {type(image).__name__}")

    if not image.is_floating_point():
        return image.to(torch.float32).div_(torch.iinfo(image.dtype).max)

    # copy=True because the caller keeps its own reference: when no flip fires this tensor is
    # what the dataset hands on, and an in-place op downstream must not reach the caller's image.
    scaled = image.to(torch.float32, copy=True)
    if scaled.numel() and (scaled.min() < 0.0 or scaled.max() > 1.0):
        lo, hi = scaled.min().item(), scaled.max().item()
        raise ValueError(f"Float image must already be scaled to [0, 1], got [{lo:.4g}, {hi:.4g}]")
    return scaled


def _flip_boxes(boxes: Tensor, extent: int, axis: int) -> Tensor:
    """Mirror xyxy boxes along one axis (0 for x, 1 for y) of a frame `extent` wide."""
    if boxes.numel() == 0:
        return boxes
    lo, hi = axis, axis + 2
    flipped = boxes.clone()
    flipped[:, lo] = extent - boxes[:, hi]
    flipped[:, hi] = extent - boxes[:, lo]
    return flipped


@dataclass
class DetectionTransform:
    """Box-aware image transforms; a flip rewrites the boxes in the matching axis."""

    hflip_prob: float = 0.0
    vflip_prob: float = 0.0

    def __call__(
        self, image: Image.Image | Tensor, target: dict[str, Any]
    ) -> tuple[Tensor, dict[str, Any]]:
        # Detection models in torchvision normalise inside `model.transform`, so the image is
        # only scaled to [0, 1] here - applying ImageNet statistics twice would shift the input.
        tensor = to_float_tensor(image)
        height, width = tensor.shape[-2:]
        out = dict(target)
        boxes = out.get("boxes")

        # Draw only when a flip can actually fire, so an eval transform consumes nothing from
        # the global RNG and a seeded training run stays reproducible.
        if self.hflip_prob > 0.0 and torch.rand(1).item() < self.hflip_prob:
            tensor = torch.flip(tensor, dims=[-1])
            boxes = None if boxes is None else _flip_boxes(boxes, width, axis=0)
        if self.vflip_prob > 0.0 and torch.rand(1).item() < self.vflip_prob:
            tensor = torch.flip(tensor, dims=[-2])
            boxes = None if boxes is None else _flip_boxes(boxes, height, axis=1)

        if boxes is not None:
            out["boxes"] = boxes
        return tensor, out


def train_transform(hflip: float = 0.5) -> DetectionTransform:
    """Training transform: horizontal flips only, since sonar tiles have a fixed up-down sense."""
    return DetectionTransform(hflip_prob=hflip)


def eval_transform() -> DetectionTransform:
    """Evaluation transform: tensor conversion, no geometry."""
    return DetectionTransform()
