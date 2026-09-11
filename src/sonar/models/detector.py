"""Faster R-CNN construction and backbone freezing."""

from __future__ import annotations

from collections.abc import MutableMapping
from weakref import WeakKeyDictionary

from torch import nn
from torchvision.models import MobileNet_V3_Large_Weights, ResNet50_Weights
from torchvision.models.detection import (
    FasterRCNN_MobileNet_V3_Large_FPN_Weights,
    FasterRCNN_ResNet50_FPN_Weights,
    fasterrcnn_mobilenet_v3_large_fpn,
    fasterrcnn_resnet50_fpn,
)
from torchvision.models.detection.faster_rcnn import FasterRCNN, FastRCNNPredictor

_BACKBONES = {
    "resnet50": (
        fasterrcnn_resnet50_fpn,
        FasterRCNN_ResNet50_FPN_Weights,
        ResNet50_Weights,
    ),
    "mobilenet": (
        fasterrcnn_mobilenet_v3_large_fpn,
        FasterRCNN_MobileNet_V3_Large_FPN_Weights,
        MobileNet_V3_Large_Weights,
    ),
}


def build_detector(
    num_classes: int = 3,
    *,
    backbone: str = "resnet50",
    pretrained: bool = True,
    min_size: int = 800,
    max_size: int = 1333,
    score_thresh: float = 0.0,
    nms_thresh: float = 0.5,
    detections_per_img: int = 300,
) -> FasterRCNN:
    """Faster R-CNN with a `num_classes`-way box predictor, background included.

    `score_thresh` defaults to 0.0 so predictions keep their full score range: anything higher
    truncates the precision-recall curve before the metrics are ever computed, which is how the
    original project ended up reporting average precision from 0.5-floored exports.
    """
    if backbone not in _BACKBONES:
        raise ValueError(f"unknown backbone {backbone!r}, expected one of {sorted(_BACKBONES)}")

    builder, detector_weights, backbone_weights = _BACKBONES[backbone]
    detector = builder(
        weights=detector_weights.DEFAULT if pretrained else None,
        weights_backbone=backbone_weights.DEFAULT if pretrained else None,
        min_size=min_size,
        max_size=max_size,
        box_score_thresh=score_thresh,
        box_nms_thresh=nms_thresh,
        box_detections_per_img=detections_per_img,
    )

    in_features = detector.roi_heads.box_predictor.cls_score.in_features
    detector.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return detector


def backbone_out_channels(detector: FasterRCNN) -> int:
    """Channel width of the FPN feature maps, which the domain heads have to match."""
    return int(detector.backbone.out_channels)


# Keyed on the backbone so a garbage-collected detector takes its snapshot with it.
_PRE_FREEZE_REQUIRES_GRAD: MutableMapping[nn.Module, dict[str, bool]] = WeakKeyDictionary()


def freeze_backbone(detector: FasterRCNN, frozen: bool) -> None:
    """Freeze the backbone, or put back the flags it had before the first freeze.

    Unfreezing cannot simply enable everything: torchvision's pretrained builders hold the stem
    back (`trainable_backbone_layers=3`), and a round trip that ignored that would quietly hand
    a warm-up schedule more trainable parameters than the caller asked for. A backbone this
    function never froze has no snapshot, so it falls back to enabling everything.
    """
    backbone = detector.backbone
    if frozen:
        # Snapshot once: a second freeze would otherwise record the all-frozen state as the
        # thing to restore, which is exactly the state we are trying to escape.
        if backbone not in _PRE_FREEZE_REQUIRES_GRAD:
            _PRE_FREEZE_REQUIRES_GRAD[backbone] = {
                name: param.requires_grad for name, param in backbone.named_parameters()
            }
        for param in backbone.parameters():
            param.requires_grad_(False)
        return

    saved = _PRE_FREEZE_REQUIRES_GRAD.get(backbone, {})
    for name, param in backbone.named_parameters():
        param.requires_grad_(saved.get(name, True))
