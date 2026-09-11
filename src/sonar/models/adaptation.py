"""Unsupervised domain adaptation wrappers (DANN and DCCAN) around a torchvision FasterRCNN.

The wrapper deliberately re-implements the detector's forward pass instead of calling
`detector(images, targets)`. Doing so lets one backbone pass per domain feed the RPN, the ROI
heads and the domain heads at once, and forces the target domain through `detector.transform`
so the discriminator sees the same input distribution as the detector.
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.image_list import ImageList
from torchvision.models.detection.roi_heads import RoIHeads

from sonar.models.detector import backbone_out_channels
from sonar.models.grl import GradientReversal, ramp
from sonar.models.heads import DomainDiscriminator, ProposalDomainHead

MAX_ROIS_PER_IMAGE = 256


@dataclass
class AdaptationConfig:
    """Loss weights and schedules for the adversarial alignment heads."""

    dann_lambda: float = 0.20
    cdan_lambda: float = 0.30
    proposal_lambda: float = 0.10
    cdan_temperature: float = 0.6
    conf_thresh_start: float = 0.40
    conf_thresh_end: float = 0.20
    conditioning: str = "proxy"
    proxy_aux_weight: float = 1.0
    feature_level: str = "0"


def confidence_threshold(progress: float, cfg: AdaptationConfig) -> float:
    """Decay the CDAN confidence gate over the first half of training, then hold it flat."""
    clamped = min(max(progress, 0.0), 1.0)
    fraction = min(clamped / 0.5, 1.0)
    span = cfg.conf_thresh_end - cfg.conf_thresh_start
    return cfg.conf_thresh_start + span * fraction


@contextmanager
def _inference_mode(module: nn.Module) -> Iterator[None]:
    """Temporarily put a submodule in eval mode without touching the rest of the model."""
    was_training = module.training
    module.train(False)
    try:
        yield
    finally:
        module.train(was_training)


def _as_feature_dict(features: Tensor | dict[str, Tensor]) -> dict[str, Tensor]:
    # An FPN backbone hands back a dict of levels, a plain one a single tensor. Naming that
    # tensor "0" lets cfg.feature_level address either backbone the same way.
    if isinstance(features, Tensor):
        return OrderedDict([("0", features)])
    return features


def _subsample(count: int, limit: int, device: torch.device) -> Tensor:
    if count <= limit:
        return torch.arange(count, device=device)
    return torch.randperm(count, device=device)[:limit]


class DomainAdaptiveDetector(nn.Module):
    """FasterRCNN plus adversarial domain heads; `forward` returns unsummed named losses."""

    def __init__(
        self,
        detector: FasterRCNN,
        cfg: AdaptationConfig,
        mode: str = "dccan",
    ) -> None:
        super().__init__()
        if mode not in {"dann", "dccan"}:
            raise ValueError(f"mode must be 'dann' or 'dccan', got {mode!r}")
        if cfg.conditioning not in {"proxy", "roi"}:
            raise ValueError(f"conditioning must be 'proxy' or 'roi', got {cfg.conditioning!r}")

        self.detector = detector
        self.cfg = cfg
        self.mode = mode

        channels = backbone_out_channels(detector)
        self.num_classes = detector.roi_heads.box_predictor.cls_score.out_features

        self.global_grl = GradientReversal()
        self.domain_disc = DomainDiscriminator(channels)

        if mode == "dccan":
            self.proposal_grl = GradientReversal()
            self.proposal_head = ProposalDomainHead(channels)
            self.cdan_grl = GradientReversal()
            if cfg.conditioning == "proxy":
                self.proxy_cls = nn.Linear(channels, self.num_classes)
                conditioned_dim = channels
            else:
                conditioned_dim = detector.roi_heads.box_predictor.cls_score.in_features
            self.cdan_disc = DomainDiscriminator(conditioned_dim * self.num_classes)

        self._progress = 0.0
        self.set_progress(0.0)

    @property
    def progress(self) -> float:
        return self._progress

    def set_progress(self, progress: float) -> None:
        """Update every GRL coefficient from the training-progress ramp."""
        self._progress = float(progress)
        self.global_grl.set_coeff(ramp(self._progress, self.cfg.dann_lambda))
        if self.mode == "dccan":
            self.proposal_grl.set_coeff(ramp(self._progress, self.cfg.proposal_lambda))
            self.cdan_grl.set_coeff(ramp(self._progress, self.cfg.cdan_lambda))

    def forward(
        self,
        source_images: list[Tensor],
        source_targets: list[dict],
        target_images: list[Tensor],
    ) -> dict[str, Tensor]:
        # In eval mode the RPN and ROI heads return detections and empty loss dicts, so the
        # caller would sum the domain losses alone and train the adversary against no detector.
        if not self.training:
            raise RuntimeError("forward() requires training mode; call predict() for inference")

        detector = self.detector
        cfg = self.cfg

        source_list, source_targets = detector.transform(source_images, source_targets)
        source_features = _as_feature_dict(detector.backbone(source_list.tensors))

        # The unlabelled domain goes through the same normalisation and resizing; feeding the
        # backbone raw stacked tensors here was the original B4 bug.
        target_list, _ = detector.transform(target_images, None)
        target_features = _as_feature_dict(detector.backbone(target_list.tensors))

        proposals, rpn_losses = detector.rpn(source_list, source_features, source_targets)
        _, roi_losses = detector.roi_heads(
            source_features, proposals, source_list.image_sizes, source_targets
        )

        losses: dict[str, Tensor] = {}
        losses.update(rpn_losses)
        losses.update(roi_losses)

        source_map = self._level(source_features)
        target_map = self._level(target_features)
        source_pooled = F.adaptive_avg_pool2d(source_map, 1).flatten(1)
        target_pooled = F.adaptive_avg_pool2d(target_map, 1).flatten(1)

        losses["loss_dann"] = self._global_loss(source_pooled, target_pooled)

        if self.mode == "dann":
            return losses

        losses["loss_proposal"] = self._proposal_loss(source_map, target_map)

        if cfg.conditioning == "proxy":
            cdan_loss, aux_loss = self._proxy_cdan(source_pooled, target_pooled, source_targets)
            losses["loss_cdan"] = cdan_loss
            losses["loss_proxy_aux"] = aux_loss
        else:
            losses["loss_cdan"] = self._roi_cdan(
                source_features, source_list, target_features, target_list
            )
        return losses

    @torch.no_grad()
    def predict(self, images: list[Tensor]) -> list[dict]:
        """Run plain detection inference, leaving the wrapper's training mode untouched."""
        was_training = self.detector.training
        self.detector.eval()
        try:
            return self.detector(images)
        finally:
            self.detector.train(was_training)

    def _level(self, features: dict[str, Tensor]) -> Tensor:
        level = self.cfg.feature_level
        if level not in features:
            available = ", ".join(features)
            raise KeyError(f"feature level {level!r} not in backbone output; have: {available}")
        return features[level]

    def _global_loss(self, source_pooled: Tensor, target_pooled: Tensor) -> Tensor:
        pooled = torch.cat([source_pooled, target_pooled], dim=0)
        logits = self.domain_disc(self.global_grl(pooled)).flatten()
        labels = self._domain_labels(source_pooled.shape[0], target_pooled.shape[0], logits)
        return F.binary_cross_entropy_with_logits(logits, labels)

    def _proposal_loss(self, source_map: Tensor, target_map: Tensor) -> Tensor:
        source_logits = self.proposal_head(self.proposal_grl(source_map)).flatten()
        target_logits = self.proposal_head(self.proposal_grl(target_map)).flatten()
        logits = torch.cat([source_logits, target_logits], dim=0)
        labels = self._domain_labels(source_logits.shape[0], target_logits.shape[0], logits)
        return F.binary_cross_entropy_with_logits(logits, labels)

    def _proxy_cdan(
        self,
        source_pooled: Tensor,
        target_pooled: Tensor,
        source_targets: list[dict],
    ) -> tuple[Tensor, Tensor]:
        pooled = torch.cat([source_pooled, target_pooled], dim=0)
        logits = self.proxy_cls(pooled)
        n_source = source_pooled.shape[0]

        # The auxiliary task is what keeps the proxy classifier alive (B3): the CDAN branch
        # detaches its softmax, so without this loss the classifier would never see a gradient.
        presence = self._class_presence(source_targets, source_pooled)
        aux = F.binary_cross_entropy_with_logits(logits[:n_source], presence)
        aux = aux * self.cfg.proxy_aux_weight

        cdan = self._conditional_loss(pooled, logits, n_source, pooled.shape[0] - n_source)
        return cdan, aux

    def _roi_cdan(
        self,
        source_features: dict[str, Tensor],
        source_list: ImageList,
        target_features: dict[str, Tensor],
        target_list: ImageList,
    ) -> Tensor:
        roi_heads = self.detector.roi_heads
        # The RPN needs eval mode to hand back target proposals without demanding boxes. Source
        # proposals are re-drawn the same way rather than reused from the training-mode call
        # above: the two modes keep different numbers of proposals, and a discriminator handed
        # differently ranked pools can tell the domains apart on ranking instead of on features.
        with _inference_mode(self.detector.rpn):
            source_proposals, _ = self.detector.rpn(source_list, source_features, None)
            target_proposals, _ = self.detector.rpn(target_list, target_features, None)

        source_feats, source_logits = self._roi_features(
            source_features, source_proposals, source_list.image_sizes, roi_heads
        )
        target_feats, target_logits = self._roi_features(
            target_features, target_proposals, target_list.image_sizes, roi_heads
        )
        feats = torch.cat([source_feats, target_feats], dim=0)
        logits = torch.cat([source_logits, target_logits], dim=0)
        return self._conditional_loss(
            feats, logits, source_feats.shape[0], target_feats.shape[0]
        )

    def _roi_features(
        self,
        features: dict[str, Tensor],
        proposals: list[Tensor],
        image_sizes: list[tuple[int, int]],
        roi_heads: RoIHeads,
    ) -> tuple[Tensor, Tensor]:
        limited = [
            boxes[_subsample(boxes.shape[0], MAX_ROIS_PER_IMAGE, boxes.device)]
            for boxes in proposals
        ]
        pooled = roi_heads.box_roi_pool(features, limited, image_sizes)
        box_features = roi_heads.box_head(pooled)
        class_logits, _ = roi_heads.box_predictor(box_features)
        return box_features, class_logits

    def _conditional_loss(
        self,
        features: Tensor,
        logits: Tensor,
        n_source: int,
        n_target: int,
    ) -> Tensor:
        probs = F.softmax(logits / self.cfg.cdan_temperature, dim=1).detach()
        normed = F.normalize(features, dim=1)
        # Class-major flattening of the outer product. The layout is arbitrary as long as both
        # domains share it, since the discriminator learns whatever ordering it is given.
        conditioned = torch.bmm(probs.unsqueeze(2), normed.unsqueeze(1)).flatten(1)

        domain_logits = self.cdan_disc(self.cdan_grl(conditioned)).flatten()
        labels = self._domain_labels(n_source, n_target, domain_logits)
        per_sample = F.binary_cross_entropy_with_logits(domain_logits, labels, reduction="none")

        threshold = confidence_threshold(self._progress, self.cfg)
        keep = (probs.max(dim=1).values >= threshold).to(per_sample.dtype)
        # Averaging over kept samples, not over the batch, keeps the loss on the same scale as
        # the gate closes; the clamp is what turns an all-dropped batch into 0 instead of NaN.
        return (per_sample * keep).sum() / keep.sum().clamp(min=1.0)

    def _class_presence(self, targets: list[dict], reference: Tensor) -> Tensor:
        # Column 0 stays in the target and stays zero: the proxy head has to span the detector's
        # class space for the outer product, and background never appears as a ground-truth label.
        presence = torch.zeros(
            len(targets), self.num_classes, device=reference.device, dtype=reference.dtype
        )
        for row, target in enumerate(targets):
            labels = target["labels"]
            if labels.numel():
                presence[row, labels.long()] = 1.0
        return presence

    @staticmethod
    def _domain_labels(n_source: int, n_target: int, reference: Tensor) -> Tensor:
        """Source is 1, target is 0, in that concatenation order."""
        return torch.cat(
            [
                torch.ones(n_source, device=reference.device, dtype=reference.dtype),
                torch.zeros(n_target, device=reference.device, dtype=reference.dtype),
            ]
        )
