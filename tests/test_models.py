"""Tests for the detector factory, the gradient reversal layer and the domain heads."""

from __future__ import annotations

import pytest
import torch
from torch import nn

from sonar.models.detector import backbone_out_channels, build_detector, freeze_backbone
from sonar.models.grl import GradientReversal, gradient_reverse, ramp
from sonar.models.heads import DomainDiscriminator, ProposalDomainHead


@pytest.fixture
def detector():
    """A small, randomly initialised detector so the tests stay fast and offline."""
    torch.manual_seed(0)
    return build_detector(backbone="mobilenet", pretrained=False, min_size=64, max_size=64)


@pytest.fixture
def tiny_batch():
    """Two 64x64 images and matching targets, enough to drive a forward and backward pass."""
    torch.manual_seed(1)
    images = [torch.rand(3, 64, 64), torch.rand(3, 64, 64)]
    targets = [
        {"boxes": torch.tensor([[4.0, 4.0, 28.0, 28.0]]), "labels": torch.tensor([1])},
        {
            "boxes": torch.tensor([[8.0, 8.0, 40.0, 36.0], [40.0, 40.0, 60.0, 60.0]]),
            "labels": torch.tensor([1, 2]),
        },
    ]
    return images, targets


def test_gradient_reverse_leaves_forward_values_unchanged():
    x = torch.randn(4, 3, requires_grad=True)
    reversed_x = gradient_reverse(x, 2.5)

    assert torch.equal(reversed_x, x)
    # A detached copy would pass the value check and silently cut the adversarial path.
    assert reversed_x.requires_grad
    assert reversed_x.grad_fn is not None


def test_gradient_reverse_negates_and_scales_the_gradient():
    values = torch.tensor([1.0, -2.0, 3.0])
    x = values.clone().requires_grad_(True)
    gradient_reverse(x, 0.35).pow(2).sum().backward()

    # d/dx of sum(x**2) is 2x, so the layer must hand back exactly -coeff times that.
    assert torch.allclose(x.grad, -0.35 * 2.0 * values)


def test_set_coeff_changes_the_gradient_scale():
    layer = GradientReversal()
    x = torch.ones(3, requires_grad=True)

    layer(x).sum().backward()
    assert torch.equal(x.grad, torch.zeros(3))

    x.grad = None
    layer.set_coeff(0.75)
    layer(x).sum().backward()
    assert torch.allclose(x.grad, torch.full((3,), -0.75))


def test_ramp_starts_at_zero_and_saturates_at_max_coeff():
    assert ramp(0.0) == 0.0
    assert ramp(1.0) == pytest.approx(0.9999, abs=1e-4)
    assert ramp(1.0, max_coeff=0.2) == pytest.approx(0.2 * 0.9999, abs=1e-4)


def test_ramp_is_monotonic_and_clamps_out_of_range_progress():
    values = [ramp(step / 20, max_coeff=0.3) for step in range(21)]
    assert all(later > earlier for earlier, later in zip(values, values[1:]))

    # Negative progress must clamp to 0, not to a negative coefficient: a negative coefficient
    # would flip the gradient reversal back into ordinary alignment.
    assert ramp(-0.5, max_coeff=0.3) == 0.0
    assert ramp(4.0, max_coeff=0.3) == ramp(1.0, max_coeff=0.3)


def test_domain_discriminator_is_the_three_layer_mlp_with_dropout():
    head = DomainDiscriminator(in_features=256, hidden=32, dropout=0.25)

    assert [type(layer) for layer in head] == [
        nn.Linear,
        nn.ReLU,
        nn.Dropout,
        nn.Linear,
        nn.ReLU,
        nn.Dropout,
        nn.Linear,
    ]
    linears = [layer for layer in head if isinstance(layer, nn.Linear)]
    assert [(m.in_features, m.out_features) for m in linears] == [(256, 32), (32, 32), (32, 1)]
    assert [layer.p for layer in head if isinstance(layer, nn.Dropout)] == [0.25, 0.25]


def test_domain_discriminator_returns_one_logit_per_sample():
    head = DomainDiscriminator(in_features=256, hidden=32)
    assert head(torch.randn(5, 256)).shape == (5, 1)


def test_proposal_domain_head_pools_a_feature_map_to_one_logit():
    head = ProposalDomainHead(in_channels=256, hidden=8)

    # Non-square and odd sizes, because FPN levels are whatever the resized image produces.
    features = torch.randn(3, 256, 7, 9)
    logits = head(features)
    assert logits.shape == (3, 1)

    # A 1x1 first conv would score each pixel on its own; the RPN scores anchors over a
    # neighbourhood, so the head has to as well.
    assert head.conv.kernel_size == (3, 3)
    assert head.logit.kernel_size == (1, 1)

    per_location = head.logit(head.act(head.conv(features)))
    assert per_location.shape == (3, 1, 7, 9)
    assert torch.allclose(logits, per_location.sum(dim=(2, 3)) / (7 * 9), atol=1e-6)


def test_build_detector_predicts_three_classes(detector):
    predictor = detector.roi_heads.box_predictor
    assert predictor.cls_score.out_features == 3
    assert predictor.bbox_pred.out_features == 12
    assert detector.roi_heads.score_thresh == 0.0


def test_default_score_threshold_keeps_low_scoring_detections(detector, tiny_batch):
    images, _ = tiny_batch
    detector.eval()

    with torch.no_grad():
        unfiltered = detector(images)[0]
        # Same weights, same input: only the floor the original exports applied differs.
        detector.roi_heads.score_thresh = 0.5
        floored = detector(images)[0]

    assert unfiltered["scores"].numel() > floored["scores"].numel()
    assert float(unfiltered["scores"].min()) < 0.5


def test_build_detector_rejects_an_unknown_backbone():
    # Match the rejected name, not one of the suggestions: any ValueError naming a valid
    # backbone would satisfy a match on "resnet50".
    with pytest.raises(ValueError, match="vgg16"):
        build_detector(backbone="vgg16", pretrained=False)


def test_backbone_out_channels_matches_the_feature_maps(detector, tiny_batch):
    images, _ = tiny_batch
    with torch.no_grad():
        features = detector.backbone(torch.stack(images))

    channels = backbone_out_channels(detector)
    assert all(level.shape[1] == channels for level in features.values())


def test_freeze_backbone_round_trip_restores_a_partially_frozen_backbone(detector):
    # The pretrained builders hand back a backbone whose early layers are already frozen
    # (trainable_backbone_layers=3). Reproduce that here, because with a uniformly trainable
    # backbone "restore the previous flags" and "set everything True" are indistinguishable.
    names = [name for name, _ in detector.backbone.named_parameters()]
    held_back = set(names[: len(names) // 3])
    for name, param in detector.backbone.named_parameters():
        param.requires_grad_(name not in held_back)

    before = {name: p.requires_grad for name, p in detector.backbone.named_parameters()}
    assert not all(before.values())
    assert any(before.values())

    freeze_backbone(detector, True)
    assert not any(p.requires_grad for p in detector.backbone.parameters())

    # The original froze for a warm-up and never turned the backbone back on.
    freeze_backbone(detector, False)
    restored = {name: p.requires_grad for name, p in detector.backbone.named_parameters()}
    assert any(restored.values())
    assert restored == before


def test_repeated_unfreeze_does_not_widen_the_trainable_set(detector):
    # train.py calls freeze_backbone(model, False) once per epoch after the warm-up and again
    # when training ends, so the second call must not promote the held-back parameters.
    names = [name for name, _ in detector.backbone.named_parameters()]
    held_back = set(names[: len(names) // 3])
    for name, param in detector.backbone.named_parameters():
        param.requires_grad_(name not in held_back)
    before = {name: p.requires_grad for name, p in detector.backbone.named_parameters()}

    freeze_backbone(detector, True)
    freeze_backbone(detector, False)
    freeze_backbone(detector, False)

    assert {name: p.requires_grad for name, p in detector.backbone.named_parameters()} == before


def test_frozen_backbone_receives_no_gradient_and_the_heads_still_do(detector, tiny_batch):
    images, targets = tiny_batch
    freeze_backbone(detector, True)
    detector.train()

    losses = detector(images, targets)
    sum(losses.values()).backward()

    assert all(p.grad is None for p in detector.backbone.parameters())
    assert detector.roi_heads.box_predictor.cls_score.weight.grad is not None
