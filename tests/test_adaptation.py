"""Tests for the DANN / DCCAN wrapper.

Everything runs on a randomly initialised mobilenet detector at 64x64 so the suite stays offline
and quick. The adversarial losses are not expected to be *good* here, only well formed.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F
from torch import Tensor, nn

from sonar.models.adaptation import (
    AdaptationConfig,
    DomainAdaptiveDetector,
    confidence_threshold,
)
from sonar.models.detector import build_detector

DETECTION_KEYS = {"loss_classifier", "loss_box_reg", "loss_objectness", "loss_rpn_box_reg"}

# Every domain batch below is two source images followed by two target images, and the wrapper
# labels source 1 / target 0 in that order.
DOMAIN_LABELS = torch.tensor([1.0, 1.0, 0.0, 0.0])

# Class presence implied by the fixture labels: one image holds an object, the other an object
# and a shadow. Column 0 is background, which never appears in a ground-truth target.
SOURCE_PRESENCE = torch.tensor([[0.0, 1.0, 0.0], [0.0, 1.0, 1.0]])


@pytest.fixture
def domain_batch():
    """A labelled source pair and an unlabelled target pair.

    Kept local rather than shared: the class-presence expectations below are derived from these
    exact labels, so they should be readable next to them.
    """
    torch.manual_seed(1)
    source = [torch.rand(3, 64, 64), torch.rand(3, 64, 64)]
    targets = [
        {"boxes": torch.tensor([[4.0, 4.0, 28.0, 28.0]]), "labels": torch.tensor([1])},
        {
            "boxes": torch.tensor([[8.0, 8.0, 40.0, 36.0], [40.0, 40.0, 60.0, 60.0]]),
            "labels": torch.tensor([1, 2]),
        },
    ]
    target_images = [torch.rand(3, 64, 64), torch.rand(3, 64, 64)]
    return source, targets, target_images


def make_model(mode: str = "dccan", **cfg_kwargs) -> DomainAdaptiveDetector:
    torch.manual_seed(0)
    detector = build_detector(
        num_classes=3, backbone="mobilenet", pretrained=False, min_size=64, max_size=64
    )
    model = DomainAdaptiveDetector(detector, AdaptationConfig(**cfg_kwargs), mode=mode)
    model.train()
    return model


def capture_around(module: nn.Module) -> list[dict[str, Tensor]]:
    """Retain the tensors either side of every call to `module`, so their gradients compare."""
    calls: list[dict[str, Tensor]] = []

    def before(_module, args):
        args[0].retain_grad()
        calls.append({"into": args[0]})

    def after(_module, _args, output):
        output.retain_grad()
        calls[-1]["out_of"] = output

    module.register_forward_pre_hook(before)
    module.register_forward_hook(after)
    return calls


def test_forward_returns_finite_scalar_losses_under_every_expected_key(domain_batch):
    model = make_model()

    losses = model(*domain_batch)

    expected = DETECTION_KEYS | {"loss_dann", "loss_cdan", "loss_proposal", "loss_proxy_aux"}
    assert set(losses) == expected
    for name, value in losses.items():
        assert value.shape == (), f"{name} is not a scalar"
        assert torch.isfinite(value), f"{name} is not finite"


def test_forward_refuses_to_run_in_eval_mode(domain_batch):
    """Eval mode empties the RPN and ROI loss dicts, so the caller would train the adversary alone."""
    model = make_model()
    model.eval()

    with pytest.raises(RuntimeError, match="training mode"):
        model(*domain_batch)


def test_proxy_classifier_receives_a_real_gradient(domain_batch):
    """Regression test for B3: the original proxy head only ever saw a detached softmax."""
    model = make_model()

    losses = model(*domain_batch)
    sum(losses.values()).backward()

    grad = model.proxy_cls.weight.grad
    assert grad is not None
    assert torch.count_nonzero(grad) > 0


def test_cdan_conditioning_does_not_backpropagate_into_the_proxy_classifier(domain_batch):
    """The conditioning softmax stays detached; only loss_proxy_aux is allowed to train the head."""
    model = make_model(conf_thresh_start=0.0, conf_thresh_end=0.0)
    model.set_progress(1.0)

    model(*domain_batch)["loss_cdan"].backward()

    assert model.proxy_cls.weight.grad is None
    reached = [p.grad for p in model.detector.backbone.parameters() if p.grad is not None]
    assert any(torch.count_nonzero(g) > 0 for g in reached), "loss_cdan carried no gradient at all"


def test_proxy_classifier_weights_change_across_optimiser_steps(domain_batch):
    model = make_model()
    before = model.proxy_cls.weight.detach().clone()
    optimiser = torch.optim.SGD(model.parameters(), lr=0.05)

    for _ in range(2):
        optimiser.zero_grad()
        losses = model(*domain_batch)
        sum(losses.values()).backward()
        optimiser.step()

    assert not torch.allclose(before, model.proxy_cls.weight.detach())


class _StubProxy(nn.Module):
    """Proxy-classifier stand-in: the first `confident` rows spike on class 1, the rest are flat."""

    def __init__(self, num_classes: int, *, confident: int = 0, flat: float = 0.0) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.confident = confident
        self.flat = flat

    def forward(self, x: Tensor) -> Tensor:
        logits = torch.full((x.shape[0], self.num_classes), self.flat)
        logits[: self.confident, 1] = 20.0
        return logits


AUX_LOGIT = 10.0


def test_proxy_aux_loss_targets_the_classes_present_in_the_source_batch(domain_batch):
    """A constant target would satisfy the B3 fix on paper while teaching the head nothing."""
    model = make_model()
    model.proxy_cls = _StubProxy(model.num_classes, flat=AUX_LOGIT)

    aux = model(*domain_batch)["loss_proxy_aux"]

    expected = F.binary_cross_entropy_with_logits(
        torch.full((2, model.num_classes), AUX_LOGIT), SOURCE_PRESENCE
    )
    assert aux.item() == pytest.approx(expected.item(), rel=1e-5)


def test_proxy_aux_weight_scales_the_auxiliary_loss(domain_batch):
    model = make_model(proxy_aux_weight=2.0)
    model.proxy_cls = _StubProxy(model.num_classes, flat=AUX_LOGIT)

    aux = model(*domain_batch)["loss_proxy_aux"]

    expected = F.binary_cross_entropy_with_logits(
        torch.full((2, model.num_classes), AUX_LOGIT), SOURCE_PRESENCE
    )
    assert aux.item() == pytest.approx(2.0 * expected.item(), rel=1e-5)


def test_backbone_runs_once_per_domain_and_transform_sees_the_target_images(domain_batch):
    """Regression test for B4/B5: one transformed backbone pass per domain, not three raw ones."""
    model = make_model()
    source, _, target_images = domain_batch

    backbone_calls = []
    transform_inputs = []
    model.detector.backbone.register_forward_pre_hook(
        lambda _module, args: backbone_calls.append(args[0])
    )
    model.detector.transform.register_forward_pre_hook(
        lambda _module, args: transform_inputs.append(args[0])
    )

    model(*domain_batch)

    assert len(backbone_calls) == 2
    assert len(transform_inputs) == 2
    assert transform_inputs[0] is source
    assert transform_inputs[1] is target_images

    # The buggy path stacked the raw target images; the transform resizes and normalises instead,
    # so the tensor the backbone actually saw is not the naive stack.
    assert not torch.allclose(backbone_calls[1], torch.stack(target_images))


@pytest.mark.parametrize(
    ("grl_name", "loss_key", "calls"),
    [
        ("global_grl", "loss_dann", 1),
        # The proposal head runs per domain, so its GRL has to cover both feature maps.
        ("proposal_grl", "loss_proposal", 2),
        ("cdan_grl", "loss_cdan", 1),
    ],
)
def test_every_adversarial_loss_flows_through_its_gradient_reversal(
    domain_batch, grl_name, loss_key, calls
):
    """Without the GRL in the graph the domain heads would help the backbone separate domains."""
    model = make_model()
    model.set_progress(1.0)
    grl = getattr(model, grl_name)
    captured = capture_around(grl)

    model(*domain_batch)[loss_key].backward()

    assert len(captured) == calls, f"{grl_name} ran {len(captured)} times in the {loss_key} path"
    for call in captured:
        assert torch.count_nonzero(call["out_of"].grad) > 0
        assert torch.allclose(call["into"].grad, -grl.coeff * call["out_of"].grad)


def test_progress_zero_disables_every_gradient_reversal():
    model = make_model()
    model.set_progress(0.0)

    assert model.global_grl.coeff == 0.0
    assert model.proposal_grl.coeff == 0.0
    assert model.cdan_grl.coeff == 0.0


def test_progress_one_approaches_the_configured_maxima():
    cfg = dict(dann_lambda=0.2, cdan_lambda=0.3, proposal_lambda=0.1)
    model = make_model(**cfg)
    model.set_progress(1.0)

    assert model.global_grl.coeff == pytest.approx(cfg["dann_lambda"], rel=1e-3)
    assert model.cdan_grl.coeff == pytest.approx(cfg["cdan_lambda"], rel=1e-3)
    assert model.proposal_grl.coeff == pytest.approx(cfg["proposal_lambda"], rel=1e-3)


def test_gradient_reversal_coefficients_increase_with_progress():
    model = make_model()
    coeffs = []
    for progress in (0.0, 0.25, 0.5, 1.0):
        model.set_progress(progress)
        coeffs.append(model.global_grl.coeff)

    assert coeffs == sorted(coeffs)
    assert coeffs[0] < coeffs[-1]


def test_confidence_threshold_decays_over_the_first_half_then_holds():
    cfg = AdaptationConfig(conf_thresh_start=0.4, conf_thresh_end=0.2)

    assert confidence_threshold(0.0, cfg) == pytest.approx(0.4)
    assert confidence_threshold(0.25, cfg) == pytest.approx(0.3)
    assert confidence_threshold(0.5, cfg) == pytest.approx(0.2)
    assert confidence_threshold(1.0, cfg) == pytest.approx(0.2)
    assert confidence_threshold(2.0, cfg) == pytest.approx(0.2)


class _FixedDiscriminator(nn.Module):
    """Domain discriminator stand-in with one preset logit per row of the batch."""

    def __init__(self, logits: list[float]) -> None:
        super().__init__()
        self.logits = logits

    def forward(self, x: Tensor) -> Tensor:
        assert x.shape[0] == len(self.logits), "the stub was built for a different batch size"
        return torch.tensor(self.logits).unsqueeze(1)


# One cheap row and three expensive ones, so the gate's divisor shows up in the result: the cheap
# row is the only confident one, and averaging over all four instead would be four times smaller.
GATE_LOGITS = [0.0, -10.0, 10.0, 10.0]


def gated_cdan_loss(domain_batch, *, threshold: float, confident: int) -> float:
    """CDAN loss with both halves of the conditioning replaced by fixed, hand-computable values."""
    model = make_model(conf_thresh_start=threshold, conf_thresh_end=threshold)
    model.proxy_cls = _StubProxy(model.num_classes, confident=confident)
    model.cdan_disc = _FixedDiscriminator(GATE_LOGITS)
    return model(*domain_batch)["loss_cdan"].item()


def test_confidence_gate_averages_over_the_kept_samples_only(domain_batch):
    """A partial gate is the only case that separates the kept-count divisor from the batch size."""
    per_sample = F.binary_cross_entropy_with_logits(
        torch.tensor(GATE_LOGITS), DOMAIN_LABELS, reduction="none"
    )

    loss = gated_cdan_loss(domain_batch, threshold=0.5, confident=1)

    assert loss == pytest.approx(per_sample[0].item(), rel=1e-5)


def test_open_confidence_gate_keeps_every_sample(domain_batch):
    batch_mean = F.binary_cross_entropy_with_logits(torch.tensor(GATE_LOGITS), DOMAIN_LABELS)

    loss = gated_cdan_loss(domain_batch, threshold=0.0, confident=1)

    assert loss == pytest.approx(batch_mean.item(), rel=1e-5)


def test_confidence_gate_that_keeps_nothing_yields_zero_rather_than_nan(domain_batch):
    model = make_model(conf_thresh_start=1.1, conf_thresh_end=1.1)
    model.set_progress(0.0)

    losses = model(*domain_batch)

    assert torch.isfinite(losses["loss_cdan"])
    assert losses["loss_cdan"].item() == 0.0


def test_dann_mode_omits_the_class_conditional_losses(domain_batch):
    model = make_model(mode="dann")

    losses = model(*domain_batch)

    assert set(losses) == DETECTION_KEYS | {"loss_dann"}
    assert not hasattr(model, "proxy_cls")


def test_roi_conditioning_produces_a_finite_cdan_loss(domain_batch):
    model = make_model(conditioning="roi")

    losses = model(*domain_batch)

    assert torch.isfinite(losses["loss_cdan"])
    assert not hasattr(model, "proxy_cls")
    assert "loss_proxy_aux" not in losses


def test_roi_conditioning_draws_both_domains_from_an_eval_mode_rpn(domain_batch):
    """Mixed RPN modes hand the two domains differently ranked proposal pools to separate on."""
    model = make_model(conditioning="roi")
    rpn_modes = []
    model.detector.rpn.register_forward_pre_hook(
        lambda module, _args: rpn_modes.append(module.training)
    )

    model(*domain_batch)

    # One training-mode call for the detection losses, then one per domain for the ROI pools.
    assert rpn_modes == [True, False, False]


def test_roi_conditioning_gradient_reaches_the_backbone(domain_batch):
    model = make_model(conditioning="roi")
    model.set_progress(1.0)

    model(*domain_batch)["loss_cdan"].backward()

    grads = [p.grad for p in model.detector.backbone.parameters() if p.grad is not None]
    assert grads and any(torch.count_nonzero(g) > 0 for g in grads)


def test_unknown_feature_level_names_the_levels_the_backbone_offers(domain_batch):
    model = make_model(feature_level="not_a_level")

    with pytest.raises(KeyError) as excinfo:
        model(*domain_batch)

    message = str(excinfo.value)
    assert "not_a_level" in message
    assert "pool" in message


class _SplitDiscriminator(nn.Module):
    """Emits a confident source logit for the first half of the batch and a target logit after."""

    def __init__(self, n_source: int) -> None:
        super().__init__()
        self.n_source = n_source

    def forward(self, x: Tensor) -> Tensor:
        signs = torch.full((x.shape[0], 1), -10.0)
        signs[: self.n_source] = 10.0
        return signs


def test_domain_labels_are_source_one_target_zero(domain_batch):
    model = make_model(mode="dann")
    source, _, _ = domain_batch
    model.domain_disc = _SplitDiscriminator(len(source))

    losses = model(*domain_batch)

    # A near-zero BCE means the +10 logits were scored against label 1 and the -10 against 0.
    assert losses["loss_dann"].item() < 1e-3

    model.domain_disc = _SplitDiscriminator(0)
    flipped = model(*domain_batch)
    all_wrong = F.binary_cross_entropy_with_logits(torch.full((4,), -10.0), DOMAIN_LABELS)
    assert flipped["loss_dann"].item() == pytest.approx(all_wrong.item(), rel=1e-5)


def test_predict_returns_detections_and_restores_training_mode(domain_batch):
    model = make_model()
    images, _, _ = domain_batch

    outputs = model.predict(images)

    assert len(outputs) == len(images)
    assert {"boxes", "scores", "labels"} <= set(outputs[0])
    assert model.detector.training


def test_unknown_mode_and_conditioning_are_rejected():
    detector = build_detector(
        num_classes=3, backbone="mobilenet", pretrained=False, min_size=64, max_size=64
    )

    for bad_mode in ("cdan", ""):
        with pytest.raises(ValueError, match="mode must be"):
            DomainAdaptiveDetector(detector, AdaptationConfig(), mode=bad_mode)

    with pytest.raises(ValueError, match="conditioning must be"):
        DomainAdaptiveDetector(detector, AdaptationConfig(conditioning="none"))
