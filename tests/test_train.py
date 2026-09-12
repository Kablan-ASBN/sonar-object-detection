"""Tests for the training loops, the optimizer grouping and checkpoint compatibility."""

from __future__ import annotations

import math
import time
from collections.abc import Callable

import pytest
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader

from sonar.data.transforms import eval_transform
from sonar.data.voc import VOCDetection, collate_detection
from sonar.engine import train as train_module
from sonar.engine.train import (
    EpochRecord,
    TrainConfig,
    build_optimizer,
    load_checkpoint,
    save_checkpoint,
    train_adaptive,
    train_baseline,
)
from sonar.models.adaptation import AdaptationConfig, DomainAdaptiveDetector
from sonar.models.detector import build_detector

BATCHES_PER_EPOCH = 4  # 8 train ids in the fixture split, two per batch


@pytest.fixture
def detector():
    """A randomly initialised mobilenet detector at 64 px, small enough to train in a test."""
    torch.manual_seed(0)
    return build_detector(
        num_classes=3, backbone="mobilenet", pretrained=False, min_size=64, max_size=64
    )


@pytest.fixture
def loader(voc_root):
    dataset = VOCDetection(voc_root, image_set="train", transforms=eval_transform())
    return DataLoader(dataset, batch_size=2, shuffle=False, collate_fn=collate_detection)


@pytest.fixture
def built_optimizers(monkeypatch):
    """Captures the optimizer a training loop builds; the loops never hand it back."""
    built: list[torch.optim.Optimizer] = []
    real = train_module.build_optimizer

    def capture(*args, **kwargs):
        optimizer = real(*args, **kwargs)
        built.append(optimizer)
        return optimizer

    monkeypatch.setattr(train_module, "build_optimizer", capture)
    return built


class StubDetector(nn.Module):
    """Loop-mechanics stand-in: real parameters and a backbone, but no convolutions to wait for.

    The head stays trainable while the backbone is frozen, which is what a real detector does
    during a warm-up epoch and what keeps the loss attached to the graph.
    """

    def __init__(self) -> None:
        super().__init__()
        self.backbone = nn.Linear(3, 2)
        self.head = nn.Linear(2, 1)
        self.batches = 0

    def forward(self, images, targets):
        self.batches += 1
        pooled = torch.stack([image.mean() for image in images]).sum()
        features = pooled * self.backbone.weight.sum()
        return {"loss_classifier": (features * self.head.weight.sum()).pow(2)}


class StubAdaptive(nn.Module):
    """Stands in for DomainAdaptiveDetector: a wrapped detector, a domain head, a progress log."""

    def __init__(self) -> None:
        super().__init__()
        self.detector = StubDetector()
        self.domain_disc = nn.Linear(1, 1)
        self.progress: list[float] = []

    def set_progress(self, progress: float) -> None:
        self.progress.append(progress)

    def forward(self, source_images, source_targets, target_images):
        pixels = torch.stack([image.mean() for image in [*source_images, *target_images]]).sum()
        hidden = pixels * self.detector.backbone.weight.sum()
        return {"loss_dann": self.domain_disc(hidden.reshape(1, 1)).squeeze().pow(2)}


class ScriptedLosses(nn.Module):
    """Returns preset loss values so an epoch's recorded means can be checked by hand.

    The parameter is multiplied by zero: the loss stays attached to the graph, but the optimizer
    step cannot move the next batch's value.
    """

    def __init__(self, script: list[dict[str, float]]) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1))
        self.script = script
        self.batches = 0

    def forward(self, images, targets):
        values = self.script[self.batches % len(self.script)]
        self.batches += 1
        attached = self.weight.sum() * 0.0
        return {key: attached + value for key, value in values.items()}


class ExplodingAdaptive(nn.Module):
    """Adaptive-shaped stub with gradients far above any clip threshold.

    The domain head earns a second optimizer group, which is where clipping per group and
    clipping over everything at once disagree.
    """

    grad_per_weight = 1e4

    def __init__(self) -> None:
        super().__init__()
        self.detector = nn.Linear(1, 1)
        self.domain_disc = nn.Linear(1, 1)

    def set_progress(self, progress: float) -> None:
        pass

    def forward(self, source_images, source_targets, target_images):
        weights = self.detector.weight.sum() + self.domain_disc.weight.sum()
        return {"loss_dann": weights * self.grad_per_weight}


class RecordingScaler:
    """Logs the calls `_optimise` makes, delegating to a disabled scaler.

    AMP is unobservable on CPU - a disabled GradScaler makes `unscale_` a no-op - so the order of
    unscale and clip can only be checked by watching the calls themselves.
    """

    def __init__(self, events: list[str]) -> None:
        self.events = events
        # Mirrors _grad_scaler: torch 2.4 moved this and deprecated the old spelling.
        factory = getattr(torch.amp, "GradScaler", None)
        self.inner = (
            factory("cuda", enabled=False) if factory else torch.cuda.amp.GradScaler(enabled=False)
        )

    def scale(self, loss: Tensor) -> Tensor:
        self.events.append("scale")
        return self.inner.scale(loss)

    def unscale_(self, optimizer: torch.optim.Optimizer) -> None:
        self.events.append("unscale_")
        self.inner.unscale_(optimizer)

    def step(self, optimizer: torch.optim.Optimizer) -> None:
        self.events.append("step")
        self.inner.step(optimizer)

    def update(self) -> None:
        self.events.append("update")
        self.inner.update()


def image_batch(n: int) -> list[Tensor]:
    return [torch.rand(3, 16, 16) for _ in range(n)]


def grad_norm(model: nn.Module) -> float:
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    return float(torch.linalg.vector_norm(torch.stack([grad.norm() for grad in grads])))


def recording_clip(events: list[str]) -> Callable[..., Tensor]:
    """Wraps clip_grad_norm_ so its place in the AMP call order is visible."""
    real = nn.utils.clip_grad_norm_

    def clip(params, max_norm):
        events.append("clip")
        return real(params, max_norm)

    return clip


def labelled_batch(n: int) -> tuple[tuple[Tensor, ...], tuple[dict, ...]]:
    targets = tuple(
        {"boxes": torch.tensor([[1.0, 1.0, 8.0, 8.0]]), "labels": torch.tensor([1])}
        for _ in range(n)
    )
    return tuple(image_batch(n)), targets


def test_two_epochs_reduce_the_training_loss(detector, loader):
    cfg = TrainConfig(epochs=2, lr=2e-4, seed=0)
    records = train_baseline(detector, loader, cfg)

    assert len(records) == 2
    assert records[-1].losses["total"] < records[0].losses["total"]


def test_every_detection_loss_is_recorded_under_its_own_key(detector, loader):
    cfg = TrainConfig(epochs=1, lr=1e-4, seed=0)
    record = train_baseline(detector, loader, cfg)[0]

    expected = {"loss_classifier", "loss_box_reg", "loss_objectness", "loss_rpn_box_reg"}
    assert expected <= set(record.losses)
    assert all(math.isfinite(value) for value in record.losses.values())


def test_recorded_losses_are_the_mean_of_the_per_batch_values(loader):
    script = [
        {"loss_classifier": 1.0, "loss_box_reg": 4.0},
        {"loss_classifier": 2.0, "loss_box_reg": 6.0},
        {"loss_classifier": 6.0, "loss_box_reg": 2.0},
        {"loss_classifier": 3.0, "loss_box_reg": 0.0},
    ]
    model = ScriptedLosses(script)
    record = train_baseline(model, loader, TrainConfig(epochs=1, lr=1e-3))[0]

    assert model.batches == BATCHES_PER_EPOCH
    assert record.losses["loss_classifier"] == pytest.approx(3.0)
    assert record.losses["loss_box_reg"] == pytest.approx(3.0)
    assert record.losses["total"] == pytest.approx(6.0)


def test_seconds_times_the_epoch_and_not_the_validation_callback(loader):
    def slow_val(model: nn.Module) -> dict[str, float]:
        time.sleep(0.5)
        return {"AP50": 0.25}

    cfg = TrainConfig(epochs=1, lr=1e-3)
    record = train_baseline(StubDetector(), loader, cfg, val_fn=slow_val)[0]

    assert 0.0 < record.seconds < 0.25


def test_frozen_backbone_is_handed_back_when_training_ends(detector, loader):
    """Regression: the original froze the backbone for the warm-up and never restored it."""
    reference = next(iter(detector.backbone.parameters())).detach().clone()
    frozen_during_epoch: list[bool] = []

    def note(record: EpochRecord) -> None:
        frozen_during_epoch.append(
            all(not p.requires_grad for p in detector.backbone.parameters())
        )

    cfg = TrainConfig(epochs=1, lr=1e-2, freeze_backbone_epochs=1)
    train_baseline(detector, loader, cfg, on_epoch=note)

    assert frozen_during_epoch == [True]
    assert torch.equal(next(iter(detector.backbone.parameters())).detach(), reference)
    assert all(p.requires_grad for p in detector.backbone.parameters())


def test_a_released_backbone_starts_training_again(loader):
    model = StubDetector()
    grad_state: list[bool] = []
    weights: list[torch.Tensor] = []

    def note(record: EpochRecord) -> None:
        grad_state.append(model.backbone.weight.requires_grad)
        weights.append(model.backbone.weight.detach().clone())

    cfg = TrainConfig(epochs=3, lr=1e-2, freeze_backbone_epochs=2)
    train_baseline(model, loader, cfg, on_epoch=note)

    assert grad_state == [False, False, True]
    assert torch.equal(weights[0], weights[1])
    # A one-way freeze would leave the third epoch identical to the first two.
    assert not torch.equal(weights[1], weights[2])


def test_val_fn_runs_once_per_epoch_after_the_batches(loader):
    model = StubDetector()
    batches_seen: list[int] = []

    def val_fn(validated: nn.Module) -> dict[str, float]:
        batches_seen.append(validated.batches)
        return {"AP50": 0.25}

    records = train_baseline(model, loader, TrainConfig(epochs=2, lr=1e-3), val_fn=val_fn)

    # Validation must see a finished epoch, not a half-trained model mid-epoch.
    assert batches_seen == [BATCHES_PER_EPOCH, 2 * BATCHES_PER_EPOCH]
    assert [r.val_metrics for r in records] == [{"AP50": 0.25}, {"AP50": 0.25}]


def test_on_epoch_is_called_once_per_epoch_with_increasing_epochs(loader):
    seen: list[int] = []
    cfg = TrainConfig(epochs=3, lr=1e-3)
    records = train_baseline(StubDetector(), loader, cfg, on_epoch=lambda r: seen.append(r.epoch))

    assert seen == [0, 1, 2]
    assert [r.epoch for r in records] == seen
    assert [r.val_metrics for r in records] == [None, None, None]


def test_amp_unscales_the_gradients_before_clipping_them(loader, monkeypatch):
    """A clip applied to still-scaled gradients would mean a different threshold every step."""
    events: list[str] = []
    monkeypatch.setattr(train_module, "_grad_scaler", lambda cfg, device: RecordingScaler(events))
    monkeypatch.setattr(nn.utils, "clip_grad_norm_", recording_clip(events))

    cfg = TrainConfig(epochs=1, lr=1e-3, amp=True, clip_grad_norm=1.0)
    train_baseline(StubDetector(), loader, cfg)

    assert events == ["scale", "unscale_", "clip", "step", "update"] * BATCHES_PER_EPOCH


def test_no_clip_norm_skips_both_the_unscale_and_the_clip(loader, monkeypatch):
    events: list[str] = []
    monkeypatch.setattr(train_module, "_grad_scaler", lambda cfg, device: RecordingScaler(events))
    monkeypatch.setattr(nn.utils, "clip_grad_norm_", recording_clip(events))

    cfg = TrainConfig(epochs=1, lr=1e-3, amp=True, clip_grad_norm=None)
    train_baseline(StubDetector(), loader, cfg)

    assert events == ["scale", "step", "update"] * BATCHES_PER_EPOCH


def test_clipping_bounds_the_norm_over_all_parameter_groups(built_optimizers):
    model = ExplodingAdaptive()
    cfg = TrainConfig(epochs=1, lr=1e-6, clip_grad_norm=1.0, disc_lr=1e-6)
    train_adaptive(model, [labelled_batch(1)], [image_batch(1)], cfg)

    # Two groups, each blowing past the threshold on its own: clipping them separately would
    # leave a global norm of sqrt(2).
    assert len(built_optimizers[0].param_groups) == 2
    assert grad_norm(model) == pytest.approx(1.0, rel=1e-4)


def test_clip_grad_norm_none_leaves_the_gradients_untouched():
    model = ExplodingAdaptive()
    cfg = TrainConfig(epochs=1, lr=1e-6, clip_grad_norm=None)
    train_adaptive(model, [labelled_batch(1)], [image_batch(1)], cfg)

    assert grad_norm(model) == pytest.approx(math.sqrt(2) * model.grad_per_weight, rel=1e-4)


def test_build_optimizer_puts_discriminator_params_in_a_second_group():
    disc = nn.Linear(4, 1)
    model = nn.ModuleDict({"backbone": nn.Linear(4, 4), "disc": disc})

    optimizer = build_optimizer(model, TrainConfig(lr=2e-4, disc_lr=5e-3), disc.parameters())

    assert len(optimizer.param_groups) == 2
    assert optimizer.param_groups[0]["lr"] == 2e-4
    assert optimizer.param_groups[1]["lr"] == 5e-3

    # A parameter listed in two groups would be updated twice per step.
    main_ids = {id(p) for p in optimizer.param_groups[0]["params"]}
    disc_ids = {id(p) for p in optimizer.param_groups[1]["params"]}
    assert main_ids.isdisjoint(disc_ids)
    assert disc_ids == {id(p) for p in disc.parameters()}


def test_build_optimizer_falls_back_to_the_shared_lr_without_disc_lr():
    disc = nn.Linear(4, 1)
    model = nn.ModuleDict({"disc": disc, "head": nn.Linear(4, 2)})
    optimizer = build_optimizer(model, TrainConfig(lr=3e-4), disc.parameters())

    assert [group["lr"] for group in optimizer.param_groups] == [3e-4, 3e-4]


def test_build_optimizer_keeps_frozen_params_so_unfreezing_still_trains():
    model = StubDetector()
    model.backbone.weight.requires_grad_(False)

    optimizer = build_optimizer(model, TrainConfig())
    listed = {id(p) for group in optimizer.param_groups for p in group["params"]}

    assert id(model.backbone.weight) in listed


def test_build_optimizer_supports_sgd_and_rejects_anything_else():
    model = nn.Linear(2, 2)
    sgd = build_optimizer(model, TrainConfig(optimizer="sgd", momentum=0.8))
    assert isinstance(sgd, torch.optim.SGD)
    assert sgd.param_groups[0]["momentum"] == 0.8

    with pytest.raises(ValueError, match="unknown optimizer"):
        build_optimizer(model, TrainConfig(optimizer="rmsprop"))


def test_train_adaptive_steps_the_shorter_loader_and_ramps_progress():
    model = StubAdaptive()
    source = [labelled_batch(2) for _ in range(3)]
    target = [image_batch(2) for _ in range(2)]

    records = train_adaptive(model, source, target, TrainConfig(epochs=2, lr=1e-3))

    # Two epochs of min(3, 2) paired steps, with progress sweeping [0, 1).
    assert len(model.progress) == 4
    assert model.progress[0] == 0.0
    assert model.progress == sorted(model.progress)
    assert model.progress[-1] == pytest.approx(0.75)
    assert [r.epoch for r in records] == [0, 1]
    assert all("loss_dann" in r.losses for r in records)


def test_train_adaptive_releases_the_backbone_after_the_warm_up():
    model = StubAdaptive()
    trainable: list[bool] = []
    source = [labelled_batch(2) for _ in range(2)]
    target = [image_batch(2) for _ in range(2)]

    cfg = TrainConfig(epochs=2, lr=1e-2, freeze_backbone_epochs=1)
    train_adaptive(
        model,
        source,
        target,
        cfg,
        on_epoch=lambda r: trainable.append(model.detector.backbone.weight.requires_grad),
    )

    assert trainable == [False, True]
    assert all(p.requires_grad for p in model.detector.backbone.parameters())


def test_train_adaptive_trains_a_real_wrapper(detector):
    torch.manual_seed(0)
    model = DomainAdaptiveDetector(detector, AdaptationConfig(), mode="dann")
    before = model.domain_disc[0].weight.detach().clone()
    images, targets = labelled_batch(2)

    cfg = TrainConfig(epochs=1, lr=1e-3)
    record = train_adaptive(model, [(images, targets)], [image_batch(2)], cfg)[0]

    assert "loss_dann" in record.losses
    assert all(math.isfinite(value) for value in record.losses.values())
    assert not torch.equal(model.domain_disc[0].weight.detach(), before)


def test_train_adaptive_gives_the_live_domain_heads_the_disc_lr(detector, built_optimizers):
    """DOMAIN_HEAD_ATTRS is looked up by name, so a rename in adaptation.py must fail here."""
    torch.manual_seed(0)
    model = DomainAdaptiveDetector(detector, AdaptationConfig(), mode="dccan")
    images, targets = labelled_batch(2)

    cfg = TrainConfig(epochs=1, lr=1e-3, disc_lr=5e-3)
    train_adaptive(model, [(images, targets)], [image_batch(2)], cfg)

    groups = built_optimizers[0].param_groups
    assert [group["lr"] for group in groups] == [1e-3, 5e-3]

    heads = (model.domain_disc, model.cdan_disc, model.proposal_head)
    adversarial = {id(p) for head in heads for p in head.parameters()}
    assert {id(p) for p in groups[1]["params"]} == adversarial
    # proxy_cls is trained by a supervised loss, so it belongs with the detector.
    assert {id(p) for p in model.proxy_cls.parameters()} <= {id(p) for p in groups[0]["params"]}


def test_checkpoint_round_trips_weights_config_and_extras(tmp_path):
    torch.manual_seed(0)
    saved = nn.Linear(4, 2)
    cfg = TrainConfig(epochs=3, lr=7e-4, optimizer="sgd")
    path = tmp_path / "runs" / "model.pt"

    save_checkpoint(path, saved, cfg, extra={"best_ap50": 0.61})
    loaded = nn.Linear(4, 2)
    meta = load_checkpoint(path, loaded)

    assert torch.equal(saved.weight, loaded.weight)
    assert meta["format"] == 2
    assert meta["config"]["lr"] == 7e-4
    assert meta["config"]["optimizer"] == "sgd"
    assert meta["extra"] == {"best_ap50": 0.61}


def test_load_checkpoint_accepts_a_bare_state_dict(tmp_path):
    """The first notebooks saved `model.state_dict()` directly; those files must still load."""
    torch.manual_seed(1)
    saved = nn.Linear(4, 2)
    path = tmp_path / "legacy.pth"
    torch.save(saved.state_dict(), path)

    loaded = nn.Linear(4, 2)
    meta = load_checkpoint(path, loaded)

    assert torch.equal(saved.weight, loaded.weight)
    assert meta["format"] == 1
    assert meta["config"] == {}


def test_load_checkpoint_accepts_a_model_only_dict(tmp_path):
    torch.manual_seed(2)
    saved = nn.Linear(4, 2)
    path = tmp_path / "model_key.pth"
    torch.save({"model": saved.state_dict(), "proxy_cls": {}}, path)

    loaded = nn.Linear(4, 2)
    meta = load_checkpoint(path, loaded)

    assert torch.equal(saved.bias, loaded.bias)
    assert meta["format"] == 1


def test_load_checkpoint_reports_the_missing_path(tmp_path):
    with pytest.raises(FileNotFoundError, match=r"absent\.pt"):
        load_checkpoint(tmp_path / "absent.pt", nn.Linear(2, 2))


def test_deterministic_flag_reaches_set_seed(loader, monkeypatch):
    """Regression: the flag existed in set_seed but TrainConfig had no way to set it.

    Two runs of the same seed diverged by more than the effects being measured because the
    configs could not turn cuDNN autotuning off. See experiments/results/README.md.
    """
    seen: list[tuple[int, bool]] = []
    monkeypatch.setattr(
        train_module, "set_seed", lambda seed, deterministic=False: seen.append((seed, deterministic))
    )

    train_baseline(StubDetector(), loader, TrainConfig(epochs=1, seed=7, num_workers=0))
    train_baseline(
        StubDetector(), loader, TrainConfig(epochs=1, seed=7, num_workers=0, deterministic=True)
    )

    assert seen == [(7, False), (7, True)]
