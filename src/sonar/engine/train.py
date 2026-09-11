"""Training loops for the plain detector and for the domain-adaptive wrapper.

Neither loop writes to disk. They return one `EpochRecord` per epoch and leave persistence,
logging and early stopping to the caller, which keeps the experiment scripts in charge of
where artefacts land.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
from torch import Tensor, nn

from sonar.models.detector import freeze_backbone
from sonar.utils.seed import set_seed

if TYPE_CHECKING:
    from sonar.models.adaptation import DomainAdaptiveDetector

# Adversarial heads the adaptive wrapper owns. They get their own learning rate because the
# discriminators converge far faster than the detector they are pushing against. `proxy_cls` is
# deliberately absent: it is trained by a plain supervised auxiliary loss, so it wants the
# detector's learning rate, not the discriminator's.
DOMAIN_HEAD_ATTRS = ("domain_disc", "cdan_disc", "proposal_head")


@dataclass
class TrainConfig:
    """Everything the loops need that is not the model or the data."""

    epochs: int = 20
    batch_size: int = 8
    lr: float = 2e-4
    weight_decay: float = 1e-4
    optimizer: str = "adamw"
    momentum: float = 0.9
    clip_grad_norm: float | None = 5.0
    amp: bool = False
    num_workers: int = 2
    seed: int = 1337
    freeze_backbone_epochs: int = 0
    disc_lr: float | None = None


@dataclass
class EpochRecord:
    """One epoch's outcome.

    `losses` holds the mean per-batch value of every key the model returned, plus a `total` key
    carrying the summed loss that was actually backwarded. `total` therefore duplicates the other
    entries: read `losses["total"]`, never `sum(losses.values())`.
    """

    epoch: int
    losses: dict[str, float]
    seconds: float
    val_metrics: dict[str, float] | None = None


def build_optimizer(
    model: nn.Module,
    cfg: TrainConfig,
    disc_params: Iterable[nn.Parameter] | None = None,
) -> torch.optim.Optimizer:
    """Build the optimizer, optionally splitting the domain heads into their own LR group.

    Frozen parameters stay in the groups: a warm-up schedule unfreezes the backbone partway
    through training, and a parameter the optimizer never heard of would then never be updated.
    """
    disc = list(disc_params or [])
    disc_ids = {id(p) for p in disc}
    main = [p for p in model.parameters() if id(p) not in disc_ids]

    groups: list[dict[str, Any]] = [{"params": main, "lr": cfg.lr}]
    if disc:
        groups.append({"params": disc, "lr": cfg.lr if cfg.disc_lr is None else cfg.disc_lr})

    name = cfg.optimizer.lower()
    if name == "adamw":
        return torch.optim.AdamW(groups, lr=cfg.lr, weight_decay=cfg.weight_decay)
    if name == "sgd":
        return torch.optim.SGD(
            groups, lr=cfg.lr, momentum=cfg.momentum, weight_decay=cfg.weight_decay
        )
    raise ValueError(f"unknown optimizer {cfg.optimizer!r}, expected 'adamw' or 'sgd'")


def train_baseline(
    model: nn.Module,
    train_loader: Iterable,
    cfg: TrainConfig,
    *,
    device: str | torch.device = "cpu",
    val_fn: Callable[[nn.Module], dict[str, float]] | None = None,
    on_epoch: Callable[[EpochRecord], None] | None = None,
) -> list[EpochRecord]:
    """Supervised training of a torchvision detector on a single labelled domain.

    Seeds the global RNG from `cfg.seed` first, so a run is reproducible from its config alone.
    That overwrites any seed the caller set, which is worth knowing if a shuffling loader was
    built beforehand.
    """
    device = torch.device(device)
    set_seed(cfg.seed)
    model.to(device)

    optimizer = build_optimizer(model, cfg)
    scaler = _grad_scaler(cfg, device)
    records: list[EpochRecord] = []

    try:
        for epoch in range(cfg.epochs):
            _apply_freeze_schedule(model, cfg, epoch)
            model.train()
            started = time.perf_counter()
            totals = _LossTotals()

            for batch in train_loader:
                images, targets = _split_batch(batch)
                images = [image.to(device) for image in images]
                targets = _targets_to_device(targets, device)

                with torch.autocast(device_type=device.type, enabled=cfg.amp):
                    losses = model(images, targets)
                    loss = sum(losses.values())

                _optimise(loss, optimizer, scaler, cfg)
                totals.add(losses, loss)

            records.append(_finish_epoch(epoch, totals, started, model, val_fn, on_epoch))
    finally:
        _release_freeze(model, cfg)

    return records


def train_adaptive(
    model: DomainAdaptiveDetector,
    source_loader: Iterable,
    target_loader: Iterable,
    cfg: TrainConfig,
    *,
    device: str | torch.device = "cpu",
    val_fn: Callable[[nn.Module], dict[str, float]] | None = None,
    on_epoch: Callable[[EpochRecord], None] | None = None,
) -> list[EpochRecord]:
    """Adversarial training: labelled source batches paired with unlabelled target batches.

    The two loaders are stepped in lockstep for `min(len(source), len(target))` batches, so the
    shorter domain sets the epoch length and neither domain is silently oversampled. Both loaders
    must therefore be sized; an unsized (iterable-style) loader raises TypeError. Like
    `train_baseline`, this reseeds the global RNG from `cfg.seed`.
    """
    device = torch.device(device)
    set_seed(cfg.seed)
    model.to(device)

    optimizer = build_optimizer(model, cfg, _domain_parameters(model))
    scaler = _grad_scaler(cfg, device)

    n_steps = min(_loader_length(source_loader), _loader_length(target_loader))
    total_steps = max(cfg.epochs * n_steps, 1)
    global_step = 0
    records: list[EpochRecord] = []

    try:
        for epoch in range(cfg.epochs):
            _apply_freeze_schedule(model, cfg, epoch)
            model.train()
            started = time.perf_counter()
            totals = _LossTotals()

            for source_batch, target_batch in zip(source_loader, target_loader):
                model.set_progress(global_step / total_steps)

                source_images, source_targets = _split_batch(source_batch)
                target_images, _ = _split_batch(target_batch)
                source_images = [image.to(device) for image in source_images]
                target_images = [image.to(device) for image in target_images]
                source_targets = _targets_to_device(source_targets, device)

                with torch.autocast(device_type=device.type, enabled=cfg.amp):
                    losses = model(source_images, source_targets, target_images)
                    loss = sum(losses.values())

                _optimise(loss, optimizer, scaler, cfg)
                totals.add(losses, loss)
                global_step += 1

            records.append(_finish_epoch(epoch, totals, started, model, val_fn, on_epoch))
    finally:
        _release_freeze(model, cfg)

    return records


def save_checkpoint(
    path: str | Path,
    model: nn.Module,
    cfg: TrainConfig,
    extra: dict | None = None,
) -> None:
    """Write a format-2 checkpoint: weights, the config that produced them, and free-form extras."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": model.state_dict(),
        "config": asdict(cfg),
        "extra": dict(extra or {}),
        "format": 2,
    }
    torch.save(payload, path)


def load_checkpoint(path: str | Path, model: nn.Module, *, strict: bool = True) -> dict:
    """Load weights into `model` and return the checkpoint metadata.

    The original notebooks saved a bare state dict, and later a `{"model": ...}` dict alongside
    several loose head state dicts. Both still load here, reported as format 1, so old runs stay
    reproducible without a migration script.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"checkpoint not found: {path}")

    # weights_only keeps a checkpoint from executing arbitrary pickled code; everything we store
    # is tensors and plain data, and the old notebook files are bare state dicts.
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(payload, dict) and isinstance(payload.get("model"), dict):
        state = payload["model"]
        meta = payload
    else:
        state = payload
        meta = {}

    result = model.load_state_dict(state, strict=strict)
    return {
        "config": meta.get("config", {}),
        "extra": meta.get("extra", {}),
        "format": meta.get("format", 1),
        "missing_keys": list(result.missing_keys),
        "unexpected_keys": list(result.unexpected_keys),
    }


class _LossTotals:
    """Running per-key sums so an epoch can report mean per-batch losses."""

    def __init__(self) -> None:
        self.sums: dict[str, float] = {}
        self.batches = 0

    def add(self, losses: dict[str, Tensor], total: Tensor) -> None:
        self.batches += 1
        for key, value in losses.items():
            self.sums[key] = self.sums.get(key, 0.0) + float(value.detach())
        # The quantity that was backwarded, kept as its own key so a caller never has to re-sum
        # the parts; see EpochRecord for the duplication that implies.
        self.sums["total"] = self.sums.get("total", 0.0) + float(total.detach())

    def means(self) -> dict[str, float]:
        if not self.batches:
            return {}
        return {key: value / self.batches for key, value in self.sums.items()}


def _finish_epoch(
    epoch: int,
    totals: _LossTotals,
    started: float,
    model: nn.Module,
    val_fn: Callable[[nn.Module], dict[str, float]] | None,
    on_epoch: Callable[[EpochRecord], None] | None,
) -> EpochRecord:
    # Stop the clock before validation runs: `seconds` reports the training epoch, and val_fn can
    # easily cost more than the epoch it follows.
    seconds = time.perf_counter() - started
    record = EpochRecord(
        epoch=epoch,
        losses=totals.means(),
        seconds=seconds,
        val_metrics=val_fn(model) if val_fn is not None else None,
    )
    if on_epoch is not None:
        on_epoch(record)
    return record


def _optimise(
    loss: Tensor,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
    cfg: TrainConfig,
) -> None:
    optimizer.zero_grad(set_to_none=True)
    scaler.scale(loss).backward()

    if cfg.clip_grad_norm is not None:
        # Gradients are still scaled at this point under AMP, so the clip threshold would
        # otherwise mean something different on every step.
        scaler.unscale_(optimizer)
        # One norm over every group, not one per group: splitting the domain heads into their own
        # LR group must not quietly raise the bound `clip_grad_norm` sets.
        params = [p for group in optimizer.param_groups for p in group["params"]]
        nn.utils.clip_grad_norm_(params, cfg.clip_grad_norm)

    scaler.step(optimizer)
    scaler.update()


def _grad_scaler(cfg: TrainConfig, device: torch.device) -> torch.cuda.amp.GradScaler:
    """Loss scaling only buys anything for fp16 on CUDA; CPU autocast runs in bfloat16."""
    return torch.cuda.amp.GradScaler(enabled=cfg.amp and device.type == "cuda")


def _split_batch(batch: Any) -> tuple[list[Tensor], list[dict]]:
    """Accept either a `(images, targets)` pair from `collate_detection` or bare images."""
    if isinstance(batch, Sequence) and len(batch) == 2 and not isinstance(batch[0], Tensor):
        images, targets = batch
        return list(images), list(targets)
    return list(batch), []


def _targets_to_device(targets: Sequence[dict], device: torch.device) -> list[dict]:
    return [
        {key: value.to(device) if isinstance(value, Tensor) else value for key, value in t.items()}
        for t in targets
    ]


def _loader_length(loader: Iterable) -> int:
    if not hasattr(loader, "__len__"):
        raise TypeError("train_adaptive needs sized loaders to pair the two domains")
    return len(loader)  # type: ignore[arg-type]


def _domain_parameters(model: nn.Module) -> list[nn.Parameter]:
    """Parameters of the adversarial heads, which take `disc_lr` when one is configured."""
    params: list[nn.Parameter] = []
    for attr in DOMAIN_HEAD_ATTRS:
        head = getattr(model, attr, None)
        if isinstance(head, nn.Module):
            params.extend(head.parameters())
    return params


def _detector_of(model: nn.Module) -> nn.Module:
    """The wrapped torchvision detector, or the model itself when it already is one."""
    return getattr(model, "detector", model)


def _apply_freeze_schedule(model: nn.Module, cfg: TrainConfig, epoch: int) -> None:
    if cfg.freeze_backbone_epochs <= 0:
        return
    freeze_backbone(_detector_of(model), epoch < cfg.freeze_backbone_epochs)


def _release_freeze(model: nn.Module, cfg: TrainConfig) -> None:
    """Hand the backbone back even if training stopped early; the original never did."""
    if cfg.freeze_backbone_epochs > 0:
        freeze_backbone(_detector_of(model), False)
