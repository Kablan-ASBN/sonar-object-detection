"""Command line front end. Installed as `sonar`, also runnable as `python -m sonar.cli`."""

from __future__ import annotations

import argparse
import csv
import sys
from collections.abc import Callable, Collection, Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # annotations only, so the handlers keep their lazy imports
    from torch import nn
    from torch.utils.data import DataLoader
    from torchvision.models.detection import FasterRCNN

    from sonar.engine.evaluate import GroundTruth
    from sonar.engine.train import EpochRecord, TrainConfig

COMMANDS = ("convert", "split", "preprocess", "audit", "train", "eval", "predict")

# Sections a training config may carry, and the keys the `data` section accepts. A key nothing
# reads is rejected rather than ignored: a misspelled `val_root` used to turn validation off and
# a misspelled `hflip` used to fall back to 0.5, both without a word on stdout.
CONFIG_SECTIONS = frozenset({"name", "mode", "model", "data", "train", "adaptation"})
DATA_KEYS = frozenset(
    {"root", "source_root", "target_root", "train_split", "val_root", "val_split", "hflip"}
)


def _named_path(value: str) -> tuple[str, Path]:
    """Parse a `NAME=PATH` option value into its two halves."""
    name, separator, path = value.partition("=")
    if not separator or not name or not path:
        raise argparse.ArgumentTypeError(f"expected NAME=PATH, got {value!r}")
    return name, Path(path)


def build_parser() -> argparse.ArgumentParser:
    """The full `sonar` parser, one subparser per subcommand."""
    parser = argparse.ArgumentParser(
        prog="sonar",
        description="Sonar object and shadow detection: data prep, audits, training, scoring.",
    )
    # The subcommand is deliberately optional so a bare `sonar` prints the whole help instead of
    # a one-line usage error. `main` turns the missing command into exit code 2 itself. Every
    # subparser overwrites `func`, and `parser` names whichever help to print when it does not.
    parser.set_defaults(func=None, parser=parser)
    sub = parser.add_subparsers(dest="command", metavar="{" + ",".join(COMMANDS) + "}")

    _add_convert(sub)
    _add_split(sub)
    _add_preprocess(sub)
    _add_audit(sub)
    _add_train(sub)
    _add_eval(sub)
    _add_predict(sub)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run one subcommand and return its exit code."""
    parser = build_parser()
    args = parser.parse_args(None if argv is None else list(argv))
    if args.func is None:
        args.parser.print_help()
        return 2
    try:
        return args.func(args)
    except (OSError, ValueError) as error:
        # A missing file, a filtered export or a bad config is the caller's mistake, not a crash.
        # The command still owes the shell an exit code, so the message goes to stderr and the
        # traceback stays out of the way; anything else propagates as the bug it is.
        print(f"sonar {args.command}: {error}", file=sys.stderr)
        return 1


def _add_convert(sub: argparse._SubParsersAction) -> None:
    cmd = sub.add_parser("convert", help="build a VOC dataset root from a YOLO images/labels pair")
    cmd.add_argument("--images", type=Path, required=True, metavar="DIR")
    cmd.add_argument("--labels", type=Path, required=True, metavar="DIR")
    cmd.add_argument("--out", type=Path, required=True, metavar="ROOT")
    cmd.add_argument(
        "--keep-empty",
        action="store_true",
        help="keep background-only tiles as annotations with no objects instead of dropping them",
    )
    cmd.add_argument("--workers", type=int, default=8)
    cmd.set_defaults(func=_convert)


def _add_split(sub: argparse._SubParsersAction) -> None:
    cmd = sub.add_parser("split", help="generate one seeded split and write it to every root")
    cmd.add_argument("--root", type=Path, required=True, metavar="ROOT")
    cmd.add_argument(
        "--also-root",
        type=Path,
        action="append",
        default=[],
        metavar="ROOT",
        help="another root that must carry the identical split; repeatable",
    )
    cmd.add_argument("--seed", type=int, default=42)
    cmd.add_argument(
        "--ratios",
        type=float,
        nargs=3,
        default=[0.8, 0.1, 0.1],
        metavar=("TRAIN", "VAL", "TEST"),
    )
    cmd.set_defaults(func=_split)


def _add_preprocess(sub: argparse._SubParsersAction) -> None:
    cmd = sub.add_parser("preprocess", help="write a denoised or augmented copy of a dataset root")
    cmd.add_argument("--src", type=Path, required=True, metavar="ROOT")
    cmd.add_argument("--dst", type=Path, required=True, metavar="ROOT")
    cmd.add_argument("--mode", choices=("denoised", "clahe_augmented"), required=True)
    cmd.add_argument("--seed", type=int, default=42)
    cmd.set_defaults(func=_preprocess)


def _add_audit(sub: argparse._SubParsersAction) -> None:
    audit = sub.add_parser("audit", help="check a dataset before it reaches a model")
    audit.set_defaults(parser=audit)
    kinds = audit.add_subparsers(dest="audit_command", metavar="{leakage,annotations}")

    leakage = kinds.add_parser(
        "leakage", help="exit 1 when a training split intersects an evaluation split"
    )
    leakage.add_argument(
        "--root",
        type=_named_path,
        action="append",
        required=True,
        metavar="NAME=PATH",
        help="dataset root to include in the matrix; repeatable",
    )
    leakage.set_defaults(func=_audit_leakage)

    annotations = kinds.add_parser(
        "annotations", help="report degenerate, out-of-bounds and unknown-class boxes"
    )
    annotations.add_argument("--root", type=Path, required=True, metavar="PATH")
    annotations.set_defaults(func=_audit_annotations)


def _add_train(sub: argparse._SubParsersAction) -> None:
    cmd = sub.add_parser("train", help="train a baseline or an adaptation model from a YAML config")
    cmd.add_argument("--config", type=Path, required=True, metavar="FILE")
    cmd.add_argument("--out", type=Path, default=None, metavar="FILE", help="checkpoint path")
    cmd.add_argument("--device", default="cpu")
    cmd.set_defaults(func=_train)


def _add_eval(sub: argparse._SubParsersAction) -> None:
    cmd = sub.add_parser("eval", help="score prediction files against one ground truth")
    cmd.add_argument("--gt-root", type=Path, required=True, metavar="PATH")
    cmd.add_argument("--split", default="test")
    cmd.add_argument(
        "--preds",
        type=_named_path,
        action="append",
        required=True,
        metavar="NAME=FILE",
        help="prediction CSV to score under this name; repeatable",
    )
    cmd.add_argument("--froc", type=Path, default=None, metavar="OUT.csv")
    cmd.add_argument(
        "--allow-filtered",
        action="store_true",
        help="score a file whose sidecar records a score floor, or has no sidecar at all, "
        "accepting that its AP sits on a truncated curve",
    )
    cmd.set_defaults(func=_eval)


def _add_predict(sub: argparse._SubParsersAction) -> None:
    cmd = sub.add_parser("predict", help="run a checkpoint over a split and export a CSV")
    cmd.add_argument("--checkpoint", type=Path, required=True, metavar="FILE")
    cmd.add_argument("--root", type=Path, required=True, metavar="PATH")
    cmd.add_argument("--split", default="test")
    cmd.add_argument("--out", type=Path, required=True, metavar="FILE")
    cmd.add_argument("--mode", choices=("raw", "vis"), default="raw")
    cmd.add_argument(
        "--backbone",
        choices=("resnet50", "mobilenet"),
        default="resnet50",
        help="must match the checkpoint; weights are loaded into a freshly built detector",
    )
    cmd.add_argument("--num-classes", type=int, default=3)
    cmd.add_argument("--batch-size", type=int, default=4)
    cmd.add_argument("--device", default="cpu")
    cmd.set_defaults(func=_predict)


# Each handler imports what it needs when it runs: torch and torchvision cost seconds to import,
# and `sonar --help`, `sonar split` and both audits never touch them.


def _convert(args: argparse.Namespace) -> int:
    from sonar.data.convert import convert_yolo_to_voc

    stats = convert_yolo_to_voc(
        args.images,
        args.labels,
        args.out,
        keep_empty=args.keep_empty,
        workers=args.workers,
    )
    print(stats)
    print(f"wrote {args.out}")
    return 0


def _split(args: argparse.Namespace) -> int:
    from sonar.data.splits import propagate_split, stratified_split

    roots = [args.root, *args.also_root]
    split = stratified_split(args.root, seed=args.seed, ratios=tuple(args.ratios))

    # One split generated once and copied to every root. Roots that each shuffle their own ids
    # are what put 90% of the validation set into the training set in the original project.
    propagate_split(split, roots)

    sizes = ", ".join(f"{name} {len(ids)}" for name, ids in split.as_dict().items())
    print(f"seed {args.seed}, ratios {tuple(args.ratios)}: {sizes}")
    for root in roots:
        print(f"wrote {Path(root) / 'ImageSets' / 'Main'}")
    return 0


def _preprocess(args: argparse.Namespace) -> int:
    from sonar.data.preprocess import build_variant

    written = build_variant(args.src, args.dst, mode=args.mode, seed=args.seed)
    print(f"{args.mode}: wrote {written} images to {args.dst}")
    return 0


def _audit_leakage(args: argparse.Namespace) -> int:
    from sonar.audit.leakage import check_leakage

    report = check_leakage(dict(args.root))
    print(report.format())
    if not report.has_leakage:
        print("clean: no training split intersects any evaluation split")
        return 0

    shared = sum(overlap.count for overlap in report.overlaps)
    print(f"leakage: {shared} shared id(s) across {len(report.overlaps)} split pair(s)")
    return 1


def _audit_annotations(args: argparse.Namespace) -> int:
    from sonar.audit.annotations import audit_annotations

    report = audit_annotations(args.root)
    print(report.format())
    # Degenerate and out-of-bounds boxes are dropped by the loader rather than fatal, so this
    # report is advisory; only the leakage check gates CI.
    return 0


def _train(args: argparse.Namespace) -> int:
    import inspect

    import yaml

    from sonar.engine.train import TrainConfig, save_checkpoint, train_adaptive, train_baseline
    from sonar.models.detector import build_detector
    from sonar.utils.seed import set_seed

    config = yaml.safe_load(Path(args.config).read_text()) or {}
    if not isinstance(config, Mapping):
        raise ValueError(f"{args.config}: expected a YAML mapping of config sections")
    _reject_unknown("top-level", config, CONFIG_SECTIONS)

    data = dict(config.get("data") or {})
    _reject_unknown("data", data, DATA_KEYS)
    train_section = dict(config.get("train") or {})
    _reject_unknown("train", train_section, TrainConfig.__dataclass_fields__)
    cfg = TrainConfig(**train_section)
    set_seed(cfg.seed)

    model_section = dict(config.get("model") or {})
    _reject_unknown("model", model_section, inspect.signature(build_detector).parameters)
    detector = build_detector(model_section.pop("num_classes", 3), **model_section)

    mode = config.get("mode", "baseline")
    split = data.get("train_split", "train")
    hflip = data.get("hflip", 0.5)
    val_fn = _validation_fn(data, args.device)

    model: nn.Module = detector
    if mode == "baseline":
        loader = _detection_loader(_require(data, "root"), split, cfg, hflip=hflip)
        records = train_baseline(
            model, loader, cfg, device=args.device, val_fn=val_fn, on_epoch=_print_epoch
        )
    else:
        from sonar.models.adaptation import AdaptationConfig, DomainAdaptiveDetector

        adaptation = dict(config.get("adaptation") or {})
        _reject_unknown("adaptation", adaptation, AdaptationConfig.__dataclass_fields__)
        model = DomainAdaptiveDetector(detector, AdaptationConfig(**adaptation), mode=mode)
        source = _detection_loader(_require(data, "source_root"), split, cfg, hflip=hflip)
        target = _detection_loader(_require(data, "target_root"), split, cfg, hflip=hflip)
        records = train_adaptive(
            model, source, target, cfg, device=args.device, val_fn=val_fn, on_epoch=_print_epoch
        )

    out = args.out or Path("outputs/checkpoints") / f"{config.get('name', mode)}.pt"
    save_checkpoint(out, model, cfg, extra={"mode": mode, "config": config, "epochs": len(records)})
    print(f"wrote {out}")
    return 0


def _eval(args: argparse.Namespace) -> int:
    from sonar.engine.evaluate import GroundTruth, compare
    from sonar.engine.predict import read_predictions

    gt = GroundTruth.load(args.gt_root, args.split)
    # Every model is read against this one ground truth, and by default every file has to prove
    # through its sidecar that it was exported unfiltered.
    models = {
        name: read_predictions(path, require_raw=not args.allow_filtered)
        for name, path in args.preds
    }

    table = compare(models, gt)
    summary = f"root={gt.root} split={args.split}, {len(gt)} images, {gt.n_boxes()} boxes"
    print(f"ground truth: {summary}")
    print(table.to_string(float_format=lambda value: f"{value:.4f}"))

    if args.froc is not None:
        _write_froc(models, gt, args.froc)
        print(f"wrote {args.froc}")
    return 0


def _predict(args: argparse.Namespace) -> int:
    from sonar.data.voc import VOCDetection
    from sonar.engine.predict import RAW, VIS, predict_dataset, write_predictions

    cfg = VIS if args.mode == "vis" else RAW
    detector = _load_detector(args.checkpoint, args.backbone, args.num_classes)
    dataset = VOCDetection(args.root, args.split)
    predictions = predict_dataset(
        detector, dataset, device=args.device, batch_size=args.batch_size, cfg=cfg
    )

    # The sidecar carries the config that produced the file, so `sonar eval` can refuse a vis
    # export instead of quoting AP from a truncated curve.
    write_predictions(predictions, args.out, cfg=cfg)
    detections = sum(int(entry["scores"].numel()) for entry in predictions.values())
    print(f"{len(predictions)} images, {detections} detections, {args.mode} postprocessing")
    print(f"wrote {args.out} and {args.out}.meta.json")
    return 0


def _reject_unknown(what: str, section: Mapping[str, Any], allowed: Collection[str]) -> None:
    """Refuse a config section carrying a key nothing reads, so a typo cannot pass silently."""
    unknown = sorted(set(section) - set(allowed))
    if unknown:
        raise ValueError(f"unknown key(s) in the {what} config section: {', '.join(unknown)}")


def _require(data: Mapping[str, Any], key: str) -> Any:
    """One required key of the `data` section, named in the error when the config omits it."""
    if key not in data:
        raise ValueError(f"the data config section needs a {key!r} key")
    return data[key]


def _detection_loader(
    root: str | Path, split: str, cfg: TrainConfig, *, hflip: float | None
) -> DataLoader:
    """A detection DataLoader whose shuffling and worker seeds follow `cfg.seed`."""
    import torch
    from torch.utils.data import DataLoader

    from sonar.data.transforms import eval_transform, train_transform
    from sonar.data.voc import VOCDetection, collate_detection
    from sonar.utils.seed import seed_worker

    transform = eval_transform() if hflip is None else train_transform(hflip)
    generator = torch.Generator()
    generator.manual_seed(cfg.seed)
    return DataLoader(
        VOCDetection(root, split, transforms=transform),
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        collate_fn=collate_detection,
        worker_init_fn=seed_worker,
        generator=generator,
    )


def _validation_fn(
    data: Mapping[str, Any], device: str
) -> Callable[[nn.Module], dict[str, float]] | None:
    """Per-epoch scoring against one fixed root and split, or None when the config names none."""
    if "val_root" not in data:
        return None

    from sonar.engine.evaluate import GroundTruth, evaluate_model

    gt = GroundTruth.load(data["val_root"], data.get("val_split", "test"))

    def run(model: nn.Module) -> dict[str, float]:
        # The adaptive wrapper's forward signature takes two domains; the detector inside it is
        # the thing that can be scored on its own.
        return evaluate_model(getattr(model, "detector", model), gt, device=device)

    return run


def _load_detector(checkpoint: str | Path, backbone: str, num_classes: int) -> FasterRCNN:
    """Restore a detector from a baseline or an adaptation checkpoint."""
    import torch

    from sonar.models.detector import build_detector

    path = Path(checkpoint)
    if not path.is_file():
        raise FileNotFoundError(f"checkpoint not found: {path}")

    payload = torch.load(path, map_location="cpu")
    state = payload["model"] if isinstance(payload.get("model"), dict) else payload
    # An adaptation run saves the whole wrapper, so the detector weights arrive under a prefix
    # alongside domain heads that no plain detector has.
    if any(key.startswith("detector.") for key in state):
        state = {k[len("detector.") :]: v for k, v in state.items() if k.startswith("detector.")}

    detector = build_detector(num_classes, backbone=backbone, pretrained=False)
    detector.load_state_dict(state)
    return detector


def _write_froc(models: Mapping[str, dict], gt: GroundTruth, path: str | Path) -> None:
    """Write one FROC curve per model as long-form `model,fppi,recall` rows."""
    from sonar.engine.evaluate import froc_curve

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["model", "fppi", "recall"])
        for name, predictions in models.items():
            fppi, recall = froc_curve(predictions, gt)
            writer.writerows(
                [name, float(f), float(r)] for f, r in zip(fppi.tolist(), recall.tolist())
            )


def _print_epoch(record: EpochRecord) -> None:
    losses = " ".join(f"{key}={value:.4f}" for key, value in sorted(record.losses.items()))
    line = f"epoch {record.epoch + 1}: {losses} ({record.seconds:.1f}s)"
    if record.val_metrics:
        line += " | " + " ".join(f"{k}={v:.4f}" for k, v in sorted(record.val_metrics.items()))
    print(line)


if __name__ == "__main__":
    raise SystemExit(main())
