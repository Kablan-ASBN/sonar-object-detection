"""Measure the three defects from docs/AUDIT.md that each favoured the proposed model.

One training run on the denoised source domain answers two of them at once: under the 2025 splits,
archived in docs/evidence/original_splits/ and used here on purpose, the raw validation split
shares 219 of its 242 ids with that training set while the raw test split shares none. Training
twice, once with the original image-only flip and once with the box-aware flip, answers the third.

  leak inflation  = AP50(raw/val, 90% seen) - AP50(raw/test, 0% seen)
  flip cost       = AP50(box-aware) - AP50(image-only), both on raw/test
  threshold gap   = AP50(unfiltered) - AP50(score >= 0.5), same model, same split

The backbone is MobileNetV3-FPN at 320px so this runs on a laptop CPU. Absolute numbers are not
comparable to the ResNet-50 results in the dissertation; only the within-experiment deltas are.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from sonar.data.transforms import DetectionTransform, to_float_tensor
from sonar.data.voc import VOCDetection, collate_detection
from sonar.engine.evaluate import GroundTruth, coco_metrics
from sonar.engine.predict import RAW, PostprocessConfig, predict_dataset
from sonar.engine.train import TrainConfig, train_baseline
from sonar.models.detector import build_detector
from sonar.utils.seed import set_seed

REPO = Path(__file__).resolve().parent.parent
SOURCE = REPO / "data" / "line2voc_preprocessed"
TARGET = REPO / "data" / "line2voc"

FLOORED = PostprocessConfig(score_thresh=0.5)


@dataclass
class BrokenFlipTransform:
    """The original bug, reconstructed: flips the pixels and leaves the boxes where they were.

    Kept here rather than in the package so the defect stays reproducible without being
    importable by accident.
    """

    hflip_prob: float = 0.5

    def __call__(self, image, target):
        tensor = to_float_tensor(image)
        if torch.rand(1).item() < self.hflip_prob:
            tensor = torch.flip(tensor, dims=[-1])
        return tensor, dict(target)


ARCHIVE = REPO / "docs" / "evidence" / "original_splits"


def archived_ids(root: Path, split: str) -> list[str]:
    """The 2025 split for `root`, read from the archive rather than from data/.

    The live splits under data/ have been regenerated and no longer leak, so the leaky
    condition this experiment prices has to come from the archived files.
    """
    path = ARCHIVE / root.name / f"{split}.txt"
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def subset(root: Path, split: str, limit: int | None, seed: int = 0) -> list[str]:
    ids = archived_ids(root, split)
    if limit is None or limit >= len(ids):
        return ids
    generator = torch.Generator().manual_seed(seed)
    picked = torch.randperm(len(ids), generator=generator)[:limit].tolist()
    return [ids[i] for i in sorted(picked)]


def score(model, root: Path, split: str, ids, args) -> dict[str, dict[str, float]]:
    gt = GroundTruth.load(root, split, ids=ids)
    dataset = VOCDetection(root, split, ids=ids)
    preds = predict_dataset(model, dataset, device=args.device, cfg=RAW)

    floored = {
        image_id: _apply_floor(entry, FLOORED.score_thresh) for image_id, entry in preds.items()
    }
    return {
        "unfiltered": coco_metrics(preds, gt),
        "score_floor_0.5": coco_metrics(floored, gt),
    }


def _apply_floor(entry: dict, threshold: float) -> dict:
    keep = entry["scores"] >= threshold
    return {k: v[keep] for k, v in entry.items()}


def run(flip_mode: str, args) -> dict:
    set_seed(args.seed, deterministic=False)

    transform = (
        BrokenFlipTransform(0.5)
        if flip_mode == "image_only"
        else DetectionTransform(hflip_prob=0.5)
    )
    train_ids = subset(SOURCE, "train", args.n_train)
    dataset = VOCDetection(SOURCE, "train", transforms=transform, ids=train_ids)
    loader = DataLoader(
        dataset,
        batch_size=args.batch,
        shuffle=True,
        collate_fn=collate_detection,
        num_workers=args.workers,
    )

    model = build_detector(
        num_classes=3, backbone="mobilenet", min_size=args.size, max_size=args.size
    )
    cfg = TrainConfig(
        epochs=args.epochs,
        batch_size=args.batch,
        optimizer="sgd",
        lr=5e-3,
        momentum=0.9,
        weight_decay=5e-4,
        clip_grad_norm=5.0,
        seed=args.seed,
        num_workers=args.workers,
    )

    started = time.time()
    history = train_baseline(model, loader, cfg, device=args.device)
    elapsed = time.time() - started

    val_ids = subset(TARGET, "val", args.n_eval)
    test_ids = subset(TARGET, "test", args.n_eval)
    return {
        "flip_mode": flip_mode,
        "n_train": len(train_ids),
        "epochs": args.epochs,
        "train_seconds": round(elapsed, 1),
        "final_loss": history[-1].losses if history else {},
        "raw_val_leaked": score(model, TARGET, "val", val_ids, args),
        "raw_test_clean": score(model, TARGET, "test", test_ids, args),
    }


def report(results: list[dict]) -> str:
    by_mode = {r["flip_mode"]: r for r in results}
    lines = [
        "| flip during training | raw/val AP50 (90% seen) "
        "| raw/test AP50 (unseen) | leak inflation |",
        "|---|---|---|---|",
    ]
    for mode, r in by_mode.items():
        v = r["raw_val_leaked"]["unfiltered"]["AP50"]
        t = r["raw_test_clean"]["unfiltered"]["AP50"]
        label = "image only (original bug)" if mode == "image_only" else "image and boxes (fixed)"
        lines.append(f"| {label} | {v:.4f} | {t:.4f} | {v - t:+.4f} |")

    if "image_only" in by_mode and "boxes" in by_mode:
        broken = by_mode["image_only"]["raw_test_clean"]["unfiltered"]["AP50"]
        fixed = by_mode["boxes"]["raw_test_clean"]["unfiltered"]["AP50"]
        lines += ["", f"Cost of the flip bug on the clean split: {fixed - broken:+.4f} AP50"]

    lines += ["", "| model | unfiltered AP50 | score floor 0.5 | gap |", "|---|---|---|---|"]
    for mode, r in by_mode.items():
        block = r["raw_test_clean"]
        a, b = block["unfiltered"]["AP50"], block["score_floor_0.5"]["AP50"]
        lines.append(f"| {mode} | {a:.4f} | {b:.4f} | {a - b:+.4f} |")
    return "\n".join(lines)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n-train", type=int, default=500, help="source images, None for all")
    p.add_argument("--n-eval", type=int, default=None, help="evaluation images per split")
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--size", type=int, default=320)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--device", default="cpu")
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--out", type=Path, default=REPO / "experiments/results/ablation.json")
    args = p.parse_args()

    results = [run(mode, args) for mode in ("image_only", "boxes")]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2))

    table = report(results)
    (args.out.with_suffix(".md")).write_text(table + "\n")
    print("\n" + table)
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
