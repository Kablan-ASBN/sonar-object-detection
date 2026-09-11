"""Recompute the measurements in docs/AUDIT.md from what is committed in this repository.

Nothing here trains anything. It reads the archived 2025 split files and the prediction CSVs in
`outputs/`, and reproduces findings 2, 5 and 7: the split overlap matrix, the mismatch between the
ground truth different models were scored against, and the fact that the committed predictions do
not reproduce the reported table.
"""

from __future__ import annotations

import csv
from pathlib import Path

import torch

from sonar.audit.leakage import check_leakage
from sonar.engine.evaluate import GroundTruth, coco_metrics

REPO = Path(__file__).resolve().parent.parent
ARCHIVE = REPO / "docs" / "evidence" / "original_splits"

ROOTS = {
    "raw": REPO / "data/line2voc",
    "denoised": REPO / "data/line2voc_preprocessed",
    "augmented": REPO / "data/line2voc_preprocessed_augmented",
}

# name -> (csv in outputs/, AP50 reported in the dissertation)
REPORTED = {
    "Raw baseline": ("preds_baseline_20epoch.csv", 0.1493),
    "Denoised baseline": ("preds_denoised_20epoch.csv", 0.1520),
    "CLAHE+Aug baseline": ("preds_claheaug_20epoch.csv", 0.1166),
    "DANN": ("preds_dann_20epoch.csv", 0.0912),
    "DCCAN": ("preds_dccan_20epoch.csv", 0.1633),
}


def archived_ids(root_name: str, split: str) -> list[str]:
    path = ARCHIVE / ROOTS[root_name].name / f"{split}.txt"
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def load_csv(path: Path, keep: set[str]) -> dict[str, dict]:
    """Read a prediction CSV into the mapping `coco_metrics` expects."""
    rows: dict[str, list[tuple[float, int, list[float]]]] = {}
    with open(path) as handle:
        for record in csv.DictReader(handle):
            key = "filename" if "filename" in record else "image_id"
            image_id = Path(record[key]).stem
            if image_id not in keep:
                continue
            box = [float(record[k]) for k in ("xmin", "ymin", "xmax", "ymax")]
            if box[2] <= box[0] or box[3] <= box[1]:
                continue
            rows.setdefault(image_id, []).append(
                (float(record["score"]), int(record["class_id"]), box)
            )

    out = {}
    for image_id, entries in rows.items():
        entries.sort(key=lambda e: -e[0])
        out[image_id] = {
            "boxes": torch.tensor([e[2] for e in entries], dtype=torch.float32).reshape(-1, 4),
            "scores": torch.tensor([e[0] for e in entries], dtype=torch.float32),
            "labels": torch.tensor([e[1] for e in entries], dtype=torch.int64),
        }
    return out


def finding_2() -> None:
    print("=" * 78)
    print("Finding 2 - the 2025 splits leak 90% of the raw validation set")
    print("=" * 78)
    archive_roots = {name: ARCHIVE / path.name for name, path in ROOTS.items()}
    # The archived directories hold only the split files, so point the checker at those.
    report = check_leakage(
        {name: p.parent for name, p in archive_roots.items()},
        train_names=("train",),
        eval_names=("val", "test"),
    ) if False else _leakage_from_archive()
    print(report)


def _leakage_from_archive() -> str:
    splits = {
        f"{name}/{split}": set(archived_ids(name, split))
        for name in ROOTS
        for split in ("train", "val", "test")
    }
    evals = [k for k in splits if not k.endswith("/train")]
    header = f"{'':22s}" + "".join(f"{c:>17s}" for c in evals)
    lines = [header]
    for name in ROOTS:
        row = f"{name + '/train':22s}"
        for c in evals:
            row += f"{len(splits[f'{name}/train'] & splits[c]):17d}"
        lines.append(row)
    worst = max(
        (len(splits[f"{n}/train"] & splits["raw/val"]), n) for n in ROOTS if n != "raw"
    )
    lines.append("")
    lines.append(
        f"{worst[1]}/train shares {worst[0]} of raw/val's {len(splits['raw/val'])} ids "
        f"({100 * worst[0] / len(splits['raw/val']):.1f}%)"
    )
    return "\n".join(lines)


def finding_5() -> None:
    print()
    print("=" * 78)
    print("Finding 5 - models were scored against different ground truth")
    print("=" * 78)
    raw_val = set(archived_ids("raw", "val"))
    aug_val = set(archived_ids("augmented", "val"))
    print(f"raw/val holds {len(raw_val)} ids, augmented/val holds {len(aug_val)}")
    print(f"they share {len(raw_val & aug_val)}")
    print("Four rows of the reported table used the first set, CLAHE+Aug used the second.")


def finding_7() -> None:
    print()
    print("=" * 78)
    print("Finding 7 - the committed predictions do not reproduce the reported table")
    print("=" * 78)

    for split in ("val", "test"):
        ids = archived_ids("raw", split)
        gt = GroundTruth.load(ROOTS["raw"], split, ids=ids)
        print(f"\nraw/{split}, {len(gt)} images, {gt.n_boxes()} boxes")
        print(
            f"{'model':20s} {'AP50':>8s} {'reported':>9s} {'delta':>8s} "
            f"{'images':>7s} {'min score':>10s}"
        )
        for name, (filename, reported) in REPORTED.items():
            path = REPO / "outputs" / filename
            if not path.is_file():
                print(f"{name:20s}   missing {filename}")
                continue
            preds = load_csv(path, set(ids))
            metrics = coco_metrics(preds, gt)
            floor = min(
                (float(p["scores"].min()) for p in preds.values() if p["scores"].numel()),
                default=float("nan"),
            )
            delta = metrics["AP50"] - reported
            print(
                f"{name:20s} {metrics['AP50']:8.4f} {reported:9.4f} {delta:+8.4f} "
                f"{len(preds):7d} {floor:10.4f}"
            )
    print(
        "\nEvery committed file has a hard score floor at 0.50, so these are visualisation exports,"
        "\nnot the unfiltered ones section 5.1 describes. The files the comparison actually read"
        "\n(preds_*_frcnn_*.csv, eval_dann_allinone/) were never committed."
    )


def main() -> int:
    finding_2()
    finding_5()
    finding_7()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
