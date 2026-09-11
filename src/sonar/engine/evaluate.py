"""Detection metrics, always scored against one explicit ground truth."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from torchvision.ops import box_iou

from sonar.data.voc import CLASS_NAMES, VOCDetection, collate_detection, parse_annotation

if TYPE_CHECKING:  # pandas is only needed by `compare`, which imports it lazily
    import pandas


def _annotation_size(path: Path) -> tuple[int, int]:
    """Image width and height as recorded in a VOC annotation."""
    size = ET.parse(path).getroot().find("size")
    width = None if size is None else size.find("width")
    height = None if size is None else size.find("height")
    if width is None or height is None or width.text is None or height.text is None:
        raise ValueError(f"{path}: <size> must give both <width> and <height>")
    return int(float(width.text)), int(float(height.text))


@dataclass(frozen=True)
class GroundTruth:
    """Ground truth for exactly one dataset root and one split.

    Everything downstream takes this object rather than a root path, so a table of results cannot
    quietly compare models that were scored against different data. That is the defect which made
    the original comparison meaningless.
    """

    root: Path
    split: str
    ids: tuple[str, ...]
    boxes: dict[str, Tensor]
    labels: dict[str, Tensor]
    sizes: dict[str, tuple[int, int]]

    @classmethod
    def load(
        cls, root: str | Path, split: str, ids: Sequence[str] | None = None
    ) -> GroundTruth:
        """Load one split. `ids` overrides the split file, to score against an archived split."""
        root = Path(root)
        # Going through the dataset keeps split reading and annotation parsing in one place.
        ids = tuple(VOCDetection(root, split, ids=ids).ids)
        boxes: dict[str, Tensor] = {}
        labels: dict[str, Tensor] = {}
        sizes: dict[str, tuple[int, int]] = {}
        for image_id in ids:
            path = root / "Annotations" / f"{image_id}.xml"
            raw_boxes, raw_labels = parse_annotation(path)
            boxes[image_id] = (
                torch.tensor(raw_boxes, dtype=torch.float32)
                if raw_boxes
                else torch.zeros((0, 4), dtype=torch.float32)
            )
            labels[image_id] = (
                torch.tensor(raw_labels, dtype=torch.int64)
                if raw_labels
                else torch.zeros((0,), dtype=torch.int64)
            )
            sizes[image_id] = _annotation_size(path)
        return cls(root=root, split=split, ids=ids, boxes=boxes, labels=labels, sizes=sizes)

    def __len__(self) -> int:
        return len(self.ids)

    def n_boxes(self) -> int:
        return int(sum(int(t.numel()) for t in self.labels.values()))

    def targets(self) -> list[dict[str, Tensor]]:
        """Per-image targets in `ids` order, in torchmetrics' input format."""
        return [{"boxes": self.boxes[i], "labels": self.labels[i]} for i in self.ids]


def _as_prediction(entry: Mapping[str, Any] | None) -> dict[str, Tensor]:
    """One prediction as float32 boxes/scores and int64 labels; `None` means no detections."""
    if entry is None:
        return {
            "boxes": torch.zeros((0, 4), dtype=torch.float32),
            "scores": torch.zeros((0,), dtype=torch.float32),
            "labels": torch.zeros((0,), dtype=torch.int64),
        }
    missing = {"boxes", "scores", "labels"} - set(entry)
    if missing:
        raise ValueError(f"Prediction is missing {sorted(missing)}")
    boxes = torch.as_tensor(entry["boxes"], dtype=torch.float32).reshape(-1, 4).detach().cpu()
    scores = torch.as_tensor(entry["scores"], dtype=torch.float32).reshape(-1).detach().cpu()
    labels = torch.as_tensor(entry["labels"], dtype=torch.int64).reshape(-1).detach().cpu()
    if not (len(boxes) == len(scores) == len(labels)):
        raise ValueError(
            f"Prediction arrays disagree: {len(boxes)} boxes, {len(scores)} scores, "
            f"{len(labels)} labels"
        )
    return {"boxes": boxes, "scores": scores, "labels": labels}


def _align(predictions: Mapping[str, dict], gt: GroundTruth) -> list[dict[str, Tensor]]:
    """Predictions in `gt.ids` order; an id the ground truth does not contain is an error."""
    unexpected = sorted(set(predictions) - set(gt.ids))
    if unexpected:
        raise ValueError(
            f"Predictions cover {len(unexpected)} id(s) outside {gt.root.name}/{gt.split}, "
            f"e.g. {unexpected[:3]}. Score against the ground truth the predictions came from."
        )
    # A missing id is a model that found nothing there, not an image to drop from the denominator.
    return [_as_prediction(predictions.get(i)) for i in gt.ids]


# Axes of pycocotools' precision array, which is [IoU, recall, class, area, detection cap].
_AREA_ALL = 0
_IOU_50, _IOU_75 = 0, 5  # rows of the standard 0.50:0.05:0.95 sweep


def _mean_over_evaluated(values: Tensor) -> float:
    """Mean of one slice of a pycocotools result, skipping the -1 cells it never evaluated."""
    evaluated = values[values > -1]
    return float(evaluated.mean()) if evaluated.numel() else -1.0


def _clamp(value: float) -> float:
    """An unevaluated slice reads as zero, which for a class in the ground truth is what it is.

    Every slice `coco_metrics` reads was measured against ground truth that exists, so the only
    way one comes back empty is a model that detected nothing: a genuine zero. The one case where
    zero would lie, a class the ground truth never mentions, is kept separate as NaN.
    """
    return max(float(value), 0.0)


def coco_metrics(
    predictions: Mapping[str, dict],
    gt: GroundTruth,
    *,
    max_detections: int = 100,
) -> dict[str, float]:
    """COCO-style AP/AR for one set of predictions against one ground truth.

    An id the ground truth does not hold is an error; an id it holds that the predictions omit is
    scored as a model that found nothing there. Every number is computed at `max_detections`
    detections per image and per class, and the recall key carries that cap, so the default
    reports `mAR100`. A class absent from the ground truth scores NaN rather than 0.0, which would
    read as a model failure rather than as nothing to measure.
    """
    if max_detections < 1:
        raise ValueError(f"max_detections must be at least 1, got {max_detections}")
    if not gt.ids:
        raise ValueError(f"{gt.root.name}/{gt.split} holds no images to score against")

    preds = _align(predictions, gt)
    targets = gt.targets()
    # torchmetrics takes mAP from pycocotools' stats[0], which pycocotools summarises with a
    # hard-coded cap of 100 detections: at any other cap that slice is empty and the score arrives
    # as -1. Averaging the arrays here keeps every number at the cap the caller asked for.
    caps = sorted([1, 10, max_detections])
    cap = caps.index(max_detections)

    metric = MeanAveragePrecision(
        box_format="xyxy",
        max_detection_thresholds=caps,
        extended_summary=True,
    )
    metric.update(preds, targets)
    summary = metric.compute()
    precision = summary["precision"][:, :, :, _AREA_ALL, cap]

    per_class = MeanAveragePrecision(
        box_format="xyxy",
        iou_thresholds=[0.5],
        max_detection_thresholds=caps,
        extended_summary=True,
    )
    per_class.update(preds, targets)
    class_summary = per_class.compute()
    class_precision = class_summary["precision"][:, :, :, _AREA_ALL, cap]
    ap50_by_class = {
        int(label): _mean_over_evaluated(class_precision[:, :, column])
        for column, label in enumerate(class_summary["classes"].reshape(-1).tolist())
    }

    present = {int(label) for t in targets for label in t["labels"].tolist()}
    metrics = {
        "mAP": _clamp(_mean_over_evaluated(precision)),
        "AP50": _clamp(_mean_over_evaluated(precision[_IOU_50])),
        "AP75": _clamp(_mean_over_evaluated(precision[_IOU_75])),
        f"mAR{max_detections}": _clamp(
            _mean_over_evaluated(summary["recall"][:, :, _AREA_ALL, cap])
        ),
    }
    for label, name in CLASS_NAMES.items():
        # A class with no ground truth has no defined AP; reporting 0.0 would read as a failure.
        metrics[f"AP50_{name}"] = (
            _clamp(ap50_by_class.get(label, -1.0)) if label in present else float("nan")
        )
    return metrics


def _greedy_matches(
    prediction: Mapping[str, Tensor],
    gt_boxes: Tensor,
    gt_labels: Tensor,
    iou_thresh: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Score and hit/miss flag for every detection in one image, highest score first."""
    scores = prediction["scores"]
    order = torch.argsort(scores, descending=True)
    boxes = prediction["boxes"][order]
    labels = prediction["labels"][order]
    hits = np.zeros(len(order), dtype=bool)
    if len(order) == 0 or len(gt_boxes) == 0:
        return scores[order].numpy(), hits

    ious = box_iou(boxes, gt_boxes)
    taken = torch.zeros(len(gt_boxes), dtype=torch.bool)
    for row in range(len(order)):
        # Same class only, and a box already claimed by a higher-scoring detection is spent.
        eligible = (gt_labels == labels[row]) & ~taken
        if not bool(eligible.any()):
            continue
        candidates = ious[row].clone()
        candidates[~eligible] = -1.0
        best = int(torch.argmax(candidates))
        if float(candidates[best]) >= iou_thresh:
            taken[best] = True
            hits[row] = True
    return scores[order].numpy(), hits


def _sweep(scores: np.ndarray, limit: int = 200) -> np.ndarray:
    """Descending operating points taken from the observed scores themselves."""
    if scores.size == 0:
        return np.array([0.0], dtype=float)
    unique = np.unique(scores)
    if unique.size > limit:
        unique = np.unique(np.quantile(unique, np.linspace(0.0, 1.0, limit)))
    return unique[::-1]


def froc_curve(
    predictions: Mapping[str, dict],
    gt: GroundTruth,
    *,
    iou_thresh: float = 0.5,
    thresholds: Sequence[float] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """False positives per image against recall, class-aware and one detection per box.

    An IoU exactly at `iou_thresh` counts as a hit. Ids are treated as in `coco_metrics`: one
    outside the ground truth is an error, one missing from `predictions` is an image where the
    model found nothing.

    The default operating points come from the scores that are actually present. A fixed grid
    silently truncates the curve for an export that was already thresholded, which is how the
    original score-floored CSVs still looked plausible.
    """
    preds = _align(predictions, gt)
    all_scores: list[np.ndarray] = []
    all_hits: list[np.ndarray] = []
    for image_id, prediction in zip(gt.ids, preds):
        scores, hits = _greedy_matches(
            prediction, gt.boxes[image_id], gt.labels[image_id], iou_thresh
        )
        all_scores.append(scores)
        all_hits.append(hits)

    scores = np.concatenate(all_scores) if all_scores else np.zeros(0)
    hits = np.concatenate(all_hits) if all_hits else np.zeros(0, dtype=bool)
    n_images = max(len(gt), 1)
    n_boxes = max(gt.n_boxes(), 1)

    sweep = np.asarray(thresholds, dtype=float) if thresholds is not None else _sweep(scores)
    fppi = np.empty(sweep.size, dtype=float)
    recall = np.empty(sweep.size, dtype=float)
    for i, thresh in enumerate(sweep):
        kept = scores >= thresh
        true_positives = int(hits[kept].sum())
        fppi[i] = (int(kept.sum()) - true_positives) / n_images
        recall[i] = true_positives / n_boxes

    order = np.lexsort((recall, fppi))
    return fppi[order], recall[order]


def evaluate_model(
    model: Any,
    gt: GroundTruth,
    *,
    device: str | torch.device = "cpu",
    batch_size: int = 4,
    image_root: Path | None = None,
) -> dict[str, float]:
    """Run `model` over the ground truth's own images and score the result.

    `image_root` swaps in another root for the pixels, to score a preprocessed copy of the same
    tiles against these labels. It still has to be a VOC root: the dataset parses its annotations
    on the way past, even though only the images are used.
    """
    root = Path(image_root) if image_root is not None else gt.root
    dataset = VOCDetection(root, gt.split, ids=list(gt.ids))
    # Unshuffled, because results are attached to ids by position in this loader.
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_detection,
    )

    model.to(device)
    model.eval()
    predictions: dict[str, dict] = {}
    cursor = 0
    with torch.no_grad():
        for images, _ in loader:
            outputs = model([image.to(device) for image in images])
            for output in outputs:
                predictions[gt.ids[cursor]] = _as_prediction(output)
                cursor += 1
    return coco_metrics(predictions, gt)


def compare(
    models: Mapping[str, Mapping[str, dict]],
    gt: GroundTruth,
    **kw: Any,
) -> pandas.DataFrame:
    """Score several prediction sets against one ground truth, best AP50 first.

    `kw` reaches `coco_metrics` unchanged, so every row of the table is scored the same way.
    """
    import pandas as pd

    if not isinstance(gt, GroundTruth):
        raise ValueError(
            "compare() takes exactly one GroundTruth; scoring models against different ground "
            f"truths is not comparable (got {type(gt).__name__})"
        )

    rows = {name: coco_metrics(preds, gt, **kw) for name, preds in models.items()}
    table = pd.DataFrame.from_dict(rows, orient="index")
    table.index.name = "model"
    return table.sort_values("AP50", ascending=False)
