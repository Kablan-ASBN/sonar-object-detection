"""Detection export and the post-processing that separates metric files from picture files.

Metrics need every box the detector emitted, because a precision-recall curve can only be drawn
over the scores that were kept; figures need a handful of clean ones. The original exported a
single file floored at score 0.5 and scored it (B7), so `RAW` and `VIS` below are those two
audiences and every written file carries a sidecar naming which one it is.
"""

from __future__ import annotations

import csv
import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import torch
from torch import Tensor
from torch.utils.data import DataLoader
from torchvision.ops import batched_nms, box_iou, clip_boxes_to_image

from sonar.data.voc import CLASS_NAMES, collate_detection

CSV_COLUMNS = ("image_id", "class_id", "class_name", "score", "xmin", "ymin", "xmax", "ymax")
META_FORMAT = 1
_META_KEYS = frozenset({"format", "postprocess", "image_ids", "n_detections"})


@dataclass
class PostprocessConfig:
    """Filters `postprocess` applies to one image's detections."""

    score_thresh: float = 0.0
    nms_iou: float | None = None
    class_thresholds: dict[int, float] | None = None
    merge_iou: float | None = None
    cross_class_iou: float | None = None
    min_box_size: int = 0
    max_detections: int = 300


RAW = PostprocessConfig()
VIS = PostprocessConfig(
    score_thresh=0.5,
    nms_iou=0.5,
    merge_iou=0.6,
    cross_class_iou=0.85,
    min_box_size=8,
    max_detections=100,
)


def _active_filters(cfg: PostprocessConfig) -> tuple[str, ...]:
    """Settings in `cfg` that can drop a detection, as `name=value` strings. Empty means raw."""
    # A score floor is only the most visible way to lose detections. Per-class floors, either
    # suppression pass and the size and count caps all shorten the curve just as quietly, so
    # `require_raw` has to ask about all of them. RAW is the baseline that removes nothing.
    active = []
    if cfg.score_thresh > 0.0:
        active.append(f"score_thresh={cfg.score_thresh}")
    if any(floor > 0.0 for floor in (cfg.class_thresholds or {}).values()):
        active.append(f"class_thresholds={cfg.class_thresholds}")
    if cfg.nms_iou is not None:
        active.append(f"nms_iou={cfg.nms_iou}")
    if cfg.merge_iou is not None:
        active.append(f"merge_iou={cfg.merge_iou}")
    if cfg.cross_class_iou is not None:
        active.append(f"cross_class_iou={cfg.cross_class_iou}")
    if cfg.min_box_size > 0:
        active.append(f"min_box_size={cfg.min_box_size}")
    if cfg.max_detections < RAW.max_detections:
        active.append(f"max_detections={cfg.max_detections}")
    return tuple(active)


@dataclass(frozen=True)
class PredictionMeta:
    """What a prediction CSV was produced with, read back from its sidecar."""

    postprocess: PostprocessConfig
    image_ids: tuple[str, ...] = ()
    n_detections: int = 0
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def is_raw(self) -> bool:
        """True only when nothing in the config could have removed a detection."""
        return not _active_filters(self.postprocess)


def classwise_nms(
    boxes: Tensor, scores: Tensor, labels: Tensor, iou: float
) -> tuple[Tensor, Tensor, Tensor]:
    """Non-maximum suppression within each class, so an object never suppresses its own shadow."""
    if boxes.numel() == 0:
        return boxes, scores, labels
    keep = batched_nms(boxes, scores, labels, iou)
    return boxes[keep], scores[keep], labels[keep]


def merge_overlapping(
    boxes: Tensor, scores: Tensor, labels: Tensor, iou: float
) -> tuple[Tensor, Tensor, Tensor]:
    """Fuse same-class boxes above `iou` into a score-weighted average carrying the best score."""
    if boxes.numel() == 0:
        return boxes, scores, labels

    fused_boxes: list[Tensor] = []
    fused_scores: list[Tensor] = []
    fused_labels: list[int] = []
    for cls in labels.unique().tolist():
        index = torch.nonzero(labels == cls, as_tuple=False).squeeze(1)
        order = index[torch.argsort(scores[index], descending=True)]
        cls_boxes, cls_scores = boxes[order], scores[order]
        overlap = box_iou(cls_boxes, cls_boxes)
        available = torch.ones(order.numel(), dtype=torch.bool, device=boxes.device)
        for rank in range(order.numel()):
            if not available[rank]:
                continue
            members = (overlap[rank] >= iou) & available
            weights = cls_scores[members]
            member_boxes = cls_boxes[members]
            total = weights.sum()
            # Weighting by score keeps the fused box near the confident member rather than
            # halfway between a good detection and a weak duplicate. All-zero weights reach
            # here whenever merge_iou is set without a score floor, and normalising by a
            # clamped zero would put the fused box at the origin; a plain mean at least lands
            # where the members are.
            if float(total) > 0.0:
                fused_boxes.append((member_boxes * weights[:, None]).sum(0) / total)
            else:
                fused_boxes.append(member_boxes.mean(0))
            fused_scores.append(weights.max())
            fused_labels.append(int(cls))
            available &= ~members

    merged_boxes = torch.stack(fused_boxes)
    merged_scores = torch.stack(fused_scores)
    merged_labels = torch.tensor(fused_labels, dtype=labels.dtype, device=labels.device)
    order = torch.argsort(merged_scores, descending=True)
    return merged_boxes[order], merged_scores[order], merged_labels[order]


def suppress_cross_class(
    boxes: Tensor, scores: Tensor, labels: Tensor, iou: float
) -> tuple[Tensor, Tensor, Tensor]:
    """Drop the weaker of two near-identical boxes that disagree about the class."""
    if boxes.numel() == 0:
        return boxes, scores, labels

    order = torch.argsort(scores, descending=True)
    ordered_labels = labels[order]
    overlap = box_iou(boxes[order], boxes[order])
    keep = torch.ones(order.numel(), dtype=torch.bool, device=boxes.device)
    for rank in range(order.numel()):
        if not keep[rank]:
            continue
        clash = (overlap[rank] >= iou) & (ordered_labels != ordered_labels[rank])
        clash[: rank + 1] = False
        keep &= ~clash
    final = order[keep]
    return boxes[final], scores[final], labels[final]


def _threshold_mask(scores: Tensor, labels: Tensor, cfg: PostprocessConfig) -> Tensor:
    per_class = cfg.class_thresholds or {}
    if not per_class:
        return scores >= cfg.score_thresh
    floors = torch.tensor(
        [per_class.get(int(label), cfg.score_thresh) for label in labels],
        dtype=scores.dtype,
        device=scores.device,
    )
    return scores >= floors


def postprocess(
    boxes: Tensor,
    scores: Tensor,
    labels: Tensor,
    cfg: PostprocessConfig,
    image_size: tuple[int, int] | None = None,
) -> tuple[Tensor, Tensor, Tensor]:
    """Apply `cfg` to one image's detections; `image_size` is (height, width), as torchvision."""
    boxes = boxes.to(torch.float32)
    if boxes.numel() == 0:
        return boxes.reshape(-1, 4), scores, labels

    if image_size is not None:
        boxes = clip_boxes_to_image(boxes, image_size)

    widths = boxes[:, 2] - boxes[:, 0]
    heights = boxes[:, 3] - boxes[:, 1]
    keep = (widths > 0) & (heights > 0)
    if cfg.min_box_size > 0:
        keep &= (widths >= cfg.min_box_size) & (heights >= cfg.min_box_size)
    keep &= _threshold_mask(scores, labels, cfg)
    boxes, scores, labels = boxes[keep], scores[keep], labels[keep]

    if cfg.nms_iou is not None:
        boxes, scores, labels = classwise_nms(boxes, scores, labels, cfg.nms_iou)
    if cfg.merge_iou is not None:
        boxes, scores, labels = merge_overlapping(boxes, scores, labels, cfg.merge_iou)
    if cfg.cross_class_iou is not None:
        boxes, scores, labels = suppress_cross_class(boxes, scores, labels, cfg.cross_class_iou)

    if scores.numel() > cfg.max_detections:
        top = torch.argsort(scores, descending=True)[: cfg.max_detections]
        boxes, scores, labels = boxes[top], scores[top], labels[top]
    return boxes, scores, labels


def predict_dataset(
    model: torch.nn.Module,
    dataset: Any,
    *,
    device: str | torch.device = "cpu",
    batch_size: int = 4,
    cfg: PostprocessConfig = RAW,
) -> dict[str, dict]:
    """Run `model` over `dataset` and return `{image_id: {"boxes", "scores", "labels"}}`."""
    # The CSV is keyed by VOC image id, not by position, so the dataset's own id list is what
    # the results have to be zipped against.
    ids = list(dataset.ids) if hasattr(dataset, "ids") else [str(i) for i in range(len(dataset))]
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
        for images, _targets in loader:
            batch = [image.to(device) for image in images]
            outputs = model.predict(batch) if hasattr(model, "predict") else model(batch)
            # Results are matched to ids by position, so a short batch would not fail here - it
            # would file every later detection under the wrong image.
            if len(outputs) != len(batch):
                raise ValueError(
                    f"model returned {len(outputs)} outputs for a batch of {len(batch)} images; "
                    "predictions cannot be matched to image ids"
                )
            for image, output in zip(batch, outputs):
                boxes, scores, labels = postprocess(
                    output["boxes"].detach().cpu(),
                    output["scores"].detach().cpu(),
                    output["labels"].detach().cpu(),
                    cfg,
                    image_size=(image.shape[-2], image.shape[-1]),
                )
                predictions[ids[cursor]] = {"boxes": boxes, "scores": scores, "labels": labels}
                cursor += 1
    return predictions


def meta_path(path: str | Path) -> Path:
    """Location of the sidecar belonging to a prediction CSV."""
    return Path(f"{Path(path)}.meta.json")


def write_predictions(
    predictions: Mapping[str, dict],
    path: str | Path,
    *,
    cfg: PostprocessConfig = RAW,
) -> None:
    """Write a prediction CSV plus the sidecar that records how it was filtered.

    `cfg` must be the config the predictions were produced with: the sidecar is what later
    refuses the file for metrics, and the RAW default would claim a filtered file is unfiltered.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    n_detections = 0
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(CSV_COLUMNS)
        for image_id in sorted(predictions):
            entry = predictions[image_id]
            boxes = entry["boxes"].detach().cpu()
            scores = entry["scores"].detach().cpu()
            labels = entry["labels"].detach().cpu()
            for box, score, label in zip(boxes, scores, labels):
                class_id = int(label)
                xmin, ymin, xmax, ymax = (float(v) for v in box.tolist())
                # repr() is the shortest string that reads back bit-identical; rounding here
                # would quietly move the boxes that later get scored.
                writer.writerow(
                    [
                        image_id,
                        class_id,
                        CLASS_NAMES.get(class_id, "unknown"),
                        repr(float(score)),
                        repr(xmin),
                        repr(ymin),
                        repr(xmax),
                        repr(ymax),
                    ]
                )
                n_detections += 1

    meta = {
        "format": META_FORMAT,
        "postprocess": asdict(cfg),
        "image_ids": sorted(predictions),
        "n_detections": n_detections,
    }
    meta_path(path).write_text(json.dumps(meta, indent=2) + "\n")


def read_meta(path: str | Path) -> PredictionMeta | None:
    """Parse the sidecar beside a prediction CSV, or None when there is none."""
    sidecar = meta_path(path)
    if not sidecar.is_file():
        return None
    payload = json.loads(sidecar.read_text())
    stored = dict(payload.get("postprocess", {}))
    # JSON has no integer keys, so the per-class thresholds come back as strings.
    thresholds = stored.get("class_thresholds")
    if thresholds:
        stored["class_thresholds"] = {int(k): float(v) for k, v in thresholds.items()}
    known = {f: stored[f] for f in PostprocessConfig.__dataclass_fields__ if f in stored}
    return PredictionMeta(
        postprocess=PostprocessConfig(**known),
        image_ids=tuple(payload.get("image_ids", ())),
        n_detections=int(payload.get("n_detections", 0)),
        extra={k: v for k, v in payload.items() if k not in _META_KEYS},
    )


def read_predictions(path: str | Path, *, require_raw: bool = False) -> dict[str, dict]:
    """Read a prediction CSV back into tensors; `require_raw` refuses anything filtered.

    `require_raw` also refuses a file with no sidecar at all: an unknown score floor is not the
    same as no score floor, and the original exports are exactly the files without one.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Missing prediction file: {path}")

    meta = read_meta(path)
    if require_raw:
        if meta is None:
            raise ValueError(
                f"{path} has no {meta_path(path).name} sidecar, so its score floor is unknown; "
                "re-export it with write_predictions before scoring"
            )
        filters = _active_filters(meta.postprocess)
        if filters:
            raise ValueError(
                f"{path} was exported with {', '.join(filters)}; "
                "metrics computed from it would sit on a truncated precision-recall curve"
            )

    rows: dict[str, tuple[list[list[float]], list[float], list[int]]] = {}
    # Ids from the sidecar are seeded first: an image with no detections has no CSV row, and
    # losing it would silently turn a miss into an absence.
    for image_id in meta.image_ids if meta is not None else ():
        rows[image_id] = ([], [], [])

    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        missing = [c for c in CSV_COLUMNS if c not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(f"{path} is missing prediction columns: {', '.join(missing)}")
        for row in reader:
            boxes, scores, labels = rows.setdefault(row["image_id"], ([], [], []))
            boxes.append(
                [float(row["xmin"]), float(row["ymin"]), float(row["xmax"]), float(row["ymax"])]
            )
            scores.append(float(row["score"]))
            labels.append(int(row["class_id"]))

    predictions: dict[str, dict] = {}
    for image_id, (boxes, scores, labels) in rows.items():
        predictions[image_id] = {
            "boxes": torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4),
            "scores": torch.tensor(scores, dtype=torch.float32),
            "labels": torch.tensor(labels, dtype=torch.int64),
        }
    return predictions
