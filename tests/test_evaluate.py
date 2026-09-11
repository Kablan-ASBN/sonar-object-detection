"""Tests for detection metrics and the ground-truth binding that keeps them comparable."""

from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image
from torch import Tensor
from torchvision.ops import box_iou

from sonar.engine.evaluate import GroundTruth, coco_metrics, compare, evaluate_model, froc_curve


def perfect_predictions(gt: GroundTruth, *, score: float = 0.9, dx: float = 0.0) -> dict[str, dict]:
    """One detection per ground-truth box, optionally displaced along x."""
    offset = torch.tensor([dx, 0.0, dx, 0.0])
    return {
        image_id: {
            "boxes": gt.boxes[image_id] + offset,
            "scores": torch.full((len(gt.labels[image_id]),), score),
            "labels": gt.labels[image_id].clone(),
        }
        for image_id in gt.ids
    }


def empty_predictions(gt: GroundTruth) -> dict[str, dict]:
    return {
        image_id: {
            "boxes": torch.zeros((0, 4)),
            "scores": torch.zeros((0,)),
            "labels": torch.zeros((0,), dtype=torch.int64),
        }
        for image_id in gt.ids
    }


def shift_annotations(root: Path, ids: Sequence[str], dy: int) -> None:
    """Move every box in `ids` vertically, so the root's ground truth genuinely differs."""
    for image_id in ids:
        path = root / "Annotations" / f"{image_id}.xml"
        tree = ET.parse(path)
        for bndbox in tree.getroot().iter("bndbox"):
            for tag in ("ymin", "ymax"):
                node = bndbox.find(tag)
                node.text = str(int(node.text) + dy)
        tree.write(path)


def blank_images(root: Path) -> None:
    """Replace every image with flat black, leaving the annotations as they were."""
    for path in sorted((root / "JPEGImages").glob("*.jpg")):
        with Image.open(path) as image:
            size = image.size
        Image.new("RGB", size).save(path, quality=95)


def one_image_gt(boxes: list[list[float]], labels: list[int]) -> GroundTruth:
    """A single-image ground truth with exact geometry, for pinning the matcher's rules."""
    return GroundTruth(
        root=Path("in-memory"),
        split="matcher",
        ids=("matcher",),
        boxes={"matcher": torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4)},
        labels={"matcher": torch.tensor(labels, dtype=torch.int64)},
        sizes={"matcher": (64, 48)},
    )


def detections(
    boxes: list[list[float]], scores: list[float], labels: list[int]
) -> dict[str, dict]:
    """Predictions for the single image of `one_image_gt`."""
    return {
        "matcher": {
            "boxes": torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4),
            "scores": torch.tensor(scores, dtype=torch.float32),
            "labels": torch.tensor(labels, dtype=torch.int64),
        }
    }


class SquareDetector(torch.nn.Module):
    """Reports the bright square it can actually see, and counts the images it was shown.

    The fixture puts that square at a different place in every image, and the `object` box is
    exactly the square, so a detection is only correct if it was filed against its own image.
    """

    def __init__(self) -> None:
        super().__init__()
        self.seen = 0

    def forward(self, images: list[Tensor]) -> list[dict[str, Tensor]]:
        self.seen += len(images)
        outputs = []
        for image in images:
            ys, xs = torch.where(image.mean(dim=0) > 0.7)
            if xs.numel() == 0:
                outputs.append(
                    {
                        "boxes": torch.zeros((0, 4)),
                        "scores": torch.zeros((0,)),
                        "labels": torch.zeros((0,), dtype=torch.int64),
                    }
                )
                continue
            box = [[xs.min(), ys.min(), xs.max() + 1, ys.max() + 1]]
            outputs.append(
                {
                    "boxes": torch.tensor(box, dtype=torch.float32),
                    "scores": torch.tensor([0.9]),
                    "labels": torch.tensor([1]),
                }
            )
        return outputs


def test_ground_truth_load_matches_the_split_file(voc_root: Path):
    gt = GroundTruth.load(voc_root, "test")
    listed = (voc_root / "ImageSets" / "Main" / "test.txt").read_text().split()

    assert list(gt.ids) == listed
    assert len(gt) == len(listed)
    assert all(gt.sizes[i] == (64, 48) for i in gt.ids)
    # Both split images carry an object and a shadow, so a parser that dropped a class shows here.
    assert set(torch.cat([gt.labels[i] for i in gt.ids]).tolist()) == {1, 2}


def test_perfect_predictions_give_ap50_of_one(voc_root: Path):
    gt = GroundTruth.load(voc_root, "test")
    metrics = coco_metrics(perfect_predictions(gt), gt)

    assert metrics["AP50"] == pytest.approx(1.0)
    assert metrics["mAP"] == pytest.approx(1.0)
    assert metrics["mAR100"] == pytest.approx(1.0)
    assert metrics["AP50_object"] == pytest.approx(1.0)
    assert metrics["AP50_shadow"] == pytest.approx(1.0)


def test_empty_predictions_give_ap50_of_zero(voc_root: Path):
    gt = GroundTruth.load(voc_root, "test")
    metrics = coco_metrics(empty_predictions(gt), gt)

    assert metrics["AP50"] == 0.0
    assert metrics["mAP"] == 0.0
    assert metrics["mAR100"] == 0.0
    # Both classes are in this ground truth, so zero is a score here and not a missing measurement.
    assert metrics["AP50_object"] == 0.0
    assert metrics["AP50_shadow"] == 0.0


def test_a_class_absent_from_the_ground_truth_scores_nan_not_zero(voc_root: Path):
    # The fixture's first images carry an object and no shadow.
    gt = GroundTruth.load(voc_root, "test", ids=["img_000", "img_001"])
    metrics = coco_metrics(perfect_predictions(gt), gt)

    assert metrics["AP50_object"] == pytest.approx(1.0)
    assert math.isnan(metrics["AP50_shadow"])


def test_iou_just_above_half_hits_and_just_below_misses(voc_root: Path):
    gt = GroundTruth.load(voc_root, "test")
    first = gt.ids[0]
    above = perfect_predictions(gt, dx=2.6)
    below = perfect_predictions(gt, dx=2.8)

    iou_above = float(box_iou(above[first]["boxes"][:1], gt.boxes[first][:1]))
    iou_below = float(box_iou(below[first]["boxes"][:1], gt.boxes[first][:1]))
    assert iou_below < 0.5 < iou_above

    assert coco_metrics(above, gt)["AP50"] == pytest.approx(1.0)
    assert coco_metrics(below, gt)["AP50"] == 0.0
    # An IoU of 0.509 is a hit at 0.5 and nothing at all at 0.75, which keeps the two columns apart.
    assert coco_metrics(above, gt)["AP75"] == 0.0


def test_missing_image_ids_are_penalised_not_ignored(voc_root: Path):
    gt = GroundTruth.load(voc_root, "train")
    full = perfect_predictions(gt)
    # Every other id, so both classes survive and only the image count changes.
    half = {i: full[i] for i in gt.ids[::2]}

    all_ids = coco_metrics(full, gt)
    some_ids = coco_metrics(half, gt)

    # Dropping half the images must halve recall, not shrink the denominator with them.
    assert all_ids["AP50"] == pytest.approx(1.0)
    assert some_ids["AP50"] < all_ids["AP50"]
    assert some_ids["mAR100"] == pytest.approx(0.5, abs=0.05)


def test_predictions_from_another_split_are_rejected(voc_root: Path):
    gt = GroundTruth.load(voc_root, "test")
    strays = perfect_predictions(gt)
    strays["img_000"] = strays[gt.ids[0]]

    with pytest.raises(ValueError, match="outside"):
        coco_metrics(strays, gt)


def test_max_detections_other_than_100_is_still_a_real_score(voc_root: Path):
    gt = GroundTruth.load(voc_root, "test")
    predictions = perfect_predictions(gt)

    for cap in (5, 100, 300):
        metrics = coco_metrics(predictions, gt, max_detections=cap)

        # pycocotools summarises mAP at a hard-coded cap of 100 and reports -1 at any other, which
        # would arrive here as a plausible-looking zero against perfect predictions.
        assert metrics["mAP"] == pytest.approx(1.0)
        assert metrics["AP50"] == pytest.approx(1.0)
        assert metrics["AP50_object"] == pytest.approx(1.0)
        # The recall key names the cap it was measured at rather than always claiming 100.
        assert metrics[f"mAR{cap}"] == pytest.approx(1.0)


def test_max_detections_caps_the_detections_that_count(voc_root: Path):
    gt = GroundTruth.load(voc_root, "test")
    noisy = {}
    for image_id in gt.ids:
        boxes = gt.boxes[image_id]
        # Three confident false alarms per image, scored above the true boxes.
        alarms = torch.tensor(
            [[0.0, 40.0, 6.0, 46.0], [50.0, 2.0, 58.0, 10.0], [30.0, 30.0, 38.0, 38.0]]
        )
        noisy[image_id] = {
            "boxes": torch.cat([alarms, boxes]),
            "scores": torch.cat([torch.full((3,), 0.99), torch.full((len(boxes),), 0.5)]),
            "labels": torch.cat([torch.ones(3, dtype=torch.int64), gt.labels[image_id]]),
        }

    capped = coco_metrics(noisy, gt, max_detections=1)
    uncapped = coco_metrics(noisy, gt, max_detections=100)

    # At one detection per image and class the true object box never survives the false alarms.
    assert capped["AP50_object"] == 0.0
    assert uncapped["AP50_object"] > capped["AP50_object"]
    # Every summary column has to be measured at the same cap, not just the per-class ones.
    assert capped["AP50"] < uncapped["AP50"]


def test_froc_recall_is_monotone_in_fppi_and_starts_low(voc_root: Path):
    gt = GroundTruth.load(voc_root, "train")
    predictions = perfect_predictions(gt)
    for rank, image_id in enumerate(gt.ids):
        entry = predictions[image_id]
        # True boxes found with rising confidence, plus one low-scoring false alarm per image.
        predictions[image_id] = {
            "boxes": torch.cat([entry["boxes"], torch.tensor([[0.0, 40.0, 6.0, 46.0]])]),
            "scores": torch.cat(
                [
                    torch.full_like(entry["scores"], 0.55 + 0.05 * rank),
                    torch.tensor([0.1 + 0.01 * rank]),
                ]
            ),
            "labels": torch.cat([entry["labels"], torch.tensor([1])]),
        }

    fppi, recall = froc_curve(predictions, gt)

    assert np.all(np.diff(recall) >= -1e-9)
    # The most confident operating point sees only the best image, so recall starts well short.
    assert fppi[0] == 0.0
    assert recall[0] < 0.5
    assert recall[-1] == pytest.approx(1.0)
    assert fppi.max() == pytest.approx(1.0)


def test_froc_default_thresholds_follow_the_observed_scores(voc_root: Path):
    gt = GroundTruth.load(voc_root, "test")
    # Every score sits below a hand-rolled 0.05..0.95 grid, as a score-floored export would.
    quiet = perfect_predictions(gt, score=0.02)

    _, swept_recall = froc_curve(quiet, gt)
    _, gridded_recall = froc_curve(quiet, gt, thresholds=np.arange(0.05, 1.0, 0.05))

    assert swept_recall.max() == pytest.approx(1.0)
    assert gridded_recall.max() == 0.0


def test_froc_matching_is_class_aware():
    gt = one_image_gt([[0, 0, 8, 8], [20, 0, 28, 8]], [1, 2])
    # A shadow-labelled box sitting exactly on the object box is a false alarm, not a hit.
    fppi, recall = froc_curve(detections([[0, 0, 8, 8]], [0.9], [2]), gt)

    assert recall.max() == 0.0
    assert fppi.max() == pytest.approx(1.0)


def test_froc_gives_each_ground_truth_box_at_most_one_hit():
    gt = one_image_gt([[0, 0, 8, 8]], [1])
    duplicates = detections([[0, 0, 8, 8], [0, 0, 8, 8]], [0.9, 0.8], [1, 1])

    fppi, recall = froc_curve(duplicates, gt)

    assert recall.max() == pytest.approx(1.0)
    # The second copy of the box has nothing left to match, so it counts against the model.
    assert fppi.max() == pytest.approx(1.0)


def test_froc_counts_an_iou_exactly_at_the_threshold_as_a_hit():
    # 8 / (16 + 8 - 8) is 0.5 with no floating-point slack; the shifted box comes to 0.4875.
    gt = one_image_gt([[0, 0, 4, 4]], [1])

    _, at_threshold = froc_curve(detections([[0, 0, 2, 4]], [0.9], [1]), gt)
    _, below_threshold = froc_curve(detections([[0.05, 0, 2, 4]], [0.9], [1]), gt)

    assert at_threshold.max() == pytest.approx(1.0)
    assert below_threshold.max() == 0.0


def test_froc_matches_the_highest_scoring_detection_first():
    gt = one_image_gt([[0, 0, 10, 10]], [1])
    # The confident box overlaps enough to claim the ground truth; the exact box arrives too late.
    preds = detections([[2, 0, 12, 10], [0, 0, 10, 10]], [0.9, 0.5], [1, 1])

    fppi, recall = froc_curve(preds, gt)

    assert fppi[0] == 0.0
    assert recall[0] == pytest.approx(1.0)


def test_ground_truths_from_two_roots_differ_and_compare_takes_one(voc_root_factory):
    raw = voc_root_factory("raw")
    shifted = voc_root_factory("shifted")
    gt_raw = GroundTruth.load(raw, "test")
    shift_annotations(shifted, gt_raw.ids, dy=-5)
    gt_shifted = GroundTruth.load(shifted, "test")

    assert gt_raw.root != gt_shifted.root
    assert gt_raw.ids == gt_shifted.ids
    assert not torch.equal(gt_raw.boxes[gt_raw.ids[0]], gt_shifted.boxes[gt_raw.ids[0]])

    predictions = perfect_predictions(gt_raw)
    honest = coco_metrics(predictions, gt_raw)["AP50"]
    # The original tabulated numbers measured against whichever root was loaded at the time.
    mixed = coco_metrics(predictions, gt_shifted)["AP50"]
    assert honest == pytest.approx(1.0)
    assert mixed < honest

    table = compare({"raw": predictions}, gt_raw)
    assert table.loc["raw", "AP50"] == pytest.approx(1.0)

    with pytest.raises(ValueError, match="exactly one GroundTruth"):
        compare({"raw": predictions}, [gt_raw, gt_shifted])


def test_compare_orders_models_by_ap50_and_not_by_another_column(voc_root: Path):
    gt = GroundTruth.load(voc_root, "train")
    full = perfect_predictions(gt)
    models = {
        # Loose boxes clear IoU 0.5 everywhere, so AP50 is perfect while the IoU sweep is not.
        "wide": perfect_predictions(gt, dx=2.6),
        # Exact boxes on half the images: worse on AP50 than "wide", better on every other column.
        "partial": {i: full[i] for i in gt.ids[::2]},
        "blind": empty_predictions(gt),
    }

    table = compare(models, gt)

    assert list(table.index) == ["wide", "partial", "blind"]
    assert table.loc["partial", "mAP"] > table.loc["wide", "mAP"]
    assert "AP50_shadow" in table.columns


def test_compare_passes_its_options_through_to_coco_metrics(voc_root: Path):
    gt = GroundTruth.load(voc_root, "test")

    table = compare({"sharp": perfect_predictions(gt)}, gt, max_detections=300)

    assert "mAR300" in table.columns
    assert table.loc["sharp", "mAP"] == pytest.approx(1.0)


def test_evaluate_model_files_each_prediction_against_its_own_image(voc_root: Path):
    gt = GroundTruth.load(voc_root, "train")
    model = SquareDetector()

    metrics = evaluate_model(model, gt, batch_size=3)

    assert model.seen == len(gt)
    # The detector can only be right about the image it was shown, so a perfect object score is
    # the id-to-prediction mapping holding across batch boundaries.
    assert metrics["AP50_object"] == pytest.approx(1.0)


def test_evaluate_model_reads_its_pixels_from_image_root(voc_root_factory):
    labelled = voc_root_factory("labelled")
    blanked = voc_root_factory("blanked")
    blank_images(blanked)
    gt = GroundTruth.load(labelled, "train")

    own_images = evaluate_model(SquareDetector(), gt, batch_size=3)
    other_images = evaluate_model(SquareDetector(), gt, batch_size=3, image_root=blanked)

    assert own_images["AP50_object"] == pytest.approx(1.0)
    assert other_images["AP50_object"] == 0.0
