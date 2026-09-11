"""Prediction post-processing and CSV export, including the score-floored export bug (B7).

The mutation notes on the IoU and clipping tests record what each assertion is there to catch.
"""

from __future__ import annotations

import csv
import json

import pytest
import torch
from torch import nn

from sonar.data.voc import VOCDetection
from sonar.engine.predict import (
    RAW,
    VIS,
    PostprocessConfig,
    classwise_nms,
    merge_overlapping,
    meta_path,
    postprocess,
    predict_dataset,
    read_meta,
    read_predictions,
    suppress_cross_class,
    write_predictions,
)


def boxes_scores_labels(rows: list[tuple[float, float, float, float, float, int]]):
    """Build the (boxes, scores, labels) triple from (xmin, ymin, xmax, ymax, score, label) rows."""
    boxes = torch.tensor([r[:4] for r in rows], dtype=torch.float32)
    scores = torch.tensor([r[4] for r in rows], dtype=torch.float32)
    labels = torch.tensor([r[5] for r in rows], dtype=torch.int64)
    return boxes, scores, labels


def test_raw_keeps_low_score_box_that_vis_drops():
    boxes, scores, labels = boxes_scores_labels([(10, 10, 30, 30, 0.01, 1)])

    kept_raw = postprocess(boxes, scores, labels, RAW)[1]
    kept_vis = postprocess(boxes, scores, labels, VIS)[1]

    assert kept_raw.tolist() == [pytest.approx(0.01)]
    assert kept_vis.numel() == 0


def test_raw_config_removes_nothing_from_a_crowded_image():
    rows = [(0, 0, 4, 4, 0.02, 1), (0, 0, 4, 4, 0.9, 2), (1, 1, 5, 5, 0.5, 1)]
    boxes, scores, labels = boxes_scores_labels(rows)

    kept_boxes, kept_scores, _ = postprocess(boxes, scores, labels, RAW)

    assert len(kept_boxes) == len(rows)
    assert sorted(kept_scores.tolist()) == sorted(scores.tolist())


def test_classwise_nms_keeps_a_duplicate_of_a_different_class():
    boxes, scores, labels = boxes_scores_labels(
        [(10, 10, 30, 30, 0.9, 1), (10, 10, 30, 30, 0.8, 2)]
    )

    _, kept_scores, kept_labels = classwise_nms(boxes, scores, labels, 0.5)

    assert sorted(kept_labels.tolist()) == [1, 2]
    assert sorted(kept_scores.tolist()) == pytest.approx([0.8, 0.9])


def test_classwise_nms_suppresses_a_duplicate_of_the_same_class():
    boxes, scores, labels = boxes_scores_labels(
        [(10, 10, 30, 30, 0.9, 1), (11, 11, 31, 31, 0.8, 1)]
    )

    _, kept_scores, kept_labels = classwise_nms(boxes, scores, labels, 0.5)

    assert kept_labels.tolist() == [1]
    assert kept_scores.tolist() == pytest.approx([0.9])


def test_classwise_nms_keeps_two_same_class_boxes_that_barely_overlap():
    # IoU here is 0.14, well under the threshold. Hardcoding the threshold to 0.0 fuses these.
    boxes, scores, labels = boxes_scores_labels(
        [(10, 10, 30, 30, 0.9, 1), (25, 10, 45, 30, 0.8, 1)]
    )

    kept_boxes, kept_scores, _ = classwise_nms(boxes, scores, labels, 0.5)

    assert len(kept_boxes) == 2
    assert sorted(kept_scores.tolist()) == pytest.approx([0.8, 0.9])


def test_suppress_cross_class_keeps_only_the_higher_scoring_label():
    boxes, scores, labels = boxes_scores_labels(
        [(10, 10, 30, 30, 0.4, 1), (10, 10, 30, 30, 0.7, 2)]
    )

    _, kept_scores, kept_labels = suppress_cross_class(boxes, scores, labels, 0.85)

    assert kept_labels.tolist() == [2]
    assert kept_scores.tolist() == pytest.approx([0.7])


def test_suppress_cross_class_leaves_distant_boxes_alone():
    boxes, scores, labels = boxes_scores_labels(
        [(10, 10, 30, 30, 0.7, 1), (60, 60, 80, 80, 0.4, 2)]
    )

    kept_boxes, _, kept_labels = suppress_cross_class(boxes, scores, labels, 0.85)

    assert len(kept_boxes) == 2
    assert sorted(kept_labels.tolist()) == [1, 2]


def test_merge_overlapping_fuses_same_class_boxes_and_keeps_the_best_score():
    boxes, scores, labels = boxes_scores_labels(
        [(10, 10, 30, 30, 0.9, 1), (12, 12, 32, 32, 0.3, 1)]
    )

    kept_boxes, kept_scores, kept_labels = merge_overlapping(boxes, scores, labels, 0.6)

    assert len(kept_boxes) == 1
    assert kept_scores.tolist() == pytest.approx([0.9])
    assert kept_labels.tolist() == [1]
    # Score weighting must pull the fused box towards the confident member, not to the midpoint.
    assert 10.0 < float(kept_boxes[0, 0]) < 11.0


def test_merge_overlapping_keeps_separate_classes_separate():
    boxes, scores, labels = boxes_scores_labels(
        [(10, 10, 30, 30, 0.9, 1), (10, 10, 30, 30, 0.8, 2)]
    )

    _, _, kept_labels = merge_overlapping(boxes, scores, labels, 0.6)

    assert sorted(kept_labels.tolist()) == [1, 2]


def test_merge_overlapping_leaves_same_class_boxes_below_the_threshold_alone():
    # IoU here is 0.14, under the 0.6 threshold: fusing every same-class box would join these.
    boxes, scores, labels = boxes_scores_labels(
        [(10, 10, 30, 30, 0.9, 1), (25, 10, 45, 30, 0.8, 1)]
    )

    kept_boxes, kept_scores, _ = merge_overlapping(boxes, scores, labels, 0.6)

    assert len(kept_boxes) == 2
    assert kept_scores.tolist() == pytest.approx([0.9, 0.8])


def test_merge_overlapping_falls_back_to_a_plain_mean_when_every_score_is_zero():
    """merge_iou without a score floor can fuse boxes whose weights sum to zero."""
    boxes, scores, labels = boxes_scores_labels(
        [(10, 10, 30, 30, 0.0, 1), (12, 12, 32, 32, 0.0, 1)]
    )

    kept_boxes, _, _ = merge_overlapping(boxes, scores, labels, 0.6)

    assert len(kept_boxes) == 1
    assert kept_boxes[0].tolist() == [11.0, 11.0, 31.0, 31.0]


def test_boxes_are_clipped_into_the_image_and_degenerate_results_dropped():
    # image_size is (height, width). The wide box has to stop at x=64 and the tall one at y=48,
    # so swapping the pair moves both of them; the last box leaves the frame entirely.
    rows = [
        (-5, -5, 20, 20, 0.9, 1),
        (10, 10, 90, 30, 0.8, 1),
        (34, 10, 50, 80, 0.7, 2),
        (70, 10, 90, 30, 0.6, 1),
    ]
    boxes, scores, labels = boxes_scores_labels(rows)

    kept_boxes, kept_scores, _ = postprocess(boxes, scores, labels, RAW, image_size=(48, 64))

    assert kept_boxes.tolist() == [
        [0.0, 0.0, 20.0, 20.0],
        [10.0, 10.0, 64.0, 30.0],
        [34.0, 10.0, 50.0, 48.0],
    ]
    assert kept_scores.tolist() == pytest.approx([0.9, 0.8, 0.7])


def test_min_box_size_drops_thin_detections():
    cfg = PostprocessConfig(min_box_size=8)
    boxes, scores, labels = boxes_scores_labels(
        [(10, 10, 30, 30, 0.9, 1), (10, 10, 14, 30, 0.9, 1)]
    )

    kept_boxes, _, _ = postprocess(boxes, scores, labels, cfg)

    assert len(kept_boxes) == 1
    assert kept_boxes[0].tolist() == [10.0, 10.0, 30.0, 30.0]


def test_class_thresholds_override_the_global_floor():
    cfg = PostprocessConfig(score_thresh=0.2, class_thresholds={2: 0.8})
    boxes, scores, labels = boxes_scores_labels(
        [(10, 10, 30, 30, 0.5, 1), (40, 40, 60, 60, 0.5, 2)]
    )

    _, _, kept_labels = postprocess(boxes, scores, labels, cfg)

    assert kept_labels.tolist() == [1]


def test_max_detections_keeps_the_highest_scores():
    cfg = PostprocessConfig(max_detections=2)
    rows = [(i * 10, 0, i * 10 + 5, 5, 0.1 * (i + 1), 1) for i in range(5)]
    boxes, scores, labels = boxes_scores_labels(rows)

    _, kept_scores, _ = postprocess(boxes, scores, labels, cfg)

    assert sorted(kept_scores.tolist(), reverse=True) == pytest.approx([0.5, 0.4])


def test_postprocess_handles_an_image_with_no_detections():
    boxes = torch.zeros((0, 4), dtype=torch.float32)
    scores = torch.zeros((0,), dtype=torch.float32)
    labels = torch.zeros((0,), dtype=torch.int64)

    kept_boxes, kept_scores, kept_labels = postprocess(
        boxes, scores, labels, VIS, image_size=(48, 64)
    )

    assert kept_boxes.shape == (0, 4)
    assert kept_scores.numel() == 0
    assert kept_labels.numel() == 0


def test_csv_round_trips_the_exact_float_values(tmp_path):
    predictions = {
        "img_000": {
            "boxes": torch.tensor([[1.2345679, 2.7182817, 31.415926, 42.424244]]),
            "scores": torch.tensor([0.0123456791]),
            "labels": torch.tensor([1]),
        },
        "img_001": {
            "boxes": torch.tensor([[0.1, 0.2, 63.9, 47.9], [5.0, 5.0, 6.5, 6.5]]),
            "scores": torch.tensor([0.9999999, 0.3333333]),
            "labels": torch.tensor([2, 1]),
        },
    }
    # Nested on purpose: write_predictions creates the run directory rather than failing.
    path = tmp_path / "runs" / "exp1" / "preds.csv"

    write_predictions(predictions, path, cfg=RAW)
    restored = read_predictions(path)

    assert path.is_file()
    assert set(restored) == set(predictions)
    for image_id, entry in predictions.items():
        assert torch.equal(restored[image_id]["boxes"], entry["boxes"])
        assert torch.equal(restored[image_id]["scores"], entry["scores"])
        assert torch.equal(restored[image_id]["labels"], entry["labels"])


def test_csv_header_is_exactly_the_agreed_columns(tmp_path):
    path = tmp_path / "preds.csv"
    predictions = {
        "img_000": {
            "boxes": torch.tensor([[1.0, 2.0, 3.0, 4.0]]),
            "scores": torch.tensor([0.5]),
            "labels": torch.tensor([2]),
        }
    }

    write_predictions(predictions, path)

    rows = list(csv.reader(path.open()))
    assert rows[0] == [
        "image_id", "class_id", "class_name", "score", "xmin", "ymin", "xmax", "ymax"
    ]
    assert rows[1][:3] == ["img_000", "2", "shadow"]


def test_an_image_with_zero_detections_survives_the_round_trip(tmp_path):
    predictions = {
        "img_000": {
            "boxes": torch.zeros((0, 4)),
            "scores": torch.zeros((0,)),
            "labels": torch.zeros((0,), dtype=torch.int64),
        },
        "img_001": {
            "boxes": torch.tensor([[1.0, 1.0, 9.0, 9.0]]),
            "scores": torch.tensor([0.7]),
            "labels": torch.tensor([1]),
        },
    }
    path = tmp_path / "preds.csv"

    write_predictions(predictions, path)
    restored = read_predictions(path)

    # An image that predicted nothing is not the same as an image that was never scored: the
    # evaluator has to see it to count the misses.
    assert set(restored) == {"img_000", "img_001"}
    assert restored["img_000"]["boxes"].shape == (0, 4)


def test_sidecar_records_the_config_that_produced_the_file(tmp_path):
    path = tmp_path / "preds_vis.csv"
    predictions = {
        "img_000": {
            "boxes": torch.tensor([[10.0, 10.0, 30.0, 30.0]]),
            "scores": torch.tensor([0.9]),
            "labels": torch.tensor([1]),
        }
    }

    write_predictions(predictions, path, cfg=VIS)

    payload = json.loads(meta_path(path).read_text())
    assert payload["format"] == 1
    assert payload["postprocess"]["score_thresh"] == pytest.approx(0.5)
    assert payload["postprocess"]["nms_iou"] == pytest.approx(0.5)
    assert payload["postprocess"]["min_box_size"] == 8
    assert payload["image_ids"] == ["img_000"]
    assert payload["n_detections"] == len(list(csv.reader(path.open()))) - 1

    meta = read_meta(path)
    assert meta is not None
    assert meta.postprocess == VIS
    assert meta.n_detections == 1
    assert meta.is_raw is False
    # `extra` is for caller keys; the sidecar's own format version is not one of them.
    assert meta.extra == {}


def test_sidecar_round_trips_per_class_thresholds(tmp_path):
    cfg = PostprocessConfig(score_thresh=0.3, class_thresholds={1: 0.4, 2: 0.6})
    path = tmp_path / "preds.csv"

    write_predictions({}, path, cfg=cfg)
    meta = read_meta(path)

    assert meta is not None
    assert meta.postprocess.class_thresholds == {1: 0.4, 2: 0.6}


def test_require_raw_rejects_a_score_floored_export(tmp_path):
    """Regression test for B7: a filtered CSV must not be usable as a metrics file."""
    detections = {
        "img_000": {
            "boxes": torch.tensor([[10.0, 10.0, 30.0, 30.0], [40.0, 10.0, 60.0, 30.0]]),
            "scores": torch.tensor([0.9, 0.05]),
            "labels": torch.tensor([1, 1]),
        }
    }
    raw_path = tmp_path / "preds_raw.csv"
    vis_path = tmp_path / "preds_vis.csv"

    def exported(cfg):
        keys = ("boxes", "scores", "labels")
        return {
            image_id: dict(zip(keys, postprocess(v["boxes"], v["scores"], v["labels"], cfg)))
            for image_id, v in detections.items()
        }

    raw, floored = exported(RAW), exported(VIS)
    write_predictions(raw, raw_path, cfg=RAW)
    write_predictions(floored, vis_path, cfg=VIS)

    # The floored file is a different file, not a cosmetic variant: the low-score detection that
    # extends the precision-recall curve is simply gone.
    raw_scores = read_predictions(raw_path)["img_000"]["scores"]
    floored_scores = read_predictions(vis_path)["img_000"]["scores"]
    assert raw_scores.tolist() == pytest.approx([0.9, 0.05])
    assert floored_scores.tolist() == pytest.approx([0.9])

    read_predictions(raw_path, require_raw=True)
    with pytest.raises(ValueError, match="score_thresh"):
        read_predictions(vis_path, require_raw=True)


@pytest.mark.parametrize(
    "cfg",
    [
        PostprocessConfig(class_thresholds={1: 0.9}),
        PostprocessConfig(nms_iou=0.5),
        PostprocessConfig(merge_iou=0.6),
        PostprocessConfig(min_box_size=8),
        PostprocessConfig(max_detections=1),
    ],
)
def test_require_raw_rejects_a_file_filtered_without_a_score_floor(tmp_path, cfg):
    """Every one of these removes detections, so none of them is a metrics file either."""
    predictions = {
        "img_000": {
            "boxes": torch.tensor([[10.0, 10.0, 30.0, 30.0]]),
            "scores": torch.tensor([0.9]),
            "labels": torch.tensor([1]),
        }
    }
    path = tmp_path / "preds.csv"

    write_predictions(predictions, path, cfg=cfg)

    assert read_meta(path).is_raw is False
    with pytest.raises(ValueError, match="truncated precision-recall curve"):
        read_predictions(path, require_raw=True)


def test_require_raw_rejects_a_file_with_no_provenance(tmp_path):
    """A CSV written the original way carries no sidecar, so its score floor is unknowable."""
    legacy = tmp_path / "legacy.csv"
    with legacy.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["image_id", "class_id", "class_name", "score", "xmin", "ymin", "xmax", "ymax"]
        )
        writer.writerow(["img_000", 1, "object", 0.91, 10.0, 10.0, 30.0, 30.0])

    assert len(read_predictions(legacy)["img_000"]["scores"]) == 1
    with pytest.raises(ValueError, match="sidecar"):
        read_predictions(legacy, require_raw=True)


def test_read_predictions_reports_a_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError, match=r"absent\.csv"):
        read_predictions(tmp_path / "absent.csv")


def test_read_predictions_rejects_a_csv_with_the_wrong_columns(tmp_path):
    path = tmp_path / "wrong.csv"
    path.write_text("image_id,score\nimg_000,0.5\n")

    with pytest.raises(ValueError, match="xmin"):
        read_predictions(path)


class CountingDetector(nn.Module):
    """Emits two boxes per image, the larger scored from the call order so ids are traceable."""

    def __init__(self) -> None:
        super().__init__()
        self.seen = 0

    def forward(self, images):
        outputs = []
        for _ in images:
            # The 1x1 box scores 0.9 so that only min_box_size can remove it.
            outputs.append(
                {
                    "boxes": torch.tensor([[2.0, 2.0, 20.0, 20.0], [0.0, 0.0, 1.0, 1.0]]),
                    "scores": torch.tensor([0.1 * (self.seen + 1), 0.9]),
                    "labels": torch.tensor([1, 2]),
                }
            )
            self.seen += 1
        return outputs


class OffFrameDetector(nn.Module):
    """Emits one box running off the right and bottom edges of every image."""

    def forward(self, images):
        return [
            {
                "boxes": torch.tensor([[10.0, 10.0, 200.0, 200.0]]),
                "scores": torch.tensor([0.9]),
                "labels": torch.tensor([1]),
            }
            for _ in images
        ]


class ShortBatchDetector(nn.Module):
    """Returns one output fewer than it was given, the way a bad collate does."""

    def forward(self, images):
        return [
            {
                "boxes": torch.tensor([[1.0, 1.0, 9.0, 9.0]]),
                "scores": torch.tensor([0.5]),
                "labels": torch.tensor([1]),
            }
            for _ in images[:-1]
        ]


def test_predict_dataset_keys_results_by_image_id_in_dataset_order(voc_root):
    # Reversed ids: sorting the results, or keying them by position in the split file, both
    # disagree with dataset order here, where a pre-sorted id list would hide either.
    ordered = list(reversed(VOCDetection(voc_root, "train").ids))
    dataset = VOCDetection(voc_root, "train", ids=ordered)
    model = CountingDetector()

    predictions = predict_dataset(model, dataset, batch_size=3)

    assert list(predictions) == ordered
    for position, image_id in enumerate(ordered):
        score = float(predictions[image_id]["scores"][0])
        assert score == pytest.approx(0.1 * (position + 1), abs=1e-6)


def test_predict_dataset_applies_the_postprocess_config(voc_root):
    dataset = VOCDetection(voc_root, "val")
    raw = predict_dataset(CountingDetector(), dataset, cfg=RAW)
    vis = predict_dataset(CountingDetector(), dataset, cfg=VIS)

    first = dataset.ids[0]
    assert len(raw[first]["scores"]) == 2
    # VIS removes the 0.9-scoring 1x1 box on size and the 0.1 box on score; dropping either
    # filter leaves one detection here.
    assert len(vis[first]["scores"]) == 0


def test_predict_dataset_clips_boxes_to_the_frame_of_each_image(voc_root):
    dataset = VOCDetection(voc_root, "val")

    predictions = predict_dataset(OffFrameDetector(), dataset)

    # The fixture images are 64 wide and 48 high, so a swapped (height, width) pair shows up.
    assert predictions[dataset.ids[0]]["boxes"].tolist() == [[10.0, 10.0, 64.0, 48.0]]


def test_predict_dataset_refuses_a_model_that_returns_too_few_outputs(voc_root):
    """Matching by position means a short batch would silently shift ids, not fail."""
    dataset = VOCDetection(voc_root, "val")

    with pytest.raises(ValueError, match="cannot be matched to image ids"):
        predict_dataset(ShortBatchDetector(), dataset, batch_size=2)
