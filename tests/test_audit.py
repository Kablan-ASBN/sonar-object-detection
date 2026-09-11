"""Tests for the audit tools, including the split-leakage check that would have caught B2."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

import pytest

from conftest import HEIGHT, WIDTH, make_xml
from sonar.audit.annotations import audit_annotations
from sonar.audit.leakage import MAX_EXAMPLES, LeakageReport, Overlap, check_leakage, collect_splits


def write_splits(
    root: Path,
    train: Sequence[str],
    val: Sequence[str],
    test: Sequence[str],
) -> None:
    main = root / "ImageSets" / "Main"
    main.mkdir(parents=True, exist_ok=True)
    for name, ids in (("train", train), ("val", val), ("test", test)):
        (main / f"{name}.txt").write_text("".join(f"{image_id}\n" for image_id in ids))


def read_matrix(text: str) -> dict[tuple[str, str], int]:
    """Read the shared-id matrix back out of format() output, keyed (row, column)."""
    lines = text.splitlines()
    headers = [i for i, line in enumerate(lines) if line.startswith("shared ids")]
    assert headers, "format() printed no matrix"

    columns = lines[headers[0] + 1].split()
    cells: dict[tuple[str, str], int] = {}
    for line in lines[headers[0] + 2 :]:
        if not line.strip():
            break
        row, *counts = line.split()
        assert len(counts) == len(columns), f"row {row} has {len(counts)} cells"
        for index, column in enumerate(columns):
            cells[(row, column)] = int(counts[index])
    return cells


def test_independently_split_roots_are_flagged_as_leaking(voc_root_factory):
    raw = voc_root_factory("raw")
    preprocessed = voc_root_factory("preprocessed")
    ids = [f"img_{i:03d}" for i in range(12)]

    # What the unseeded per-root shuffle left behind: each root is internally consistent, and
    # every id one root holds out is an id the other root trains on.
    write_splits(raw, ids[:8], ids[8:10], ids[10:])
    write_splits(preprocessed, ids[4:], ids[:2], ids[2:4])

    report = check_leakage({"raw": raw, "pre": preprocessed})
    counted = {(overlap.left, overlap.right): overlap.count for overlap in report.overlaps}
    examples = {(overlap.left, overlap.right): overlap.examples for overlap in report.overlaps}

    assert report.has_leakage is True
    assert counted == {
        ("raw/train", "pre/val"): 2,
        ("raw/train", "pre/test"): 2,
        ("pre/train", "raw/val"): 2,
        ("pre/train", "raw/test"): 2,
    }
    assert examples[("raw/train", "pre/val")] == ("img_000", "img_001")
    assert examples[("pre/train", "raw/test")] == ("img_010", "img_011")


def test_one_split_written_to_both_roots_is_not_flagged(voc_root_factory):
    raw = voc_root_factory("raw")
    preprocessed = voc_root_factory("preprocessed")
    ids = [f"img_{i:03d}" for i in range(12)]

    # The fix for B2: generate once, propagate the same lists to every root.
    shared = (ids[:8], ids[8:10], ids[10:])
    write_splits(raw, *shared)
    write_splits(preprocessed, *shared)

    report = check_leakage({"raw": raw, "pre": preprocessed})

    assert report.has_leakage is False
    assert report.overlaps == []


def test_a_root_that_leaks_into_itself_is_flagged(voc_root):
    ids = [f"img_{i:03d}" for i in range(12)]
    write_splits(voc_root, ids[:8], ids[6:10], ids[10:])

    report = check_leakage({"raw": voc_root})
    overlaps = {(overlap.left, overlap.right): overlap.count for overlap in report.overlaps}

    assert report.has_leakage is True
    assert overlaps[("raw/train", "raw/val")] == 2


def test_custom_split_names_decide_what_counts_as_training(voc_root_factory):
    root = voc_root_factory("raw")
    ids = [f"img_{i:03d}" for i in range(12)]
    write_splits(root, ids[:8], ids[8:10], ids[6:])

    default = check_leakage({"raw": root})
    swapped = check_leakage({"raw": root}, train_names=("test",), eval_names=("train",))

    assert {(o.left, o.right): o.count for o in default.overlaps} == {("raw/train", "raw/test"): 2}
    assert {(o.left, o.right): o.count for o in swapped.overlaps} == {("raw/test", "raw/train"): 2}


def test_examples_are_capped_but_the_count_is_not(voc_root_factory):
    root = voc_root_factory("raw")
    ids = [f"img_{i:03d}" for i in range(12)]
    write_splits(root, ids[:10], ids[:7], ids[10:])

    report = check_leakage({"raw": root})
    overlap = report.overlaps[0]

    assert overlap.count == 7
    assert overlap.examples == tuple(ids[:MAX_EXAMPLES])
    # The listing has to admit it is truncated, or seven shared ids read as five.
    assert "share 7 ids: img_000, img_001, img_002, img_003, img_004, ..." in report.format()


def test_format_names_both_offending_splits(voc_root_factory):
    raw = voc_root_factory("raw")
    preprocessed = voc_root_factory("preprocessed")
    ids = [f"img_{i:03d}" for i in range(12)]
    write_splits(raw, ids[:8], ids[8:10], ids[10:])
    write_splits(preprocessed, ids[4:], ids[:2], ids[2:4])

    text = check_leakage({"raw": raw, "pre": preprocessed}).format()
    sections = text.split("LEAKAGE")
    assert len(sections) == 2, "format() printed no LEAKAGE section"
    offenders = [line for line in sections[1].splitlines() if line.strip()]

    assert any("raw/train" in line and "pre/val" in line and "2 ids" in line for line in offenders)
    assert any("pre/train" in line and "raw/test" in line and "2 ids" in line for line in offenders)

    # The matrix is the evidence: the shared count sits in its cell, and a pair that was compared
    # and came back clean shows as a zero rather than as a gap.
    matrix = read_matrix(text)
    assert matrix[("raw/train", "pre/val")] == 2
    assert matrix[("pre/train", "raw/test")] == 2
    assert matrix[("raw/train", "raw/val")] == 0


def test_a_report_built_from_overlaps_alone_still_prints_a_matrix():
    # The spec's two-field constructor: no record of which pairs were compared, so the matrix is
    # drawn from the offending pairs. It must still appear.
    report = LeakageReport(
        {"raw/train": 8, "pre/val": 2},
        [Overlap("raw/train", "pre/val", 2, ("img_000", "img_001"))],
    )

    assert read_matrix(report.format())[("raw/train", "pre/val")] == 2


def test_format_of_a_clean_pair_says_so(voc_root_factory):
    report = check_leakage({"a": voc_root_factory("a"), "b": voc_root_factory("b")})
    text = report.format()

    assert report.has_leakage is False
    assert "LEAKAGE" not in text
    assert "no training id appears in any evaluation split" in text
    assert read_matrix(text)[("a/train", "b/val")] == 0


def test_collect_splits_keys_and_sizes(voc_root):
    splits = collect_splits({"raw": voc_root})

    assert set(splits) == {"raw/train", "raw/val", "raw/test"}
    assert len(splits["raw/train"]) == 8
    assert sum(len(ids) for ids in splits.values()) == 12


def test_collect_splits_skips_a_split_the_root_does_not_have(voc_root):
    (voc_root / "ImageSets" / "Main" / "test.txt").unlink()
    splits = collect_splits({"raw": voc_root})

    assert set(splits) == {"raw/train", "raw/val"}


def test_collect_splits_raises_for_a_root_without_splits(tmp_path):
    with pytest.raises(FileNotFoundError, match="split directory not found"):
        collect_splits({"raw": tmp_path / "nothing"})


def test_report_round_trips_through_its_dict(voc_root_factory):
    raw = voc_root_factory("raw")
    preprocessed = voc_root_factory("preprocessed")
    ids = [f"img_{i:03d}" for i in range(12)]
    write_splits(raw, ids[:8], ids[8:10], ids[10:])
    write_splits(preprocessed, ids[6:], ids[:3], ids[3:6])

    report = check_leakage({"raw": raw, "pre": preprocessed})
    payload = json.loads(json.dumps(report.to_dict()))

    assert payload["has_leakage"] is True
    assert payload["splits"]["raw/train"] == 8
    # The examples must be ids that really are in both splits, not arbitrary ones.
    assert payload["overlaps"][0]["left"] == "raw/train"
    assert payload["overlaps"][0]["right"] == "pre/val"
    assert payload["overlaps"][0]["examples"] == ["img_000", "img_001", "img_002"]

    restored = LeakageReport(
        payload["splits"],
        [
            Overlap(item["left"], item["right"], item["count"], tuple(item["examples"]))
            for item in payload["overlaps"]
        ],
        tuple(tuple(pair) for pair in payload["checked"]),
    )
    assert restored.format() == report.format()


def test_audit_counts_every_class_in_a_clean_root(voc_root):
    report = audit_annotations(voc_root)

    assert report.n_images == 12
    assert report.n_boxes == 18
    assert report.class_counts == {"object": 12, "shadow": 6}
    assert report.is_clean is True
    assert report.empty_images == []


def test_audit_finds_degenerate_and_out_of_bounds_boxes(voc_root):
    annotations = voc_root / "Annotations"
    make_xml(annotations / "img_000.xml", WIDTH, HEIGHT, [("object", 10, 10, 10, 20)])
    make_xml(
        annotations / "img_001.xml",
        WIDTH,
        HEIGHT,
        [("object", 4, 4, 12, 12), ("shadow", 30, 30, WIDTH + 20, HEIGHT + 5)],
    )

    report = audit_annotations(voc_root)

    assert report.degenerate == [("img_000", 1)]
    assert report.out_of_bounds == [("img_001", 1)]
    assert report.is_clean is False
    # A zero-width box leaves its image with nothing to learn from.
    assert "img_000" in report.empty_images
    # Faulty boxes still count towards the totals; the fault lists say how many are unusable.
    assert report.n_boxes == 19
    assert report.class_counts == {"object": 12, "shadow": 7}


def test_audit_flags_a_negative_coordinate(voc_root):
    annotations = voc_root / "Annotations"
    make_xml(annotations / "img_000.xml", WIDTH, HEIGHT, [("object", -4, 10, 12, 20)])
    make_xml(annotations / "img_001.xml", WIDTH, HEIGHT, [("object", 10, -4, 20, 12)])

    report = audit_annotations(voc_root)

    assert report.out_of_bounds == [("img_000", 1), ("img_001", 1)]
    assert report.is_clean is False


def test_audit_puts_the_frame_edge_inside_the_image(voc_root):
    annotations = voc_root / "Annotations"
    make_xml(annotations / "img_000.xml", WIDTH, HEIGHT, [("object", 0, 0, WIDTH, HEIGHT)])
    make_xml(annotations / "img_001.xml", WIDTH, HEIGHT, [("object", 4, 4, WIDTH + 1, 12)])
    make_xml(annotations / "img_002.xml", WIDTH, HEIGHT, [("object", 4, 4, 12, HEIGHT + 1)])

    report = audit_annotations(voc_root)

    # A box ending exactly at width or height covers the last pixel; only past it is a fault.
    assert report.out_of_bounds == [("img_001", 1), ("img_002", 1)]
    assert "img_000" not in report.empty_images


def test_audit_reports_unknown_class_names(voc_root):
    make_xml(
        voc_root / "Annotations" / "img_002.xml",
        WIDTH,
        HEIGHT,
        [("seabed", 4, 4, 12, 12), ("seabed", 20, 20, 28, 28)],
    )

    report = audit_annotations(voc_root)

    assert report.unknown_classes == {"seabed": 2}
    assert report.class_counts["object"] == 11
    assert report.is_clean is False
    assert "img_002" in report.empty_images


def test_audit_treats_an_empty_annotation_as_empty_not_dirty(voc_root_factory):
    root = voc_root_factory("bg")
    make_xml(root / "Annotations" / "img_003.xml", WIDTH, HEIGHT, [])

    report = audit_annotations(root)

    assert report.empty_images == ["img_003"]
    assert report.is_clean is True


def test_audit_restricts_itself_to_the_given_ids(voc_root):
    report = audit_annotations(voc_root, ids=["img_000", "img_006"])

    assert report.n_images == 2
    assert report.class_counts == {"object": 2, "shadow": 1}


def test_audit_raises_for_a_missing_annotation(voc_root):
    with pytest.raises(FileNotFoundError, match="img_999"):
        audit_annotations(voc_root, ids=["img_999"])


def test_format_lists_the_faults(voc_root):
    make_xml(voc_root / "Annotations" / "img_004.xml", WIDTH, HEIGHT, [("object", 5, 5, 5, 9)])

    text = audit_annotations(voc_root).format()

    assert "12 images" in text
    assert "degenerate boxes" in text
    assert "img_004" in text
    assert "NOT CLEAN" in text
