"""Tests for the YOLO to Pascal VOC converter."""

from __future__ import annotations

import random
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from sonar.data.convert import ConversionStats, _voc_xml, convert_yolo_to_voc, yolo_box_to_voc

WIDTH, HEIGHT = 64, 48
BACKGROUND, FOREGROUND = 20, 230


def yolo_row(
    class_id: int,
    xmin: float,
    ymin: float,
    xmax: float,
    ymax: float,
    width: int = WIDTH,
    height: int = HEIGHT,
) -> str:
    """Express a pixel box as the normalised YOLO row a labelling tool would have written."""
    cx = (xmin + xmax) / 2 / width
    cy = (ymin + ymax) / 2 / height
    return f"{class_id} {cx} {cy} {(xmax - xmin) / width} {(ymax - ymin) / height}"


def write_image(path: Path, boxes: tuple[tuple[int, int, int, int], ...] = ()) -> None:
    canvas = np.full((HEIGHT, WIDTH, 3), BACKGROUND, dtype=np.uint8)
    for xmin, ymin, xmax, ymax in boxes:
        canvas[ymin:ymax, xmin:xmax] = FOREGROUND
    Image.fromarray(canvas).save(path)


def make_dirs(tmp_path: Path) -> tuple[Path, Path, Path]:
    images_dir = tmp_path / "images"
    labels_dir = tmp_path / "labels"
    images_dir.mkdir()
    labels_dir.mkdir()
    return images_dir, labels_dir, tmp_path / "voc"


def test_known_yolo_box_maps_to_expected_pixels():
    assert yolo_box_to_voc((0.5, 0.5, 0.5, 0.5), 100, 100) == (25.0, 25.0, 75.0, 75.0)


def test_rounding_beats_truncation_on_a_half_pixel_centre():
    box = (0.505, 0.5, 0.5, 0.5)
    exact_xmin = (box[0] - box[2] / 2) * 100

    truncated = int(exact_xmin)  # what the original convert_box() produced
    rounded = yolo_box_to_voc(box, 100, 100)[0]

    assert truncated == 25
    assert rounded == 26


def test_truncation_bias_toward_the_top_left_is_gone():
    # The original int() cast never rounded up, so across a dataset every coordinate sat
    # about half a pixel low. Averaging the signed error exposes that as a systematic shift.
    rng = random.Random(0)
    truncation_error = []
    rounding_error = []
    for _ in range(400):
        w = rng.uniform(0.05, 0.4)
        h = rng.uniform(0.05, 0.4)
        cx = rng.uniform(w / 2, 1 - w / 2)
        cy = rng.uniform(h / 2, 1 - h / 2)
        exact = ((cx - w / 2) * WIDTH, (cy - h / 2) * HEIGHT)
        converted = yolo_box_to_voc((cx, cy, w, h), WIDTH, HEIGHT)
        truncation_error += [int(exact[0]) - exact[0], int(exact[1]) - exact[1]]
        rounding_error += [converted[0] - exact[0], converted[1] - exact[1]]

    assert np.mean(truncation_error) < -0.4
    assert abs(np.mean(rounding_error)) < 0.05


def test_boxes_are_clamped_inside_the_image():
    xmin, ymin, xmax, ymax = yolo_box_to_voc((0.5, 0.5, 2.0, 2.0), WIDTH, HEIGHT)
    assert (xmin, ymin) == (0.0, 0.0)
    assert (xmax, ymax) == (WIDTH - 1, HEIGHT - 1)


def test_yolo_box_of_wrong_length_raises():
    with pytest.raises(ValueError, match="4 values"):
        yolo_box_to_voc((0.5, 0.5, 0.5), WIDTH, HEIGHT)


def test_empty_label_file_is_skipped_by_default(tmp_path):
    images_dir, labels_dir, out_root = make_dirs(tmp_path)
    write_image(images_dir / "blank.jpg")
    (labels_dir / "blank.txt").write_text("")

    stats = convert_yolo_to_voc(images_dir, labels_dir, out_root, workers=2)

    assert stats == ConversionStats(converted=0, skipped_empty=1)
    assert not list((out_root / "Annotations").glob("*.xml"))


def test_keep_empty_writes_an_annotation_with_no_objects(tmp_path):
    # The original dropped background-only tiles outright, which is the behaviour
    # keep_empty=False still reproduces; the two must not agree.
    images_dir, labels_dir, out_root = make_dirs(tmp_path)
    write_image(images_dir / "blank.jpg")
    (labels_dir / "blank.txt").write_text("")

    dropped = convert_yolo_to_voc(images_dir, labels_dir, out_root / "dropped", workers=2)
    kept = convert_yolo_to_voc(
        images_dir, labels_dir, out_root / "kept", keep_empty=True, workers=2
    )

    assert dropped.converted == 0 and kept.converted == 1
    assert kept.boxes_written == 0
    assert (out_root / "kept" / "JPEGImages" / "blank.jpg").is_file()
    xml = ET.parse(out_root / "kept" / "Annotations" / "blank.xml").getroot()
    assert xml.findall("object") == []
    assert xml.find("size/width").text == str(WIDTH)


def test_stats_account_for_every_label_file(tmp_path):
    images_dir, labels_dir, out_root = make_dirs(tmp_path)
    content = {
        "one": [yolo_row(0, 8, 8, 24, 24)],
        "two": [yolo_row(0, 8, 8, 24, 24), yolo_row(1, 8, 28, 24, 40)],
        "three": [yolo_row(1, 32, 8, 48, 24)],
        "empty": [],
        "no_image": [yolo_row(0, 8, 8, 24, 24)],
        "bad_class_only": ["7 0.5 0.5 0.2 0.2"],
        "short_row": ["0 0.5 0.5", yolo_row(1, 8, 8, 24, 24)],
    }
    for stem, rows in content.items():
        (labels_dir / f"{stem}.txt").write_text("\n".join(rows))
        if stem != "no_image":
            write_image(images_dir / f"{stem}.jpg")

    stats = convert_yolo_to_voc(images_dir, labels_dir, out_root, workers=4)

    assert stats.converted + stats.skipped_no_image + stats.skipped_empty == len(content)
    assert stats.converted == 4
    assert stats.skipped_no_image == 1
    assert stats.skipped_empty == 2
    assert stats.skipped_bad_class == 1
    assert stats.boxes_written == 5
    assert len(list((out_root / "Annotations").glob("*.xml"))) == stats.converted
    assert len(list((out_root / "JPEGImages").glob("*.jpg"))) == stats.converted


def test_out_of_range_class_id_is_counted_but_its_neighbours_survive(tmp_path):
    images_dir, labels_dir, out_root = make_dirs(tmp_path)
    write_image(images_dir / "mixed.jpg")
    (labels_dir / "mixed.txt").write_text(
        "\n".join([yolo_row(0, 8, 8, 24, 24), "5 0.5 0.5 0.2 0.2", yolo_row(1, 32, 8, 48, 24)])
    )

    stats = convert_yolo_to_voc(images_dir, labels_dir, out_root, workers=2)

    assert stats.skipped_bad_class == 1
    assert stats.boxes_written == 2
    names = [n.text for n in ET.parse(out_root / "Annotations" / "mixed.xml").iter("name")]
    assert names == ["object", "shadow"]


def test_stats_match_the_files_on_disk_whatever_the_worker_count(tmp_path):
    # Tiles differ in how many boxes and bad rows they hold, so a counter that tallied files
    # instead of boxes, or folded the bad rows into boxes_written, would show up here.
    images_dir, labels_dir, out_root = make_dirs(tmp_path)
    for i in range(40):
        write_image(images_dir / f"tile_{i:03d}.jpg")
        rows = [yolo_row(0, 8, 8, 24, 24)]
        if i % 2:
            rows.append(yolo_row(1, 8, 28, 24, 40))
        if i % 5 == 0:
            rows.append("7 0.5 0.5 0.2 0.2")
        (labels_dir / f"tile_{i:03d}.txt").write_text("\n".join(rows))

    serial = convert_yolo_to_voc(images_dir, labels_dir, out_root / "serial", workers=1)
    threaded = convert_yolo_to_voc(images_dir, labels_dir, out_root / "threaded", workers=8)

    assert serial == threaded
    assert threaded == ConversionStats(converted=40, skipped_bad_class=8, boxes_written=60)
    annotations = sorted((out_root / "threaded" / "Annotations").glob("*.xml"))
    assert len(annotations) == threaded.converted
    on_disk = sum(len(ET.parse(path).findall("object")) for path in annotations)
    assert on_disk == threaded.boxes_written


def test_stats_str_reports_every_counter():
    text = str(
        ConversionStats(
            converted=3, skipped_no_image=1, skipped_empty=2, skipped_bad_class=4, boxes_written=5
        )
    )

    assert "converted 3 images (5 boxes)" in text
    assert "1 with no image" in text
    assert "2 with no usable objects" in text
    assert "4 rows" in text


def test_boxes_that_clamp_to_no_area_are_dropped_not_counted(tmp_path):
    # A box centred outside the frame clamps to xmax == xmin, which VOCDetection discards, so
    # writing it would leave boxes_written promising ground truth that never loads.
    images_dir, labels_dir, out_root = make_dirs(tmp_path)
    for stem in ("mixed", "off_frame_only"):
        write_image(images_dir / f"{stem}.jpg", boxes=((8, 8, 24, 24),))
    (labels_dir / "mixed.txt").write_text(
        "\n".join(["0 1.5 0.5 0.2 0.2", "1 0.5 0.5 0.0 0.0", yolo_row(0, 8, 8, 24, 24)])
    )
    (labels_dir / "off_frame_only.txt").write_text("0 1.5 0.5 0.2 0.2")

    stats = convert_yolo_to_voc(images_dir, labels_dir, out_root, workers=2)

    assert stats == ConversionStats(converted=1, skipped_empty=1, boxes_written=1)
    names = [n.text for n in ET.parse(out_root / "Annotations" / "mixed.xml").iter("name")]
    assert names == ["object"]
    assert not (out_root / "Annotations" / "off_frame_only.xml").exists()


def test_xml_writer_rounds_rather_than_truncates():
    # yolo_box_to_voc hands over whole pixels today; this pins the writer itself, the other
    # place an int() cast could quietly restore the top-left bias.
    xml = _voc_xml("tile", "voc", WIDTH, HEIGHT, [("object", (10.6, 10.5, 20.4, 20.5))])

    bndbox = ET.fromstring(xml).find("object/bndbox")
    assert [bndbox.find(tag).text for tag in ("xmin", "ymin", "xmax", "ymax")] == [
        "11",
        "11",
        "20",
        "21",
    ]


def test_png_input_is_written_as_jpeg(tmp_path):
    images_dir, labels_dir, out_root = make_dirs(tmp_path)
    write_image(images_dir / "tile.png", boxes=((8, 8, 24, 24),))
    (labels_dir / "tile.txt").write_text(yolo_row(0, 8, 8, 24, 24))

    convert_yolo_to_voc(images_dir, labels_dir, out_root, workers=2)

    written = out_root / "JPEGImages" / "tile.jpg"
    assert written.is_file()
    with Image.open(written) as image:
        assert image.format == "JPEG"
        assert image.size == (WIDTH, HEIGHT)


@pytest.mark.parametrize("missing", ["images", "labels"])
def test_missing_input_directory_names_the_path(tmp_path, missing):
    images_dir, labels_dir, out_root = make_dirs(tmp_path)
    absent = tmp_path / "nowhere"
    args = (absent, labels_dir) if missing == "images" else (images_dir, absent)

    # Match the module's own wording: an unguarded images_dir would still raise from the
    # directory scan, which is the same exception type with a far less useful message.
    with pytest.raises(FileNotFoundError, match=r"no such directory: .*nowhere"):
        convert_yolo_to_voc(*args, out_root)


def test_annotations_round_trip_through_vocdetection(tmp_path):
    from sonar.data.voc import VOCDetection

    images_dir, labels_dir, out_root = make_dirs(tmp_path)
    pixel_boxes = {
        "tile_a": [(0, (8, 8, 24, 24))],
        "tile_b": [(0, (8, 8, 24, 24)), (1, (32, 24, 48, 40))],
    }
    for stem, entries in pixel_boxes.items():
        write_image(images_dir / f"{stem}.jpg", boxes=tuple(box for _, box in entries))
        (labels_dir / f"{stem}.txt").write_text(
            "\n".join(yolo_row(class_id, *box) for class_id, box in entries)
        )

    convert_yolo_to_voc(images_dir, labels_dir, out_root, workers=2)
    dataset = VOCDetection(out_root, ids=sorted(pixel_boxes))

    assert len(dataset) == 2
    for index, stem in enumerate(sorted(pixel_boxes)):
        image, target = dataset[index]
        expected = [list(box) for _, box in pixel_boxes[stem]]
        assert target["boxes"].tolist() == expected
        assert target["labels"].tolist() == [class_id + 1 for class_id, _ in pixel_boxes[stem]]
        # Geometry is only right if the box still lands on the bright square it was drawn from.
        for xmin, ymin, xmax, ymax in expected:
            patch = image[:, int(ymin) : int(ymax), int(xmin) : int(xmax)]
            assert patch.mean() > 0.7
