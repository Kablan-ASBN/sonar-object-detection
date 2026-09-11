from __future__ import annotations

import random
import xml.etree.ElementTree as ET
from pathlib import Path

import cv2
import numpy as np
import pytest

from sonar.data.preprocess import (
    AugmentParams,
    apply_augment,
    apply_clahe,
    build_variant,
    median_denoise,
    sample_augment,
)


def read_annotation(path: Path) -> tuple[list[str], np.ndarray]:
    root = ET.parse(path).getroot()
    names, boxes = [], []
    for obj in root.findall("object"):
        bndbox = obj.find("bndbox")
        names.append(obj.findtext("name"))
        boxes.append([float(bndbox.findtext(t)) for t in ("xmin", "ymin", "xmax", "ymax")])
    return names, np.asarray(boxes, dtype=np.float32).reshape(-1, 4)


def mean_in_box(image: np.ndarray, box: np.ndarray) -> float:
    x1, y1, x2, y2 = (round(float(v)) for v in box)
    return float(image[y1:y2, x1:x2].mean())


def blob_image(
    width: int = 64,
    height: int = 48,
    corner: tuple[int, int] = (20, 16),
) -> tuple[np.ndarray, np.ndarray]:
    """A dark tile with one bright square, plus the box that describes the square."""
    canvas = np.full((height, width, 3), 20, dtype=np.uint8)
    x, y = corner
    canvas[y : y + 8, x : x + 8] = 240
    return canvas, np.array([[x, y, x + 8, y + 8]], dtype=np.float32)


def write_source(
    root: Path,
    image: np.ndarray,
    objects: list[tuple[str, tuple[int, int, int, int]]],
    stem: str = "img_000",
) -> Path:
    """A one-image VOC root, for the two cases the shared fixture cannot express."""
    (root / "JPEGImages").mkdir(parents=True, exist_ok=True)
    (root / "Annotations").mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(root / "JPEGImages" / f"{stem}.jpg"), image, [cv2.IMWRITE_JPEG_QUALITY, 95])

    annotation = ET.Element("annotation")
    ET.SubElement(annotation, "filename").text = f"{stem}.jpg"
    for name, box in objects:
        obj = ET.SubElement(annotation, "object")
        ET.SubElement(obj, "name").text = name
        bndbox = ET.SubElement(obj, "bndbox")
        for tag, value in zip(("xmin", "ymin", "xmax", "ymax"), box):
            ET.SubElement(bndbox, tag).text = str(value)
    (root / "Annotations" / f"{stem}.xml").write_text(ET.tostring(annotation, encoding="unicode"))
    return root


def draws(seed: int, n: int = 3, **kwargs: float) -> list[AugmentParams]:
    """`n` consecutive draws from one seeded rng, the way build_variant consumes them."""
    rng = random.Random(seed)
    return [sample_augment(rng, **kwargs) for _ in range(n)]


def test_median_denoise_preserves_shape_and_dtype():
    image = np.random.randint(0, 256, size=(48, 64, 3), dtype=np.uint8)
    out = median_denoise(image)
    assert out.shape == image.shape
    assert out.dtype == np.uint8


def test_median_denoise_removes_isolated_salt_and_pepper():
    image = np.full((32, 32, 3), 60, dtype=np.uint8)
    image[10, 10] = 255
    image[20, 20] = 0
    out = median_denoise(image, ksize=3)
    assert out[10, 10].tolist() == [60, 60, 60]
    assert out[20, 20].tolist() == [60, 60, 60]
    assert np.array_equal(out, np.full_like(image, 60))


def test_median_denoise_kernel_size_sets_the_reach():
    image = np.full((32, 32, 3), 60, dtype=np.uint8)
    image[10:13, 10:13] = 255
    # A 3x3 clump fills a 3x3 window, so only the wider kernel outvotes it.
    assert median_denoise(image, ksize=3)[11, 11].tolist() == [255, 255, 255]
    assert median_denoise(image, ksize=5)[11, 11].tolist() == [60, 60, 60]


def test_median_denoise_rejects_even_kernel():
    with pytest.raises(ValueError, match="odd"):
        median_denoise(np.zeros((8, 8, 3), dtype=np.uint8), ksize=4)


def test_apply_clahe_preserves_shape_and_dtype():
    image = np.random.randint(0, 256, size=(48, 64, 3), dtype=np.uint8)
    out = apply_clahe(image)
    assert out.shape == image.shape
    assert out.dtype == np.uint8


def test_apply_clahe_leaves_a_uniform_image_uniform():
    image = np.full((64, 64, 3), 128, dtype=np.uint8)
    out = apply_clahe(image)
    # Equalising a flat histogram may shift the level slightly, but must not invent structure.
    assert float(out.std()) == 0.0
    assert np.abs(out.astype(np.int16) - 128).max() <= 8


def test_apply_clahe_expands_low_contrast_range():
    row = np.linspace(100, 130, 64, dtype=np.uint8)
    image = np.repeat(np.tile(row, (64, 1))[:, :, None], 3, axis=2)
    assert apply_clahe(image).std() > image.std()


def test_apply_clahe_equalises_luminance_and_leaves_chroma_alone():
    image = np.random.default_rng(0).integers(60, 150, size=(48, 64, 3), dtype=np.uint8)
    before = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.int16)
    after = cv2.cvtColor(apply_clahe(image), cv2.COLOR_BGR2LAB).astype(np.int16)
    assert after[:, :, 0].std() > 2.0 * before[:, :, 0].std()
    # Equalising B, G and R independently passes the line above and moves a/b by tens of levels.
    assert np.abs(after[:, :, 1:] - before[:, :, 1:]).max() <= 4


def test_flipping_twice_restores_image_and_boxes():
    image, boxes = blob_image()
    flip = AugmentParams(hflip=True, angle_deg=0.0, brightness=0.0, contrast=1.0)
    once_image, once_boxes = apply_augment(image, boxes, flip)
    twice_image, twice_boxes = apply_augment(once_image, once_boxes, flip)
    assert not np.array_equal(once_boxes, boxes)
    assert np.array_equal(twice_boxes, boxes)
    assert np.array_equal(twice_image, image)


def test_flipped_box_matches_flipped_pixels():
    image, boxes = blob_image()
    width = image.shape[1]
    flipped, moved = apply_augment(image, boxes, AugmentParams(True, 0.0, 0.0, 1.0))
    assert moved[0, 0] == width - boxes[0, 2]
    assert moved[0, 2] == width - boxes[0, 0]
    assert mean_in_box(flipped, moved[0]) == pytest.approx(mean_in_box(image, boxes[0]))


def test_box_follows_its_content_through_rotation():
    image, boxes = blob_image()
    rotated, moved = apply_augment(image, boxes, AugmentParams(False, 10.0, 0.0, 1.0))

    bright = np.argwhere(rotated[:, :, 0] > 180)
    assert len(bright) > 40
    cy, cx = bright.mean(axis=0)
    x1, y1, x2, y2 = moved[0]
    assert x1 <= cx <= x2
    assert y1 <= cy <= y2
    # The hull of a rotated square is strictly larger than the square.
    assert (x2 - x1) > (boxes[0, 2] - boxes[0, 0])


def test_rotating_180_degrees_agrees_with_flipping_both_axes():
    """Pins the two conventions a centroid test cannot see: rotation centre and half-pixel shift."""
    image, boxes = blob_image()
    height, width = image.shape[:2]
    x1, y1, x2, y2 = boxes[0]
    rotated, moved = apply_augment(image, boxes, AugmentParams(False, 180.0, 0.0, 1.0))

    assert np.array_equal(rotated, cv2.flip(image, -1))
    # Both routes are edge-coordinate reflections, so they must agree to the last decimal.
    assert moved[0].tolist() == pytest.approx([width - x2, height - y2, width - x1, height - y1])


def test_augment_keeps_boxes_on_content_where_flipping_pixels_alone_does_not(voc_root):
    """Regression: the original pipeline moved pixels and left the annotation behind."""
    image = cv2.imread(str(voc_root / "JPEGImages" / "img_000.jpg"), cv2.IMREAD_COLOR)
    _, boxes = read_annotation(voc_root / "Annotations" / "img_000.xml")

    before = mean_in_box(image, boxes[0])
    buggy = mean_in_box(cv2.flip(image, 1), boxes[0])
    fixed_image, fixed_boxes = apply_augment(image, boxes, AugmentParams(True, 0.0, 0.0, 1.0))
    fixed = mean_in_box(fixed_image, fixed_boxes[0])

    assert buggy < before - 100
    assert fixed == pytest.approx(before, abs=2.0)


def test_apply_augment_returns_one_box_per_input_in_order():
    image, _ = blob_image()
    boxes = np.array(
        [[2.0, 2.0, 10.0, 10.0], [30.0, 20.0, 40.0, 30.0], [58.0, 40.0, 63.0, 47.0]],
        dtype=np.float32,
    )
    _, moved = apply_augment(image, boxes, AugmentParams(True, 8.0, 5.0, 1.05))
    assert moved.shape == (3, 4)
    # Order is positional, so the left-most input must still be the right-most output after a flip.
    assert moved[0, 0] > moved[2, 0]


def test_apply_augment_accepts_an_empty_box_array():
    image, _ = blob_image()
    empty = np.zeros((0, 4), dtype=np.float32)
    out, moved = apply_augment(image, empty, AugmentParams(True, 7.0, 0.0, 1.0))
    assert moved.shape == (0, 4)
    assert out.shape == image.shape


def test_augmented_boxes_stay_inside_the_frame():
    image, _ = blob_image()
    height, width = image.shape[:2]
    boxes = np.array([[0.0, 0.0, 64.0, 48.0], [60.0, 44.0, 64.0, 48.0]], dtype=np.float32)
    _, moved = apply_augment(image, boxes, AugmentParams(True, 10.0, 0.0, 1.0))
    assert moved[:, 0::2].min() >= 0.0
    assert moved[:, 1::2].min() >= 0.0
    assert moved[:, 0::2].max() <= width
    assert moved[:, 1::2].max() <= height


def test_sample_augment_is_reproducible_from_a_seeded_rng():
    first = draws(0)
    assert first == draws(0)
    assert first != draws(1)
    # build_variant draws once per image from one rng, so consecutive draws must not repeat.
    assert len({p.angle_deg for p in first}) == 3


def test_sample_augment_honours_max_angle():
    angles = [abs(p.angle_deg) for p in draws(0, n=40, max_angle=2.0)]
    assert max(angles) <= 2.0
    # Guards against an angle that ignores the bound by never leaving zero.
    assert max(angles) > 1.0


def test_build_variant_rejects_an_unknown_mode(voc_root, tmp_path):
    with pytest.raises(ValueError, match="unknown mode"):
        build_variant(voc_root, tmp_path / "out", mode="sharpen")


def test_build_variant_rejects_a_source_with_no_jpegs(tmp_path):
    """A root of .png files used to write nothing and report success."""
    (tmp_path / "src" / "JPEGImages").mkdir(parents=True)
    with pytest.raises(FileNotFoundError, match=r"no \.jpg images"):
        build_variant(tmp_path / "src", tmp_path / "out", mode="denoised")


def test_build_variant_denoised_copies_structure(voc_root, tmp_path):
    dst = tmp_path / "denoised"
    written = build_variant(voc_root, dst, mode="denoised")

    jpegs = sorted(p.stem for p in (dst / "JPEGImages").glob("*.jpg"))
    xmls = sorted(p.stem for p in (dst / "Annotations").glob("*.xml"))
    assert written == len(jpegs) == len(xmls) == 12
    assert jpegs == xmls
    for name in ("train", "val", "test"):
        copied = dst / "ImageSets" / "Main" / f"{name}.txt"
        assert copied.read_text() == (voc_root / "ImageSets" / "Main" / f"{name}.txt").read_text()

    # Denoising moves no content, so the boxes must come across unchanged.
    _, before = read_annotation(voc_root / "Annotations" / "img_007.xml")
    _, after = read_annotation(dst / "Annotations" / "img_007.xml")
    assert np.array_equal(before, after)


def test_build_variant_denoised_filters_the_pixels_it_writes(tmp_path):
    """Structure alone is satisfied by a re-encoded copy; speckle is what separates the two."""
    speckles = ((6, 6), (6, 40), (40, 10), (30, 50), (12, 55))
    canvas = np.full((48, 64, 3), 20, dtype=np.uint8)
    canvas[16:24, 20:28] = 230
    for y, x in speckles:
        canvas[y, x] = 255
    src = write_source(tmp_path / "src", canvas, [("object", (20, 16, 28, 24))])

    assert build_variant(src, tmp_path / "out", mode="denoised") == 1
    source = cv2.imread(str(src / "JPEGImages" / "img_000.jpg"), cv2.IMREAD_COLOR)
    written = cv2.imread(str(tmp_path / "out" / "JPEGImages" / "img_000.jpg"), cv2.IMREAD_COLOR)

    assert all(source[y, x, 0] > 200 for y, x in speckles)
    assert all(written[y, x, 0] < 60 for y, x in speckles)
    # The blob is wider than the kernel, so real content has to survive the filter.
    assert mean_in_box(written, np.array([20, 16, 28, 24])) > 200


def test_build_variant_clahe_augmented_keeps_boxes_on_content(voc_root, tmp_path):
    dst = tmp_path / "augmented"
    written = build_variant(voc_root, dst, mode="clahe_augmented", seed=7)
    assert written == 12
    assert len(list((dst / "Annotations").glob("*.xml"))) == written

    checked = 0
    for xml_path in sorted((dst / "Annotations").glob("*.xml")):
        image = cv2.imread(str(dst / "JPEGImages" / f"{xml_path.stem}.jpg"), cv2.IMREAD_COLOR)
        names, boxes = read_annotation(xml_path)
        means = {}
        for name, box in zip(names, boxes):
            assert box[2] > box[0] and box[3] > box[1]
            means[name] = mean_in_box(image, box)
            checked += 1
        assert means["object"] > image.mean() + 60
        if "shadow" in means:
            # The half-intensity square must be brighter than background and darker than the object.
            assert means["shadow"] > image.mean() + 40
            assert means["object"] > means["shadow"] + 30
    assert checked == 18


def test_build_variant_clahe_augmented_matches_the_pipeline_it_documents(voc_root, tmp_path):
    """Denoise, CLAHE, then one seeded augmentation - replayed by hand for the first image."""
    dst = tmp_path / "augmented"
    build_variant(voc_root, dst, mode="clahe_augmented", seed=13)

    source = cv2.imread(str(voc_root / "JPEGImages" / "img_000.jpg"), cv2.IMREAD_COLOR)
    _, boxes = read_annotation(voc_root / "Annotations" / "img_000.xml")
    # img_000 sorts first, so it takes the first draw from the seeded rng.
    params = sample_augment(random.Random(13))
    expected, expected_boxes = apply_augment(apply_clahe(median_denoise(source)), boxes, params)

    written = cv2.imread(str(dst / "JPEGImages" / "img_000.jpg"), cv2.IMREAD_COLOR)
    assert np.abs(written.astype(np.int16) - expected.astype(np.int16)).mean() < 1.0
    # Guards the tolerance: skipping a stage moves pixels ten times further than JPEG does.
    assert np.abs(source.astype(np.int16) - expected.astype(np.int16)).mean() > 10.0

    _, written_boxes = read_annotation(dst / "Annotations" / "img_000.xml")
    assert written_boxes.tolist() == np.round(expected_boxes).tolist()
    # Three of the four edges land past .5 here, so truncating instead of rounding would show.
    assert not np.array_equal(np.round(expected_boxes), np.trunc(expected_boxes))


def test_build_variant_drops_a_box_that_collapses(tmp_path):
    canvas = np.full((48, 64, 3), 20, dtype=np.uint8)
    canvas[16:24, 20:28] = 230
    objects = [("object", (20, 16, 28, 24)), ("collapsed", (40, 40, 40, 40))]
    src = write_source(tmp_path / "src", canvas, objects)

    build_variant(src, tmp_path / "out", mode="clahe_augmented", seed=5)
    names, boxes = read_annotation(tmp_path / "out" / "Annotations" / "img_000.xml")
    # A zero-area box cannot survive the hull, and its name has to leave with it.
    assert names == ["object"]
    assert boxes.shape == (1, 4)


def test_build_variant_is_deterministic_for_a_seed(voc_root, tmp_path):
    first = tmp_path / "a"
    second = tmp_path / "b"
    build_variant(voc_root, first, mode="clahe_augmented", seed=3)
    build_variant(voc_root, second, mode="clahe_augmented", seed=3)
    for xml_path in sorted((first / "Annotations").glob("*.xml")):
        # The <folder> field records the root name, so compare the geometry rather than the bytes.
        assert read_annotation(xml_path)[1].tolist() == (
            read_annotation(second / "Annotations" / xml_path.name)[1].tolist()
        )


def test_build_variant_geometry_changes_with_the_seed(voc_root, tmp_path):
    """Determinism on its own is also what no augmentation at all would give you."""
    build_variant(voc_root, tmp_path / "three", mode="clahe_augmented", seed=3)
    build_variant(voc_root, tmp_path / "four", mode="clahe_augmented", seed=4)
    differing = [
        xml_path.name
        for xml_path in sorted((tmp_path / "three" / "Annotations").glob("*.xml"))
        if read_annotation(xml_path)[1].tolist()
        != read_annotation(tmp_path / "four" / "Annotations" / xml_path.name)[1].tolist()
    ]
    assert len(differing) >= 10
