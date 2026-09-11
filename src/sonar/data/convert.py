"""YOLO to Pascal VOC conversion for the sidescan tiles.

The original notebook did two things wrong here. It truncated pixel coordinates with `int()`,
which pulled every box half a pixel toward the top-left corner, and it dropped any tile whose
label file produced no objects -- 1,676 of 3,464 images, all of the background-only ones.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from xml.dom.minidom import parseString
from xml.etree.ElementTree import Element, SubElement, tostring

from PIL import Image

IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")


@dataclass
class ConversionStats:
    """Tally for one conversion run.

    `converted + skipped_no_image + skipped_empty` equals the number of input label files.
    `skipped_bad_class` counts individual rows, not files, so it sits outside that sum.
    `boxes_written` counts only boxes that survive clamping with a positive area, so it matches
    what `sonar.data.voc.VOCDetection` will actually load.
    """

    converted: int = 0
    skipped_no_image: int = 0
    skipped_empty: int = 0
    skipped_bad_class: int = 0
    boxes_written: int = 0

    def __str__(self) -> str:
        return (
            f"converted {self.converted} images ({self.boxes_written} boxes), "
            f"skipped {self.skipped_no_image} with no image and "
            f"{self.skipped_empty} with no usable objects, "
            f"ignored {self.skipped_bad_class} rows with an unknown class id"
        )


def yolo_box_to_voc(
    box: Sequence[float], width: int, height: int
) -> tuple[float, float, float, float]:
    """Convert a normalised YOLO (cx, cy, w, h) box to pixel xyxy, clamped to the image."""
    if len(box) != 4:
        raise ValueError(f"expected a YOLO box of 4 values, got {len(box)}")
    if width < 1 or height < 1:
        raise ValueError(f"image size must be positive, got {width}x{height}")

    cx, cy, bw, bh = (float(v) for v in box)
    xmin = _round_half_up((cx - bw / 2) * width)
    ymin = _round_half_up((cy - bh / 2) * height)
    xmax = _round_half_up((cx + bw / 2) * width)
    ymax = _round_half_up((cy + bh / 2) * height)
    return (
        _clamp(xmin, width - 1),
        _clamp(ymin, height - 1),
        _clamp(xmax, width - 1),
        _clamp(ymax, height - 1),
    )


@dataclass
class _Outcome:
    written: bool
    boxes: int = 0
    bad_class: int = 0


def convert_yolo_to_voc(
    images_dir: str | Path,
    labels_dir: str | Path,
    out_root: str | Path,
    *,
    class_names: Sequence[str] = ("object", "shadow"),
    keep_empty: bool = False,
    workers: int = 8,
) -> ConversionStats:
    """Write a VOC dataset root from a YOLO images/labels pair."""
    images_dir = Path(images_dir)
    labels_dir = Path(labels_dir)
    out_root = Path(out_root)
    for directory in (images_dir, labels_dir):
        if not directory.is_dir():
            raise FileNotFoundError(f"no such directory: {directory}")
    if workers < 1:
        raise ValueError(f"workers must be at least 1, got {workers}")

    (out_root / "JPEGImages").mkdir(parents=True, exist_ok=True)
    (out_root / "Annotations").mkdir(parents=True, exist_ok=True)

    names = tuple(class_names)
    index = _index_images(images_dir)
    stats = ConversionStats()

    jobs: list[tuple[Path, Path]] = []
    for label_path in sorted(labels_dir.glob("*.txt")):
        image_path = index.get(label_path.stem)
        if image_path is None:
            stats.skipped_no_image += 1
            continue
        jobs.append((label_path, image_path))

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [
            pool.submit(_convert_one, label_path, image_path, out_root, names, keep_empty)
            for label_path, image_path in jobs
        ]
        # Workers own no shared state and the tally happens here, on one thread, so the
        # original's lock-protected global counters are not needed.
        for future in futures:
            outcome = future.result()
            stats.skipped_bad_class += outcome.bad_class
            if outcome.written:
                stats.converted += 1
                stats.boxes_written += outcome.boxes
            else:
                stats.skipped_empty += 1

    return stats


def _convert_one(
    label_path: Path,
    image_path: Path,
    out_root: Path,
    class_names: tuple[str, ...],
    keep_empty: bool,
) -> _Outcome:
    """Convert one label file plus its image; returns what the caller should count."""
    objects: list[tuple[str, tuple[float, float, float, float]]] = []
    bad_class = 0

    with Image.open(image_path) as handle:
        width, height = handle.size
        rgb = handle.convert("RGB")

    for line in label_path.read_text().splitlines():
        parts = line.split()
        if not parts:
            continue
        if len(parts) != 5:
            # A short or long row is dirty export data rather than a programming error.
            continue
        class_id, coords = _parse_row(parts, label_path)
        if not 0 <= class_id < len(class_names):
            bad_class += 1
            continue
        box = yolo_box_to_voc(coords, width, height)
        if box[2] <= box[0] or box[3] <= box[1]:
            # A box centred outside the frame clamps to zero area, and VOCDetection drops
            # exactly those, so keeping one would make boxes_written overstate the usable data.
            continue
        objects.append((class_names[class_id], box))

    if not objects and not keep_empty:
        return _Outcome(written=False, bad_class=bad_class)

    # Re-encoding at Pillow's default quality smears the low-contrast returns that the
    # shadow class depends on, so keep the loss small.
    rgb.save(out_root / "JPEGImages" / f"{image_path.stem}.jpg", "JPEG", quality=95)
    xml = _voc_xml(image_path.stem, out_root.name, width, height, objects)
    (out_root / "Annotations" / f"{image_path.stem}.xml").write_text(xml)
    return _Outcome(written=True, boxes=len(objects), bad_class=bad_class)


def _parse_row(parts: Sequence[str], label_path: Path) -> tuple[int, list[float]]:
    """Split a YOLO row into class id and coordinates, naming the file if it is corrupt."""
    try:
        return int(float(parts[0])), [float(value) for value in parts[1:]]
    except ValueError as exc:
        raise ValueError(f"{label_path}: cannot parse row {' '.join(parts)!r}") from exc


def _index_images(images_dir: Path) -> dict[str, Path]:
    """Map image stem to path once, instead of rescanning the directory per label file."""
    index: dict[str, Path] = {}
    for path in sorted(images_dir.iterdir()):
        if path.suffix.lower() in IMAGE_SUFFIXES:
            index.setdefault(path.stem, path)
    return index


def _voc_xml(
    stem: str,
    folder: str,
    width: int,
    height: int,
    objects: Sequence[tuple[str, tuple[float, float, float, float]]],
) -> str:
    annotation = Element("annotation")
    SubElement(annotation, "folder").text = folder
    SubElement(annotation, "filename").text = f"{stem}.jpg"
    size = SubElement(annotation, "size")
    SubElement(size, "width").text = str(width)
    SubElement(size, "height").text = str(height)
    SubElement(size, "depth").text = "3"

    for name, (xmin, ymin, xmax, ymax) in objects:
        node = SubElement(annotation, "object")
        SubElement(node, "name").text = name
        SubElement(node, "pose").text = "Unspecified"
        SubElement(node, "truncated").text = "0"
        SubElement(node, "difficult").text = "0"
        bndbox = SubElement(node, "bndbox")
        for tag, value in zip(("xmin", "ymin", "xmax", "ymax"), (xmin, ymin, xmax, ymax)):
            # int() would truncate, restoring the top-left bias this module exists to remove
            # if a caller ever passes sub-pixel coordinates.
            SubElement(bndbox, tag).text = str(_round_half_up(value))

    return parseString(tostring(annotation)).toprettyxml(indent="  ")


def _round_half_up(value: float) -> int:
    # Python's round() is half-to-even, which would still nudge exact .5 coordinates in a
    # size-dependent direction; half-up keeps the error symmetric across a dataset.
    return math.floor(value + 0.5)


def _clamp(value: float, upper: int) -> float:
    return float(min(max(value, 0.0), float(upper)))
