"""Structural audit of a VOC annotation directory.

The conversion from the original YOLO labels produced zero-area boxes, boxes outside the frame
they claim, class names nothing maps to, and images left with no usable object. None of the four
raise an error at training time; they quietly become bad supervision.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

# Mirrors sonar.data.voc.CLASS_MAP, kept local so the audit runs without importing torch.
KNOWN_CLASSES = frozenset({"object", "shadow"})
MAX_LISTED = 5


@dataclass
class AnnotationReport:
    """What one annotation directory contains, and what is wrong with it."""

    n_images: int
    n_boxes: int
    class_counts: dict[str, int]
    degenerate: list[tuple[str, int]]
    out_of_bounds: list[tuple[str, int]]
    empty_images: list[str]
    unknown_classes: dict[str, int]

    @property
    def is_clean(self) -> bool:
        # Empty images are not a fault: background-only tiles are legitimate training data, and
        # the reader represents them with zero boxes rather than a background-class box.
        return not (self.degenerate or self.out_of_bounds or self.unknown_classes)

    def format(self) -> str:
        """One-screen summary, with the first few offending images named."""
        classes = ", ".join(f"{name}={count}" for name, count in sorted(self.class_counts.items()))
        lines = [
            f"{self.n_images} images, {self.n_boxes} boxes",
            f"classes: {classes or 'none'}",
        ]
        lines += _listing("degenerate boxes", self.degenerate)
        lines += _listing("out-of-bounds boxes", self.out_of_bounds)

        if self.unknown_classes:
            unknown = ", ".join(
                f"{name or '<empty>'}={count}"
                for name, count in sorted(self.unknown_classes.items())
            )
            lines.append(f"unknown class names: {unknown}")

        if self.empty_images:
            shown = ", ".join(self.empty_images[:MAX_LISTED])
            more = ", ..." if len(self.empty_images) > MAX_LISTED else ""
            lines.append(f"images with no usable object: {len(self.empty_images)} ({shown}{more})")

        lines.append("clean" if self.is_clean else "NOT CLEAN")
        return "\n".join(lines)


def _listing(title: str, entries: list[tuple[str, int]]) -> list[str]:
    if not entries:
        return []
    total = sum(count for _, count in entries)
    shown = ", ".join(f"{image_id} x{count}" for image_id, count in entries[:MAX_LISTED])
    more = ", ..." if len(entries) > MAX_LISTED else ""
    return [f"{title}: {total} in {len(entries)} images ({shown}{more})"]


def _image_size(root: ET.Element, path: Path) -> tuple[int, int]:
    size = root.find("size")
    width = size.findtext("width") if size is not None else None
    height = size.findtext("height") if size is not None else None
    if width is None or height is None:
        raise ValueError(f"{path}: <size> is missing width or height")
    return int(float(width)), int(float(height))


def _box(obj: ET.Element, path: Path) -> tuple[float, float, float, float]:
    bndbox = obj.find("bndbox")
    if bndbox is None:
        raise ValueError(f"{path}: <object> has no <bndbox>")
    corners = []
    for tag in ("xmin", "ymin", "xmax", "ymax"):
        text = bndbox.findtext(tag)
        if text is None:
            raise ValueError(f"{path}: <bndbox> has no <{tag}>")
        corners.append(float(text))
    return corners[0], corners[1], corners[2], corners[3]


def audit_annotations(root: str | Path, ids: Sequence[str] | None = None) -> AnnotationReport:
    """Audit every annotation under `root`, or only those named by `ids`."""
    directory = Path(root) / "Annotations"
    if not directory.is_dir():
        raise FileNotFoundError(f"annotation directory not found: {directory}")

    if ids is None:
        paths = sorted(directory.glob("*.xml"))
    else:
        paths = [directory / f"{image_id}.xml" for image_id in ids]
        missing = [str(path) for path in paths if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"missing annotation(s): {', '.join(missing[:MAX_LISTED])}")

    class_counts: dict[str, int] = {}
    unknown_classes: dict[str, int] = {}
    degenerate: list[tuple[str, int]] = []
    out_of_bounds: list[tuple[str, int]] = []
    empty_images: list[str] = []
    n_boxes = 0

    for path in paths:
        element = ET.parse(path).getroot()
        width, height = _image_size(element, path)
        bad = 0
        outside = 0
        usable = 0

        for obj in element.findall("object"):
            name = (obj.findtext("name") or "").strip().lower()
            if name not in KNOWN_CLASSES:
                unknown_classes[name] = unknown_classes.get(name, 0) + 1
                continue

            xmin, ymin, xmax, ymax = _box(obj, path)
            # Counted before the checks below: the totals are a census of what the files declare,
            # and the fault lists then say how many of those boxes are unusable.
            n_boxes += 1
            class_counts[name] = class_counts.get(name, 0) + 1

            if xmax <= xmin or ymax <= ymin:
                bad += 1
                continue
            # VOC coordinates include the far edge, so a box ending exactly at width or height
            # is inside the frame; only a coordinate past it is a fault.
            if xmin < 0 or ymin < 0 or xmax > width or ymax > height:
                outside += 1
            usable += 1

        image_id = path.stem
        if bad:
            degenerate.append((image_id, bad))
        if outside:
            out_of_bounds.append((image_id, outside))
        if usable == 0:
            empty_images.append(image_id)

    return AnnotationReport(
        n_images=len(paths),
        n_boxes=n_boxes,
        class_counts=class_counts,
        degenerate=degenerate,
        out_of_bounds=out_of_bounds,
        empty_images=empty_images,
        unknown_classes=unknown_classes,
    )
