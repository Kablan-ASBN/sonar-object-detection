"""Build the preprocessed dataset variants used as source domains.

Every pixel operation here that moves content also moves the annotation boxes. The raw, denoised
and augmented roots are meant to be interchangeable inputs to the same detector, so an image and
its XML must never drift apart.
"""

from __future__ import annotations

import random
import shutil
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

MODES = ("denoised", "clahe_augmented")

BRIGHTNESS_JITTER = 20.0
CONTRAST_JITTER = 0.1
JPEG_QUALITY = 95


def median_denoise(image: np.ndarray, ksize: int = 3) -> np.ndarray:
    """Median filter, the speckle suppressor used to build the denoised source domain."""
    if ksize < 3 or ksize % 2 == 0:
        raise ValueError(f"median kernel must be odd and at least 3, got {ksize}")
    return cv2.medianBlur(image, ksize)


def apply_clahe(
    image: np.ndarray,
    clip_limit: float = 2.0,
    tile_grid: tuple[int, int] = (8, 8),
) -> np.ndarray:
    """Contrast-limited equalisation of luminance only, leaving chroma alone."""
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    lightness, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tuple(tile_grid))
    return cv2.cvtColor(cv2.merge((clahe.apply(lightness), a, b)), cv2.COLOR_LAB2BGR)


@dataclass
class AugmentParams:
    hflip: bool
    angle_deg: float
    brightness: float
    contrast: float


def sample_augment(rng: random.Random, max_angle: float = 10.0) -> AugmentParams:
    """Draw one augmentation. Taking `rng` as an argument keeps a whole run reproducible."""
    return AugmentParams(
        hflip=rng.random() < 0.5,
        angle_deg=rng.uniform(-max_angle, max_angle),
        brightness=rng.uniform(-BRIGHTNESS_JITTER, BRIGHTNESS_JITTER),
        contrast=rng.uniform(1.0 - CONTRAST_JITTER, 1.0 + CONTRAST_JITTER),
    )


def apply_augment(
    image: np.ndarray,
    boxes: np.ndarray,
    params: AugmentParams,
) -> tuple[np.ndarray, np.ndarray]:
    """Augment pixels and boxes together, returning one box per input box, in order.

    Boxes are clamped to the frame but degenerate results are kept: the caller knows whether a
    box that rotated out of view should be dropped or the whole sample rejected.
    """
    height, width = image.shape[:2]
    out = image
    moved = np.asarray(boxes, dtype=np.float32).reshape(-1, 4).copy()

    if params.hflip:
        out = cv2.flip(out, 1)
        xmin = moved[:, 0].copy()
        moved[:, 0] = width - moved[:, 2]
        moved[:, 2] = width - xmin

    if params.angle_deg != 0.0:
        centre = ((width - 1) / 2.0, (height - 1) / 2.0)
        matrix = cv2.getRotationMatrix2D(centre, params.angle_deg, 1.0)
        out = cv2.warpAffine(
            out,
            matrix,
            (width, height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE,
        )
        moved = _rotate_boxes(moved, matrix)

    out = np.clip(out.astype(np.float32) * params.contrast + params.brightness, 0, 255)
    moved[:, 0::2] = moved[:, 0::2].clip(0.0, width)
    moved[:, 1::2] = moved[:, 1::2].clip(0.0, height)
    return out.astype(np.uint8), moved


def build_variant(
    src_root: str | Path,
    dst_root: str | Path,
    *,
    mode: str,
    seed: int = 42,
) -> int:
    """Write a preprocessed copy of a VOC root and return the number of images written."""
    if mode not in MODES:
        raise ValueError(f"unknown mode {mode!r}, expected one of {MODES}")

    src, dst = Path(src_root), Path(dst_root)
    images_dir = src / "JPEGImages"
    if not images_dir.is_dir():
        raise FileNotFoundError(f"no JPEGImages directory under {src}")
    image_paths = sorted(images_dir.glob("*.jpg"))
    if not image_paths:
        raise FileNotFoundError(f"no .jpg images under {images_dir}")

    out_images = dst / "JPEGImages"
    out_annotations = dst / "Annotations"
    out_images.mkdir(parents=True, exist_ok=True)
    out_annotations.mkdir(parents=True, exist_ok=True)

    # One rng for the whole run, consumed in sorted order, so a seed reproduces the exact variant.
    rng = random.Random(seed)
    written = 0
    for image_path in image_paths:
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"could not decode {image_path}")
        names, boxes = _read_objects(src / "Annotations" / f"{image_path.stem}.xml")

        image = median_denoise(image)
        if mode == "clahe_augmented":
            image = apply_clahe(image)
            image, boxes = apply_augment(image, boxes, sample_augment(rng))
            names, boxes = _drop_degenerate(names, boxes)

        height, width = image.shape[:2]
        cv2.imwrite(
            str(out_images / image_path.name),
            image,
            [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY],
        )
        _write_annotation(
            out_annotations / f"{image_path.stem}.xml",
            image_path.name,
            width,
            height,
            names,
            boxes,
        )
        written += 1

    _copy_splits(src, dst)
    return written


def _rotate_boxes(boxes: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Axis-aligned hull of the four rotated corners of each box.

    Box coordinates are edge coordinates (xmax is one past the last pixel) while warpAffine works
    in pixel-centre coordinates, hence the half-pixel shift on the way in and out. The error it
    removes grows with the angle: under 0.1 px at the +/-10 degrees sample_augment draws, a whole
    pixel at 180, where the corrected result matches flipping both axes exactly.
    """
    if len(boxes) == 0:
        return boxes
    x1, y1, x2, y2 = boxes.T
    corners = np.stack(
        [
            np.stack([x1, y1], axis=1),
            np.stack([x2, y1], axis=1),
            np.stack([x2, y2], axis=1),
            np.stack([x1, y2], axis=1),
        ],
        axis=1,
    )
    rotated = (corners - 0.5) @ matrix[:, :2].T + matrix[:, 2] + 0.5
    hull = np.concatenate([rotated.min(axis=1), rotated.max(axis=1)], axis=1)
    return hull.astype(np.float32)


def _drop_degenerate(names: list[str], boxes: np.ndarray) -> tuple[list[str], np.ndarray]:
    """Remove boxes that clamping collapsed to less than a pixel."""
    if len(boxes) == 0:
        return names, boxes
    keep = (boxes[:, 2] - boxes[:, 0] >= 1.0) & (boxes[:, 3] - boxes[:, 1] >= 1.0)
    return [n for n, k in zip(names, keep) if k], boxes[keep]


def _read_objects(path: Path) -> tuple[list[str], np.ndarray]:
    """Class names and xyxy boxes of one VOC annotation, class vocabulary untouched."""
    if not path.is_file():
        raise FileNotFoundError(f"missing annotation {path}")
    root = ET.parse(path).getroot()
    names: list[str] = []
    boxes: list[list[float]] = []
    for obj in root.findall("object"):
        bndbox = obj.find("bndbox")
        if bndbox is None:
            raise ValueError(f"object without a bndbox in {path}")
        names.append((obj.findtext("name") or "").strip())
        boxes.append([float(bndbox.findtext(tag)) for tag in ("xmin", "ymin", "xmax", "ymax")])
    return names, np.asarray(boxes, dtype=np.float32).reshape(-1, 4)


def _write_annotation(
    path: Path,
    filename: str,
    width: int,
    height: int,
    names: list[str],
    boxes: np.ndarray,
) -> None:
    """Write one VOC annotation, boxes rounded to the integer coordinates the format uses."""
    annotation = ET.Element("annotation")
    ET.SubElement(annotation, "folder").text = path.parent.parent.name
    ET.SubElement(annotation, "filename").text = filename
    size = ET.SubElement(annotation, "size")
    for tag, value in (("width", width), ("height", height), ("depth", 3)):
        ET.SubElement(size, tag).text = str(value)

    for name, box in zip(names, boxes):
        obj = ET.SubElement(annotation, "object")
        ET.SubElement(obj, "name").text = name
        ET.SubElement(obj, "pose").text = "Unspecified"
        ET.SubElement(obj, "truncated").text = "0"
        ET.SubElement(obj, "difficult").text = "0"
        bndbox = ET.SubElement(obj, "bndbox")
        # Round rather than truncate: int() would bias every box towards the top-left corner.
        for tag, value in zip(("xmin", "ymin", "xmax", "ymax"), box):
            ET.SubElement(bndbox, tag).text = str(round(float(value)))

    ET.indent(annotation, space="  ")
    path.write_text(ET.tostring(annotation, encoding="unicode") + "\n")


def _copy_splits(src: Path, dst: Path) -> None:
    """Carry the split files over verbatim so every variant indexes the same ids."""
    src_splits = src / "ImageSets" / "Main"
    if not src_splits.is_dir():
        return
    dst_splits = dst / "ImageSets" / "Main"
    dst_splits.mkdir(parents=True, exist_ok=True)
    for split_file in sorted(src_splits.glob("*.txt")):
        shutil.copy2(split_file, dst_splits / split_file.name)
