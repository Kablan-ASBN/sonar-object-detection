"""Pascal VOC reader for the sonar tiles."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset

from sonar.data.transforms import eval_transform

# Label 0 is background by torchvision convention and must never appear in a target.
CLASS_MAP = {"object": 1, "shadow": 2}
CLASS_NAMES = {1: "object", 2: "shadow"}


def _read_split(root: Path, image_set: str) -> list[str]:
    path = root / "ImageSets" / "Main" / f"{image_set}.txt"
    if not path.is_file():
        raise FileNotFoundError(f"Missing split file: {path}")
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def _child_float(bndbox: ET.Element, tag: str, path: Path) -> float:
    child = bndbox.find(tag)
    if child is None or child.text is None:
        raise ValueError(f"{path}: <bndbox> has no <{tag}>")
    return float(child.text)


def parse_annotation(path: Path) -> tuple[list[list[float]], list[int]]:
    """Read one VOC XML, dropping unknown classes and degenerate boxes.

    Unusable content is skipped, malformed content is raised on: a class we do not train on is
    ordinary data, an <object> missing its <name> or <bndbox> is a broken file worth stopping for.
    """
    if not path.is_file():
        raise FileNotFoundError(f"Missing annotation: {path}")

    boxes: list[list[float]] = []
    labels: list[int] = []
    for obj in ET.parse(path).getroot().findall("object"):
        name = obj.find("name")
        if name is None or name.text is None:
            raise ValueError(f"{path}: <object> has no <name>")
        # Real exports are inconsistent about case and padding, so the name is normalised
        # before the lookup; anything still outside CLASS_MAP is a class we do not train on.
        label = CLASS_MAP.get(name.text.strip().lower())
        if label is None:
            continue
        bndbox = obj.find("bndbox")
        if bndbox is None:
            raise ValueError(f"{path}: <object> has no <bndbox>")
        xmin = _child_float(bndbox, "xmin", path)
        ymin = _child_float(bndbox, "ymin", path)
        xmax = _child_float(bndbox, "xmax", path)
        ymax = _child_float(bndbox, "ymax", path)
        if xmax <= xmin or ymax <= ymin:
            continue
        boxes.append([xmin, ymin, xmax, ymax])
        labels.append(label)
    return boxes, labels


class VOCDetection(Dataset):
    """VOC-format sonar tiles returning `(image, target)` for torchvision detectors.

    An image whose annotation holds nothing usable yields empty `boxes` and `labels`. The
    original fell back to `boxes=[[0, 0, 1, 1]], labels=[0]`, which taught the detector that
    a one-pixel corner box was a legitimate background-class instance.
    """

    def __init__(
        self,
        root: str | Path,
        image_set: str = "train",
        transforms: Callable | None = None,
        ids: Sequence[str] | None = None,
        drop_empty: bool = False,
    ) -> None:
        self.root = Path(root)
        self.image_set = image_set
        self.image_dir = self.root / "JPEGImages"
        self.annotation_dir = self.root / "Annotations"
        # __getitem__ promises a tensor, so the no-op transform is the default rather than None.
        self.transforms = transforms if transforms is not None else eval_transform()

        selected = list(_read_split(self.root, image_set)) if ids is None else [str(i) for i in ids]
        if drop_empty:
            selected = [i for i in selected if parse_annotation(self._annotation_path(i))[1]]
        self._ids = selected

    def __len__(self) -> int:
        return len(self._ids)

    def __getitem__(self, index: int) -> tuple[Tensor, dict[str, Any]]:
        image_id = self._ids[index]
        image_path = self.image_dir / f"{image_id}.jpg"
        if not image_path.is_file():
            raise FileNotFoundError(f"Missing image: {image_path}")

        image = Image.open(image_path).convert("RGB")
        boxes, labels = parse_annotation(self._annotation_path(image_id))
        box_tensor = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        label_tensor = torch.as_tensor(labels, dtype=torch.int64).reshape(-1)
        target: dict[str, Any] = {
            "boxes": box_tensor,
            "labels": label_tensor,
            "image_id": torch.tensor([index]),
        }
        return self.transforms(image, target)

    def _annotation_path(self, image_id: str) -> Path:
        return self.annotation_dir / f"{image_id}.xml"

    @property
    def ids(self) -> list[str]:
        return list(self._ids)


def collate_detection(
    batch: Sequence[tuple[Tensor, dict[str, Any]]],
) -> tuple[tuple[Tensor, ...], tuple[dict[str, Any], ...]]:
    """Keep variable-sized targets as a tuple; detection batches cannot be stacked."""
    return tuple(zip(*batch))
