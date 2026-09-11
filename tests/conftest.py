"""Shared fixtures.

The synthetic dataset is built so that geometry bugs are *visible*: every image carries a bright
square at a position derived from its index, and every annotation box is exactly that square. A
transform that moves pixels without moving boxes therefore leaves the box sitting on dark
background, which a test can detect by measuring mean intensity inside the box.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

WIDTH, HEIGHT = 64, 48
SQUARE = 8
BACKGROUND = 20
FOREGROUND = 230


def make_xml(
    path: Path,
    width: int,
    height: int,
    objects: list[tuple[str, float, float, float, float]],
) -> None:
    """Write a VOC annotation containing `objects` as (name, xmin, ymin, xmax, ymax)."""
    parts = [
        "<annotation>",
        f"  <folder>{path.parent.parent.name}</folder>",
        f"  <filename>{path.stem}.jpg</filename>",
        "  <size>",
        f"    <width>{width}</width>",
        f"    <height>{height}</height>",
        "    <depth>3</depth>",
        "  </size>",
    ]
    for name, xmin, ymin, xmax, ymax in objects:
        parts += [
            "  <object>",
            f"    <name>{name}</name>",
            "    <pose>Unspecified</pose>",
            "    <truncated>0</truncated>",
            "    <difficult>0</difficult>",
            "    <bndbox>",
            f"      <xmin>{round(xmin)}</xmin>",
            f"      <ymin>{round(ymin)}</ymin>",
            f"      <xmax>{round(xmax)}</xmax>",
            f"      <ymax>{round(ymax)}</ymax>",
            "    </bndbox>",
            "  </object>",
        ]
    parts.append("</annotation>")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(parts))


def square_for(index: int) -> tuple[int, int]:
    """Top-left corner of the bright square for image `index`. Unique per index."""
    x = 4 + (index % 6) * 9
    y = 4 + (index // 6) * 16
    return x, y


def _build_root(root: Path, n: int) -> Path:
    images = root / "JPEGImages"
    annotations = root / "Annotations"
    splits = root / "ImageSets" / "Main"
    for d in (images, annotations, splits):
        d.mkdir(parents=True, exist_ok=True)

    ids = []
    for i in range(n):
        img_id = f"img_{i:03d}"
        ids.append(img_id)

        canvas = np.full((HEIGHT, WIDTH, 3), BACKGROUND, dtype=np.uint8)
        x, y = square_for(i)
        canvas[y : y + SQUARE, x : x + SQUARE] = FOREGROUND
        objects = [("object", x, y, x + SQUARE, y + SQUARE)]

        if i >= n // 2:
            sy = y + 16
            canvas[sy : sy + SQUARE, x : x + SQUARE] = FOREGROUND // 2
            objects.append(("shadow", x, sy, x + SQUARE, sy + SQUARE))

        Image.fromarray(canvas).save(images / f"{img_id}.jpg", quality=95)
        make_xml(annotations / f"{img_id}.xml", WIDTH, HEIGHT, objects)

    cut_train = int(n * 0.67)
    cut_val = cut_train + max(1, (n - cut_train) // 2)
    (splits / "train.txt").write_text("\n".join(ids[:cut_train]))
    (splits / "val.txt").write_text("\n".join(ids[cut_train:cut_val]))
    (splits / "test.txt").write_text("\n".join(ids[cut_val:]))
    return root


@pytest.fixture(autouse=True)
def _deterministic_rng():
    """Seed every test the same way so the order tests run in cannot change an outcome.

    Without this a test that seeds the global generator and does not restore it changes the
    random initialisation of models built by later tests, which showed up as two adaptation
    tests that passed alone and failed in the full suite.
    """
    import random

    import torch

    random.seed(1337)
    np.random.seed(1337)
    torch.manual_seed(1337)


@pytest.fixture
def voc_root(tmp_path: Path) -> Path:
    return _build_root(tmp_path / "voc", 12)


@pytest.fixture
def voc_root_factory(tmp_path: Path):
    def factory(name: str, n: int = 12) -> Path:
        return _build_root(tmp_path / name, n)

    return factory


@pytest.fixture
def tiny_images():
    """Two small float image tensors plus matching targets, for model-level tests."""
    import torch

    images = [torch.rand(3, 64, 64), torch.rand(3, 64, 64)]
    targets = [
        {
            "boxes": torch.tensor([[4.0, 4.0, 28.0, 28.0]]),
            "labels": torch.tensor([1]),
        },
        {
            "boxes": torch.tensor([[8.0, 8.0, 40.0, 36.0], [40.0, 40.0, 60.0, 60.0]]),
            "labels": torch.tensor([1, 2]),
        },
    ]
    return images, targets
