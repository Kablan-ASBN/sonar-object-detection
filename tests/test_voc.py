"""Tests for the VOC reader, including the background-class regression (B8)."""

from __future__ import annotations

import pytest
import torch

from conftest import HEIGHT, WIDTH, make_xml
from sonar.data.voc import CLASS_MAP, CLASS_NAMES, VOCDetection, collate_detection


def _split_ids(root, name: str) -> list[str]:
    text = (root / "ImageSets" / "Main" / f"{name}.txt").read_text()
    return [line.strip() for line in text.splitlines() if line.strip()]


def test_length_and_ids_match_the_split_file(voc_root):
    expected = _split_ids(voc_root, "train")
    dataset = VOCDetection(voc_root, "train")

    assert len(dataset) == len(expected)
    assert dataset.ids == expected


def test_explicit_ids_override_the_split_file(voc_root):
    dataset = VOCDetection(voc_root, "train", ids=["img_009", "img_010"])
    assert dataset.ids == ["img_009", "img_010"]
    assert len(dataset) == 2


def test_labels_are_only_foreground_classes(voc_root):
    for split in ("train", "val", "test"):
        for _, target in VOCDetection(voc_root, split):
            labels = target["labels"].tolist()
            assert labels, "the fixture annotates every image"
            assert set(labels) <= set(CLASS_NAMES)
            assert 0 not in labels


def test_targets_carry_boxes_labels_and_image_id(voc_root):
    dataset = VOCDetection(voc_root, "train")
    image, target = dataset[2]

    assert image.shape == (3, HEIGHT, WIDTH)
    assert target["boxes"].dtype == torch.float32
    assert target["boxes"].shape[1] == 4
    assert target["labels"].dtype == torch.int64
    assert torch.equal(target["image_id"], torch.tensor([2]))


def test_unknown_class_yields_an_empty_target_not_a_background_box(voc_root):
    """B8: unusable annotations used to fall back to a [0, 0, 1, 1] box with label 0."""
    make_xml(voc_root / "Annotations" / "img_000.xml", WIDTH, HEIGHT, [("seaweed", 4, 4, 12, 12)])
    _, target = VOCDetection(voc_root, "train")[0]

    # The fallback was boxes=[[0, 0, 1, 1]], labels=[0]; the empty shapes are what rule it out.
    assert target["boxes"].shape == (0, 4)
    assert target["labels"].shape == (0,)


def test_class_names_are_matched_case_insensitively(voc_root):
    objects = [("Object", 10, 10, 20, 20), (" SHADOW ", 30, 10, 40, 20), ("rock", 5, 5, 9, 9)]
    make_xml(voc_root / "Annotations" / "img_004.xml", WIDTH, HEIGHT, objects)
    _, target = VOCDetection(voc_root, "train")[4]

    assert target["labels"].tolist() == [CLASS_MAP["object"], CLASS_MAP["shadow"]]


def test_a_malformed_object_raises_instead_of_being_skipped(voc_root):
    """An <object> we cannot read is a broken file; only unknown classes are skipped quietly."""
    annotation = voc_root / "Annotations" / "img_005.xml"
    annotation.write_text(
        "<annotation><object><name>object</name></object></annotation>",
    )
    with pytest.raises(ValueError, match="bndbox"):
        VOCDetection(voc_root, "train")[5]

    annotation.write_text(
        "<annotation><object><bndbox><xmin>1</xmin><ymin>1</ymin>"
        "<xmax>5</xmax><ymax>5</ymax></bndbox></object></annotation>",
    )
    with pytest.raises(ValueError, match="name"):
        VOCDetection(voc_root, "train")[5]


def test_degenerate_boxes_are_dropped(voc_root):
    objects = [
        ("object", 10, 10, 10, 20),  # zero width
        ("object", 10, 10, 20, 10),  # zero height
        ("shadow", 30, 10, 40, 20),
    ]
    make_xml(voc_root / "Annotations" / "img_001.xml", WIDTH, HEIGHT, objects)
    _, target = VOCDetection(voc_root, "train")[1]

    assert target["labels"].tolist() == [CLASS_MAP["shadow"]]
    assert target["boxes"].tolist() == [[30.0, 10.0, 40.0, 20.0]]


def test_drop_empty_removes_images_with_no_usable_objects(voc_root):
    full = VOCDetection(voc_root, "train")
    make_xml(voc_root / "Annotations" / "img_003.xml", WIDTH, HEIGHT, [])
    kept = VOCDetection(voc_root, "train", drop_empty=True)

    assert len(kept) == len(full) - 1
    assert "img_003" not in kept.ids
    assert len(VOCDetection(voc_root, "train")) == len(full), "drop_empty must be opt-in"


def test_missing_split_file_names_the_path(voc_root):
    missing = voc_root / "ImageSets" / "Main" / "trainval.txt"
    with pytest.raises(FileNotFoundError) as excinfo:
        VOCDetection(voc_root, "trainval")
    assert str(missing) in str(excinfo.value)


def test_missing_image_names_the_path(voc_root):
    image_path = voc_root / "JPEGImages" / "img_000.jpg"
    image_path.unlink()
    dataset = VOCDetection(voc_root, "train")
    with pytest.raises(FileNotFoundError) as excinfo:
        dataset[0]
    assert str(image_path) in str(excinfo.value)


def test_collate_keeps_images_and_targets_as_tuples(voc_root):
    dataset = VOCDetection(voc_root, "train")
    batch = collate_detection([dataset[0], dataset[1], dataset[2]])

    # Tuples, not lists: torchvision detectors take their arguments positionally and a list of
    # targets is what a default collate would have tried to stack.
    assert isinstance(batch, tuple)
    images, targets = batch
    assert isinstance(images, tuple)
    assert isinstance(targets, tuple)

    assert len(images) == len(targets) == 3
    assert all(isinstance(image, torch.Tensor) for image in images)
    assert all("boxes" in target for target in targets)
