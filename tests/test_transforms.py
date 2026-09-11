"""Tests for box-aware transforms, including the flip-desync regression (B1)."""

from __future__ import annotations

import pytest
import torch

from conftest import BACKGROUND, FOREGROUND
from sonar.data.transforms import (
    DetectionTransform,
    eval_transform,
    to_float_tensor,
    train_transform,
)
from sonar.data.voc import VOCDetection

WIDTH, HEIGHT = 64, 48


def _sample() -> tuple[torch.Tensor, dict]:
    image = torch.zeros(3, HEIGHT, WIDTH, dtype=torch.uint8)
    image[:, 6:14, 10:22] = 255
    target = {
        "boxes": torch.tensor([[10.0, 6.0, 22.0, 14.0]]),
        "labels": torch.tensor([1]),
        "image_id": torch.tensor([3]),
    }
    return image, target


def _mean_inside(image: torch.Tensor, box: torch.Tensor) -> float:
    xmin, ymin, xmax, ymax = (round(v) for v in box.tolist())
    return image[:, ymin:ymax, xmin:xmax].mean().item()


def test_hflip_mirrors_boxes_and_pixels_together():
    image, target = _sample()
    out_image, out_target = DetectionTransform(hflip_prob=1.0)(image, target)

    old = target["boxes"][0]
    new = out_target["boxes"][0]
    assert new[0].item() == WIDTH - old[2].item()
    assert new[2].item() == WIDTH - old[0].item()
    assert new[1].item() == old[1].item() and new[3].item() == old[3].item()
    assert torch.equal(out_image, torch.flip(image.float() / 255.0, dims=[2]))


def test_vflip_mirrors_the_other_axis():
    image, target = _sample()
    _, out_target = DetectionTransform(vflip_prob=1.0)(image, target)

    old = target["boxes"][0]
    new = out_target["boxes"][0]
    assert new[1].item() == HEIGHT - old[3].item()
    assert new[3].item() == HEIGHT - old[1].item()
    assert new[0].item() == old[0].item() and new[2].item() == old[2].item()


def test_flipping_only_the_image_moves_the_box_off_its_content(voc_root):
    """B1: the original flipped pixels and left boxes behind, so the box lost its target."""
    plain = VOCDetection(voc_root, "train", transforms=eval_transform())
    image, target = plain[0]
    box = target["boxes"][0]
    on_target = _mean_inside(image, box)

    # The original behaviour: transform the image alone, keep the annotation as it was.
    buggy_image = torch.flip(image, dims=[2])
    buggy = _mean_inside(buggy_image, box)

    flipped = VOCDetection(voc_root, "train", transforms=DetectionTransform(hflip_prob=1.0))
    flipped_image, flipped_target = flipped[0]
    flipped_box = flipped_target["boxes"][0]
    fixed = _mean_inside(flipped_image, flipped_box)

    assert flipped_box[0].item() == WIDTH - box[2].item()
    assert flipped_box[2].item() == WIDTH - box[0].item()
    assert buggy < 0.5 * on_target
    assert fixed > 0.8 * on_target
    assert fixed - buggy > 0.5


def test_zero_probability_is_a_passthrough():
    image, target = _sample()
    out_image, out_target = eval_transform()(image, target)

    assert torch.equal(out_target["boxes"], target["boxes"])
    assert torch.equal(out_image, image.float() / 255.0)
    assert torch.equal(out_target["image_id"], target["image_id"])


def test_image_is_float32_on_the_255_scale(voc_root):
    image, _ = VOCDetection(voc_root, "train", transforms=train_transform(hflip=1.0))[0]
    assert image.dtype == torch.float32
    assert image.min().item() >= 0.0 and image.max().item() <= 1.0

    # Pin both ends of the scale against the fixture's known pixel values; a range check alone
    # passes for any divisor large enough, which would hide a wrong one. The tolerance is JPEG
    # ringing around the square's edges.
    assert image.max().item() == pytest.approx(FOREGROUND / 255.0, abs=0.03)
    assert image.min().item() == pytest.approx(BACKGROUND / 255.0, abs=0.03)


def test_integer_images_are_scaled_by_their_dtype_range():
    assert to_float_tensor(torch.full((3, 2, 2), 255, dtype=torch.uint8)).max().item() == 1.0

    widened = to_float_tensor(torch.full((3, 2, 2), 200, dtype=torch.int16))
    assert widened.dtype == torch.float32
    assert widened.max().item() == pytest.approx(200 / 32767, rel=1e-6)


def test_float_image_outside_the_unit_range_is_rejected():
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        to_float_tensor(torch.full((3, 2, 2), 200.0))


def test_conversion_returns_a_copy_not_the_caller_tensor():
    source = torch.rand(3, 4, 4)
    original = source.clone()

    converted = to_float_tensor(source)
    assert converted is not source

    # No flip fires here, so this is the path that used to hand the caller's own tensor back.
    passthrough, _ = eval_transform()(source, {"boxes": torch.zeros((0, 4))})
    assert passthrough is not source
    passthrough.add_(0.25)
    assert torch.equal(source, original)


def test_empty_target_survives_a_flip():
    image, _ = _sample()
    target = {
        "boxes": torch.zeros((0, 4), dtype=torch.float32),
        "labels": torch.zeros((0,), dtype=torch.int64),
    }
    _, out_target = DetectionTransform(hflip_prob=1.0, vflip_prob=1.0)(image, target)

    assert out_target["boxes"].shape == (0, 4)
    assert out_target["labels"].shape == (0,)


def test_flip_draws_are_reproducible_under_a_torch_seed():
    image, target = _sample()
    transform = train_transform(hflip=0.5)

    def run() -> list[bool]:
        # fork_rng so seeding for this test does not leak into whatever runs next in the process.
        with torch.random.fork_rng():
            torch.manual_seed(0)
            return [
                bool(torch.equal(transform(image, target)[1]["boxes"], target["boxes"]))
                for _ in range(12)
            ]

    first = run()
    assert first == run()
    assert len(set(first)) == 2, "a 0.5 probability should fire for some draws and not others"


def test_transform_does_not_mutate_the_caller_target():
    image, target = _sample()
    DetectionTransform(hflip_prob=1.0)(image, target)
    assert torch.equal(target["boxes"], torch.tensor([[10.0, 6.0, 22.0, 14.0]]))
