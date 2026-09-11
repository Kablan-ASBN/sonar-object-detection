"""Tests for stratified splitting and cross-root propagation."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from conftest import make_xml
from sonar.data.splits import (
    group_key,
    propagate_split,
    read_split,
    stratified_split,
    write_split,
)


def _groups_of(root: Path) -> dict[str, str]:
    """Map every image id under `root` to its stratification bucket."""
    return {path.stem: group_key(path) for path in (root / "Annotations").glob("*.xml")}


def _lopsided_root(root: Path, small: int, large: int = 20) -> Path:
    """A root with a `small` shadow_only bucket next to a `large` object_only one."""
    for i in range(small):
        make_xml(root / "Annotations" / f"shadow_{i:03d}.xml", 64, 48, [("shadow", 1, 1, 9, 9)])
    for i in range(large):
        make_xml(root / "Annotations" / f"object_{i:03d}.xml", 64, 48, [("object", 1, 1, 9, 9)])
    return root


@pytest.mark.parametrize(
    ("expected", "objects"),
    [
        ("object_only", [("object", 1, 1, 9, 9)]),
        ("shadow_only", [("shadow", 1, 1, 9, 9)]),
        ("both", [("object", 1, 1, 9, 9), ("shadow", 2, 2, 8, 8)]),
        ("empty", []),
        # A class the detector never learns carries no signal, so it buckets as empty.
        ("empty", [("seabed", 1, 1, 9, 9)]),
    ],
)
def test_group_key_buckets_by_class_presence(tmp_path, expected, objects):
    path = tmp_path / "annotation.xml"
    make_xml(path, 64, 48, objects)
    assert group_key(path) == expected


def test_same_seed_reproduces_the_split_and_another_seed_changes_it(voc_root):
    first = stratified_split(voc_root, seed=42)
    again = stratified_split(voc_root, seed=42)
    other = stratified_split(voc_root, seed=7)

    assert first == again
    assert first != other


def test_subsets_are_disjoint_and_cover_every_id(voc_root_factory):
    root = voc_root_factory("cover", 36)
    split = stratified_split(root, seed=3)

    train, val, test = set(split.train), set(split.val), set(split.test)
    assert train & val == set()
    assert train & test == set()
    assert val & test == set()

    expected = {path.stem for path in (root / "Annotations").glob("*.xml")}
    assert train | val | test == expected
    assert len(split.all_ids()) == len(expected)


@pytest.mark.parametrize("ratios", [(0.6, 0.2, 0.2), (0.6, 0.3, 0.1)])
def test_each_group_is_split_in_the_requested_ratio(voc_root_factory, ratios):
    root = voc_root_factory("ratios", 36)
    split = stratified_split(root, seed=11, ratios=ratios)
    buckets = _groups_of(root)

    by_group: dict[str, dict[str, int]] = {}
    for name, ids in split.as_dict().items():
        for image_id in ids:
            by_group.setdefault(buckets[image_id], dict.fromkeys(("train", "val", "test"), 0))
            by_group[buckets[image_id]][name] += 1

    assert len(by_group) > 1, "fixture must contain more than one bucket to test stratification"
    for counts in by_group.values():
        n = sum(counts.values())
        for name, wanted in zip(("train", "val", "test"), ratios):
            # The asymmetric case is the one that notices a val/test mix-up; with 18 images per
            # bucket a swap would put 2 images where 5.4 are wanted.
            assert abs(counts[name] - wanted * n) < 1.0
            assert counts[name] >= 1


@pytest.mark.parametrize("small", [3, 5, 7])
def test_a_small_bucket_reaches_every_subset(tmp_path, small):
    root = _lopsided_root(tmp_path / "lopsided", small)
    split = stratified_split(root, seed=4)

    shadows = {f"shadow_{i:03d}" for i in range(small)}
    per_subset = [len(shadows & set(ids)) for ids in split.as_dict().values()]
    assert min(per_subset) >= 1, "truncated cut points would leave val empty for this bucket"
    assert sum(per_subset) == small


def test_class_presence_proportions_survive_the_split(voc_root_factory):
    root = voc_root_factory("strata", 36)
    split = stratified_split(root, seed=5)
    buckets = _groups_of(root)

    overall = sum(1 for key in buckets.values() if key == "both") / len(buckets)
    for ids in split.as_dict().values():
        share = sum(1 for image_id in ids if buckets[image_id] == "both") / len(ids)
        assert abs(share - overall) < 0.15


def test_a_bucket_is_dealt_the_same_way_when_another_bucket_changes(tmp_path):
    before = _lopsided_root(tmp_path / "before", 5)
    after = _lopsided_root(tmp_path / "after", 5, large=21)
    shadows = {f"shadow_{i:03d}" for i in range(5)}

    # Seeding per bucket rather than threading one RNG through all of them keeps the shadow
    # draw still when the object bucket gains a member. Several seeds, because one shuffle of
    # five ids agrees with another about one time in twenty by luck.
    for seed in range(6):
        first, second = stratified_split(before, seed=seed), stratified_split(after, seed=seed)
        for name in ("train", "val", "test"):
            assert shadows & set(first.as_dict()[name]) == shadows & set(second.as_dict()[name])


def test_ids_argument_restricts_the_split_to_the_ids_given(voc_root):
    chosen = ["img_000", "img_002", "img_006", "img_008"]
    split = stratified_split(voc_root, seed=42, ids=chosen)

    assert sorted(split.all_ids()) == chosen


def test_repeated_ids_are_rejected_rather_than_dealt_into_two_subsets(voc_root):
    with pytest.raises(ValueError, match="img_000"):
        stratified_split(voc_root, ids=["img_000", "img_000", "img_001"])


def test_missing_annotations_are_named(voc_root, tmp_path):
    with pytest.raises(FileNotFoundError, match="Annotations"):
        stratified_split(tmp_path / "not-a-root")

    with pytest.raises(FileNotFoundError, match=r"img_999\.xml"):
        stratified_split(voc_root, ids=["img_000", "img_999"])


def test_as_dict_hands_out_copies(voc_root):
    split = stratified_split(voc_root, seed=42)

    split.as_dict()["train"].append("INJECTED")
    assert "INJECTED" not in split.train


def test_propagate_split_gives_two_roots_byte_identical_files(voc_root_factory):
    raw = voc_root_factory("raw", 36)
    preprocessed = voc_root_factory("preprocessed", 36)

    # B2: the original shuffled and sliced once per dataset root, so the roots disagreed about
    # which subset an id belonged to and train ids reappeared in another root's val.
    split = stratified_split(raw, seed=42)
    propagate_split(split, [raw, preprocessed])

    for name in ("train", "val", "test"):
        left = (raw / "ImageSets" / "Main" / f"{name}.txt").read_bytes()
        right = (preprocessed / "ImageSets" / "Main" / f"{name}.txt").read_bytes()
        assert left == right

    assert not set(read_split(raw, "train")) & set(read_split(preprocessed, "val"))
    assert not set(read_split(raw, "train")) & set(read_split(preprocessed, "test"))


def test_propagate_split_rejects_a_root_missing_an_id(voc_root_factory):
    full = voc_root_factory("full", 12)
    partial = voc_root_factory("partial", 6)
    split = stratified_split(full, seed=42)
    main = full / "ImageSets" / "Main"
    before = {path.name: path.read_bytes() for path in main.glob("*.txt")}

    with pytest.raises(ValueError, match=re.escape(str(partial))):
        propagate_split(split, [full, partial])

    # Every root is validated before any is written, so the first root is left as it was rather
    # than half-propagated - a disagreement between roots is the bug this module exists to stop.
    assert {path.name: path.read_bytes() for path in main.glob("*.txt")} == before


def test_ratios_must_sum_to_one(voc_root):
    with pytest.raises(ValueError, match=r"sum to 1\.0"):
        stratified_split(voc_root, ratios=(0.8, 0.2, 0.2))


def test_split_files_round_trip(voc_root):
    split = stratified_split(voc_root, seed=2)
    write_split(split, voc_root)

    for name, ids in split.as_dict().items():
        # Sorted subsets are what keep the written files stable between runs.
        assert ids == sorted(ids)
        assert read_split(voc_root, name) == ids


def test_read_split_names_the_missing_file(voc_root):
    with pytest.raises(FileNotFoundError, match=r"trainval\.txt"):
        read_split(voc_root, "trainval")
