"""Stratified, reproducible train/val/test splits shared by every dataset root."""

from __future__ import annotations

import random
import xml.etree.ElementTree as ET
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

FOREGROUND_CLASSES = frozenset({"object", "shadow"})


@dataclass(frozen=True)
class Split:
    """One train/val/test partition of image ids."""

    train: list[str]
    val: list[str]
    test: list[str]

    def as_dict(self) -> dict[str, list[str]]:
        # Copies: frozen=True promises the subsets cannot change, and handing out the live
        # lists would give that promise away.
        return {"train": list(self.train), "val": list(self.val), "test": list(self.test)}

    def all_ids(self) -> list[str]:
        return [*self.train, *self.val, *self.test]


def group_key(annotation: Path) -> str:
    """Stratification bucket for one VOC annotation: object_only, shadow_only, both or empty."""
    root = ET.parse(annotation).getroot()
    names = {(obj.findtext("name") or "").strip() for obj in root.findall("object")}

    # Class names the detector never learns carry no stratification signal, so an annotation
    # holding only those is treated the same as an annotation holding nothing.
    present = names & FOREGROUND_CLASSES
    if present == {"object"}:
        return "object_only"
    if present == {"shadow"}:
        return "shadow_only"
    if present == {"object", "shadow"}:
        return "both"
    return "empty"


def _apportion(n: int, ratios: tuple[float, float, float]) -> list[int]:
    """How many of `n` members each subset gets: largest remainder, then a floor of one each.

    Truncating cut points send every leftover to the last subset, so a five-image bucket lands
    as 4/0/1 and validation never sees the class. Largest remainder hands each leftover to the
    subset furthest below its quota instead, and the floor pass keeps a bucket with room to
    spare represented everywhere, at the cost of at most one image from the largest subset.
    """
    quotas = [n * ratio for ratio in ratios]
    counts = [int(quota) for quota in quotas]
    by_shortfall = sorted(range(3), key=lambda i: (counts[i] - quotas[i], i))
    for i in by_shortfall[: n - sum(counts)]:
        counts[i] += 1

    wanted = [i for i, ratio in enumerate(ratios) if ratio > 0]
    if n >= len(wanted):
        for i in wanted:
            if counts[i] == 0:
                donor = max(range(3), key=lambda j: counts[j])
                counts[donor] -= 1
                counts[i] += 1
    return counts


def stratified_split(
    root: str | Path,
    *,
    seed: int = 42,
    ratios: tuple[float, float, float] = (0.8, 0.1, 0.1),
    ids: Sequence[str] | None = None,
) -> Split:
    """Partition ids by class presence so every subset sees the same class mixture.

    Deterministic for a given seed, which is the half of B2 that this function owns: the other
    half is writing the one result to every root, see `propagate_split`.
    """
    total = sum(ratios)
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"ratios must sum to 1.0, got {ratios} summing to {total}")

    annotations = Path(root) / "Annotations"
    if ids is None:
        if not annotations.is_dir():
            raise FileNotFoundError(f"annotation directory not found: {annotations}")
        ids = sorted(path.stem for path in annotations.glob("*.xml"))
    else:
        # A repeated id would be dealt into two subsets, which is the leakage this module exists
        # to prevent, so it is a caller error rather than something to silently collapse.
        repeated = sorted(i for i, count in Counter(ids).items() if count > 1)
        if repeated:
            raise ValueError(f"ids contains repeated entries: {', '.join(repeated[:3])}")

    groups: dict[str, list[str]] = {}
    for image_id in ids:
        path = annotations / f"{image_id}.xml"
        if not path.exists():
            raise FileNotFoundError(f"annotation not found: {path}")
        groups.setdefault(group_key(path), []).append(image_id)

    train: list[str] = []
    val: list[str] = []
    test: list[str] = []

    # Sorting a group before shuffling keeps the draw independent of filesystem and dict order,
    # and seeding per group keeps one bucket's draw from moving when another bucket gains or
    # loses members - the first `empty` annotation should not re-deal every other bucket.
    for key in sorted(groups):
        members = sorted(groups[key])
        random.Random(f"{seed}:{key}").shuffle(members)
        n_train, n_val, _ = _apportion(len(members), ratios)
        train += members[:n_train]
        val += members[n_train : n_train + n_val]
        test += members[n_train + n_val :]

    return Split(sorted(train), sorted(val), sorted(test))


def write_split(split: Split, root: str | Path) -> None:
    """Write the split to `root/ImageSets/Main/{train,val,test}.txt`, one id per line."""
    main = Path(root) / "ImageSets" / "Main"
    main.mkdir(parents=True, exist_ok=True)
    for name, ids in split.as_dict().items():
        (main / f"{name}.txt").write_text("".join(f"{image_id}\n" for image_id in ids))


def read_split(root: str | Path, name: str) -> list[str]:
    """Read the ids listed in `root/ImageSets/Main/{name}.txt`."""
    path = Path(root) / "ImageSets" / "Main" / f"{name}.txt"
    if not path.exists():
        raise FileNotFoundError(f"split file not found: {path}")
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def propagate_split(split: Split, roots: Sequence[str | Path]) -> None:
    """Write one split to several roots, so no root can re-deal ids into a different subset (B2)."""
    ids = split.all_ids()

    # Validate every root before writing any of them; a half-propagated split is the exact
    # state that produced the original leakage.
    for root in roots:
        path = Path(root)
        missing = [i for i in ids if not (path / "Annotations" / f"{i}.xml").exists()]
        if missing:
            raise ValueError(
                f"{path} is missing {len(missing)} id(s) referenced by the split, "
                f"starting with {', '.join(missing[:3])}"
            )

    for root in roots:
        write_split(split, root)
