"""Cross-root split leakage audit.

The original project generated a split per dataset root with an unseeded shuffle, so the raw and
the preprocessed roots disagreed about which tiles were held out: 219 of the 242 raw validation
ids were in the preprocessed training split, and every reported number on the raw domain was
measured on tiles the model had already been trained on.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

MAX_EXAMPLES = 5


@dataclass(frozen=True)
class Overlap:
    """Ids shared by one training split and one evaluation split."""

    left: str
    right: str
    count: int
    examples: tuple[str, ...]


@dataclass
class LeakageReport:
    """Split sizes plus every train/eval pair that shares at least one id."""

    splits: dict[str, int]
    overlaps: list[Overlap]
    # Every pair that was compared, so the matrix can show a clean pair as an explicit zero
    # instead of a gap. Optional: the offending pairs alone are enough to draw a matrix.
    checked: tuple[tuple[str, str], ...] = ()

    @property
    def has_leakage(self) -> bool:
        return bool(self.overlaps)

    def format(self) -> str:
        """Aligned matrix of shared-id counts, followed by the offending pairs."""
        counts = {(overlap.left, overlap.right): overlap.count for overlap in self.overlaps}
        pairs = self.checked or tuple(counts)
        rows = list(dict.fromkeys(left for left, _ in pairs))
        cols = list(dict.fromkeys(right for _, right in pairs))
        label_width = max((len(key) for key in self.splits), default=0)

        lines = ["split sizes:"]
        for key, size in self.splits.items():
            lines.append(f"  {key:<{label_width}}  {size:>5} ids")

        if rows and cols:
            cell = max(max(len(col) for col in cols), 5)
            lines += ["", "shared ids (rows: training splits, columns: evaluation splits):"]
            lines.append(" " * (label_width + 2) + "  ".join(f"{col:>{cell}}" for col in cols))
            for row in rows:
                cells = "  ".join(f"{counts.get((row, col), 0):>{cell}}" for col in cols)
                lines.append(f"  {row:<{label_width}}" + cells)

        lines.append("")
        if not self.overlaps:
            lines.append("no training id appears in any evaluation split")
            return "\n".join(lines)

        lines.append("LEAKAGE")
        for overlap in self.overlaps:
            shown = ", ".join(overlap.examples)
            more = ", ..." if overlap.count > len(overlap.examples) else ""
            noun = "id" if overlap.count == 1 else "ids"
            lines.append(
                f"  {overlap.left} and {overlap.right} share {overlap.count} {noun}: {shown}{more}"
            )
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "splits": dict(self.splits),
            "overlaps": [
                {
                    "left": overlap.left,
                    "right": overlap.right,
                    "count": overlap.count,
                    "examples": list(overlap.examples),
                }
                for overlap in self.overlaps
            ],
            "checked": [list(pair) for pair in self.checked],
            "has_leakage": self.has_leakage,
        }


def collect_splits(
    roots: Mapping[str, str | Path],
    names: Sequence[str] = ("train", "val", "test"),
) -> dict[str, set[str]]:
    """Read the named splits from every root, keyed `<root label>/<split name>`."""
    collected: dict[str, set[str]] = {}
    for label, root in roots.items():
        main = Path(root) / "ImageSets" / "Main"
        if not main.is_dir():
            raise FileNotFoundError(f"{label}: split directory not found: {main}")
        for name in names:
            path = main / f"{name}.txt"
            # A root is allowed to carry only some of the splits (a target domain often has no
            # labelled train list); only a root with no split directory at all is an error.
            if not path.is_file():
                continue
            ids = {line.strip() for line in path.read_text().splitlines() if line.strip()}
            collected[f"{label}/{name}"] = ids
    return collected


def check_leakage(
    roots: Mapping[str, str | Path],
    *,
    train_names: Sequence[str] = ("train",),
    eval_names: Sequence[str] = ("val", "test"),
) -> LeakageReport:
    """Compare every training split against every evaluation split, across all roots."""
    names = tuple(dict.fromkeys([*train_names, *eval_names]))
    splits = collect_splits(roots, names)

    train_set = set(train_names)
    eval_set = set(eval_names)
    train_keys = [key for key in splits if key.rsplit("/", 1)[1] in train_set]
    eval_keys = [key for key in splits if key.rsplit("/", 1)[1] in eval_set]

    # Same-root pairs are compared alongside the cross-root ones: the original sanity check
    # looked only at a root against itself, which is why it stayed green while raw validation
    # tiles sat in the preprocessed training split.
    checked: list[tuple[str, str]] = []
    overlaps: list[Overlap] = []
    for train_key in train_keys:
        for eval_key in eval_keys:
            checked.append((train_key, eval_key))
            shared = sorted(splits[train_key] & splits[eval_key])
            if shared:
                overlaps.append(
                    Overlap(train_key, eval_key, len(shared), tuple(shared[:MAX_EXAMPLES]))
                )

    sizes = {key: len(ids) for key, ids in splits.items()}
    return LeakageReport(sizes, overlaps, tuple(checked))
