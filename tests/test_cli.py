"""Tests for the `sonar` command line front end."""

from __future__ import annotations

import argparse
import csv
import json
import re
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from conftest import HEIGHT, WIDTH, make_xml
from sonar.cli import build_parser, main


def subcommands(parser: argparse.ArgumentParser) -> dict[str, argparse.ArgumentParser]:
    """The named subparsers of `parser`, which argparse only exposes privately."""
    actions = [a for a in parser._actions if isinstance(a, argparse._SubParsersAction)]
    return dict(actions[0].choices)


def write_splits(root: Path, **splits: list[str]) -> None:
    main_dir = root / "ImageSets" / "Main"
    main_dir.mkdir(parents=True, exist_ok=True)
    for name, ids in splits.items():
        (main_dir / f"{name}.txt").write_text("".join(f"{image_id}\n" for image_id in ids))


def read_split(root: Path, name: str) -> list[str]:
    return (root / "ImageSets" / "Main" / f"{name}.txt").read_text().split()


def ids_of(root: Path) -> list[str]:
    return sorted(path.stem for path in (root / "Annotations").glob("*.xml"))


def predictions_for(gt, score: float) -> dict[str, dict]:
    """Predictions that reproduce the ground truth exactly, at one fixed score."""
    return {
        image_id: {
            "boxes": gt.boxes[image_id].clone(),
            "scores": torch.full((gt.labels[image_id].numel(),), score),
            "labels": gt.labels[image_id].clone(),
        }
        for image_id in gt.ids
    }


def with_decoys(predictions: dict[str, dict], score: float) -> dict[str, dict]:
    """Add one off-target box per image below every true score, so a FROC curve has a shape."""
    decoy = torch.tensor([[0.0, 0.0, 4.0, 4.0]])
    return {
        image_id: {
            "boxes": torch.cat([entry["boxes"], decoy]),
            "scores": torch.cat([entry["scores"], torch.tensor([score])]),
            "labels": torch.cat([entry["labels"], torch.tensor([1])]),
        }
        for image_id, entry in predictions.items()
    }


def above_score(predictions: dict[str, dict], floor: float) -> dict[str, dict]:
    """What a score floor leaves of an export: the same model, minus the quiet detections."""
    return {
        image_id: {key: value[entry["scores"] >= floor] for key, value in entry.items()}
        for image_id, entry in predictions.items()
    }


def empty_predictions(gt) -> dict[str, dict]:
    return {
        image_id: {
            "boxes": torch.zeros((0, 4)),
            "scores": torch.zeros(0),
            "labels": torch.zeros(0, dtype=torch.int64),
        }
        for image_id in gt.ids
    }


def table_column(printed: str, column: str) -> dict[str, float]:
    """One column of the `sonar eval` table, as {model name: value}."""
    lines = [line for line in printed.splitlines() if line.split()]
    header = next(i for i, line in enumerate(lines) if column in line.split())
    names = lines[header].split()
    values = {}
    for line in lines[header + 1 :]:
        fields = line.split()
        if len(fields) == len(names) + 1:  # a data row is the model name plus every column
            values[fields[0]] = float(fields[names.index(column) + 1])
    return values


def froc_curves(path: Path) -> dict[str, list[tuple[float, float]]]:
    """The FROC CSV grouped by model, in file order."""
    curves: dict[str, list[tuple[float, float]]] = {}
    for row in csv.DictReader(path.open()):
        curves.setdefault(row["model"], []).append((float(row["fppi"]), float(row["recall"])))
    return curves


def mobilenet_checkpoint(path: Path) -> None:
    from sonar.engine.train import TrainConfig, save_checkpoint
    from sonar.models.detector import build_detector

    save_checkpoint(path, build_detector(3, backbone="mobilenet", pretrained=False), TrainConfig())


def predict_argv(checkpoint: Path, root: Path, out: Path, *extra: str) -> list[str]:
    return [
        "predict",
        "--checkpoint",
        str(checkpoint),
        "--root",
        str(root),
        "--split",
        "test",
        "--out",
        str(out),
        "--backbone",
        "mobilenet",
        *extra,
    ]


def test_build_parser_exposes_every_documented_subcommand():
    commands = subcommands(build_parser())
    assert set(commands) == {
        "convert",
        "split",
        "preprocess",
        "audit",
        "train",
        "eval",
        "predict",
    }
    assert set(subcommands(commands["audit"])) == {"leakage", "annotations"}


def test_main_without_a_command_prints_help_and_returns_two(capsys):
    code = main([])
    printed = capsys.readouterr().out

    assert code == 2
    assert "usage: sonar" in printed
    for name in ("convert", "split", "preprocess", "audit", "train", "eval", "predict"):
        assert name in printed


def test_audit_without_a_kind_prints_its_own_help_and_returns_two(capsys):
    code = main(["audit"])
    printed = capsys.readouterr().out

    assert code == 2
    assert "leakage" in printed and "annotations" in printed


def test_audit_leakage_returns_one_when_a_training_split_holds_evaluation_ids(
    voc_root_factory, capsys
):
    """The CI gate for B2: the original roots disagreed and nothing noticed."""
    source = voc_root_factory("source")
    target = voc_root_factory("target")
    ids = ids_of(source)
    write_splits(source, train=ids[:8], val=ids[8:10], test=ids[10:])
    write_splits(target, train=ids[:8], val=ids[:6], test=ids[10:])

    code = main(["audit", "leakage", "--root", f"source={source}", "--root", f"target={target}"])
    printed = capsys.readouterr().out

    assert code == 1
    assert "source/train" in printed and "target/val" in printed
    assert "LEAKAGE" in printed
    # Both roots' train splits hit target/val, and the CLI's own line sums them.
    assert "leakage: 12 shared id(s) across 2 split pair(s)" in printed


def test_audit_leakage_returns_zero_when_the_roots_carry_one_disjoint_split(
    voc_root_factory, capsys
):
    source = voc_root_factory("source")
    target = voc_root_factory("target")
    ids = ids_of(source)
    for root in (source, target):
        write_splits(root, train=ids[:8], val=ids[8:10], test=ids[10:])

    code = main(["audit", "leakage", "--root", f"source={source}", "--root", f"target={target}"])
    printed = capsys.readouterr().out

    assert code == 0
    assert "clean: no training split intersects any evaluation split" in printed


def test_split_writes_byte_identical_files_to_every_root(voc_root_factory, capsys):
    primary = voc_root_factory("primary")
    mirror = voc_root_factory("mirror")

    assert main(["split", "--root", str(primary), "--also-root", str(mirror), "--seed", "7"]) == 0

    subsets = {}
    for name in ("train", "val", "test"):
        written = (primary / "ImageSets" / "Main" / f"{name}.txt").read_bytes()
        assert written == (mirror / "ImageSets" / "Main" / f"{name}.txt").read_bytes()
        subsets[name] = written.decode().split()

    # 12 fixture ids at the default 0.8/0.1/0.1, stratified over the two class-presence groups.
    assert [len(subsets[name]) for name in ("train", "val", "test")] == [8, 2, 2]
    assert sum(len(ids) for ids in subsets.values()) == len(set().union(*subsets.values()))
    assert set().union(*subsets.values()) == set(ids_of(primary))

    # The generator and the gate have to agree, otherwise one of them is decorative.
    capsys.readouterr()
    assert main(["audit", "leakage", "--root", f"a={primary}", "--root", f"b={mirror}"]) == 0


def test_split_follows_the_seed_and_only_the_seed(voc_root_factory):
    root = voc_root_factory("root")

    assert main(["split", "--root", str(root), "--seed", "7"]) == 0
    seven = read_split(root, "train")
    assert main(["split", "--root", str(root), "--seed", "11"]) == 0
    eleven = read_split(root, "train")
    assert main(["split", "--root", str(root), "--seed", "7"]) == 0

    assert read_split(root, "train") == seven
    assert eleven != seven  # a --seed that never reached the generator would be invisible


def test_split_rejects_ratios_that_do_not_sum_to_one(voc_root, capsys):
    code = main(["split", "--root", str(voc_root), "--ratios", "0.8", "0.3", "0.1"])

    assert code == 1
    assert "sum to 1.0" in capsys.readouterr().err


def test_audit_annotations_names_a_degenerate_box(voc_root, capsys):
    make_xml(voc_root / "Annotations" / "img_000.xml", WIDTH, HEIGHT, [("object", 10, 10, 10, 20)])

    code = main(["audit", "annotations", "--root", str(voc_root)])
    printed = capsys.readouterr().out

    assert code == 0
    assert "degenerate" in printed and "img_000" in printed
    assert "NOT CLEAN" in printed


def test_eval_tabulates_two_prediction_files_against_one_ground_truth(voc_root, tmp_path, capsys):
    from sonar.engine.evaluate import GroundTruth
    from sonar.engine.predict import write_predictions

    gt = GroundTruth.load(voc_root, "test")
    hits, misses = tmp_path / "hits.csv", tmp_path / "misses.csv"
    write_predictions(with_decoys(predictions_for(gt, 0.9), 0.2), hits)
    write_predictions(empty_predictions(gt), misses)
    froc = tmp_path / "froc.csv"

    code = main(
        [
            "eval",
            "--gt-root",
            str(voc_root),
            "--split",
            "test",
            "--preds",
            f"hits={hits}",
            "--preds",
            f"misses={misses}",
            "--froc",
            str(froc),
        ]
    )
    printed = capsys.readouterr().out

    assert code == 0
    assert f"root={voc_root} split=test, {len(gt)} images, {gt.n_boxes()} boxes" in printed

    # The table has to separate the two files. Scoring both against the wrong root or the wrong
    # split — the B6 failure this command exists to prevent — would not leave hits on top.
    ap50 = table_column(printed, "AP50")
    assert set(ap50) == {"hits", "misses"}
    assert ap50["hits"] == pytest.approx(1.0)
    assert ap50["misses"] < ap50["hits"]

    curves = froc_curves(froc)
    assert set(curves) == {"hits", "misses"}
    fppi = [point[0] for point in curves["hits"]]
    recall = [point[1] for point in curves["hits"]]
    assert len(curves["hits"]) > 1  # one operating point is a dot, not a curve
    assert fppi == sorted(fppi)
    assert fppi[0] == pytest.approx(0.0) and recall[0] == pytest.approx(1.0)
    assert max(point[1] for point in curves["misses"]) == pytest.approx(0.0)


def test_eval_refuses_a_prediction_file_that_cannot_prove_it_is_unfiltered(
    voc_root, tmp_path, capsys
):
    """Regression test for B7: AP off a truncated curve is not the number anyone reads it as."""
    from sonar.engine.evaluate import GroundTruth
    from sonar.engine.predict import PostprocessConfig, meta_path, write_predictions

    gt = GroundTruth.load(voc_root, "test")
    unfiltered = predictions_for(gt, 0.3)
    floor = PostprocessConfig(score_thresh=0.5)
    floored = above_score(unfiltered, floor.score_thresh)
    raw_path, floored_path = tmp_path / "raw.csv", tmp_path / "floored.csv"
    write_predictions(unfiltered, raw_path)
    write_predictions(floored, floored_path, cfg=floor)

    common = ["eval", "--gt-root", str(voc_root), "--split", "test", "--preds"]
    assert main([*common, f"raw={raw_path}"]) == 0

    assert main([*common, f"floored={floored_path}"]) == 1
    assert "score_thresh=0.5" in capsys.readouterr().err
    assert main([*common, f"floored={floored_path}", "--allow-filtered"]) == 0

    # A CSV with no sidecar is the same refusal: nothing records what was thrown away.
    capsys.readouterr()
    meta_path(raw_path).unlink()
    assert main([*common, f"unknown={raw_path}"]) == 1
    assert "sidecar" in capsys.readouterr().err
    assert main([*common, f"unknown={raw_path}", "--allow-filtered"]) == 0


def test_predict_writes_a_raw_csv_and_its_sidecar(voc_root, tmp_path, capsys):
    from sonar.engine.predict import RAW, read_predictions

    checkpoint = tmp_path / "detector.pt"
    mobilenet_checkpoint(checkpoint)
    out = tmp_path / "preds.csv"

    code = main(predict_argv(checkpoint, voc_root, out, "--batch-size", "2"))

    assert code == 0
    assert "wrote" in capsys.readouterr().out
    meta = json.loads(Path(f"{out}.meta.json").read_text())
    assert meta["postprocess"] == asdict(RAW)
    # The expectation is the split file, not the run's own output: --split has to be obeyed.
    assert set(meta["image_ids"]) == set(read_split(voc_root, "test"))
    assert set(read_predictions(out, require_raw=True)) == set(read_split(voc_root, "test"))


def test_predict_mode_chooses_the_postprocessing_it_records(voc_root, tmp_path):
    """`--mode vis` must reach both the detections and the sidecar `eval` trusts."""
    from sonar.engine.predict import VIS, read_meta, read_predictions

    checkpoint = tmp_path / "detector.pt"
    mobilenet_checkpoint(checkpoint)
    raw_out, vis_out = tmp_path / "raw.csv", tmp_path / "vis.csv"

    assert main(predict_argv(checkpoint, voc_root, raw_out, "--mode", "raw")) == 0
    assert main(predict_argv(checkpoint, voc_root, vis_out, "--mode", "vis")) == 0

    assert read_meta(vis_out).postprocess == VIS
    assert read_meta(raw_out).postprocess.score_thresh == 0.0
    raw_count = sum(entry["scores"].numel() for entry in read_predictions(raw_out).values())
    vis_count = sum(entry["scores"].numel() for entry in read_predictions(vis_out).values())
    assert raw_count > vis_count  # an untrained detector has nothing above the 0.5 vis floor


def test_predict_accepts_an_adaptation_checkpoint(voc_root, tmp_path):
    """`train_adaptive` saves the wrapper, so the detector weights arrive under a prefix."""
    from sonar.engine.predict import read_predictions
    from sonar.engine.train import TrainConfig, save_checkpoint
    from sonar.models.adaptation import AdaptationConfig, DomainAdaptiveDetector
    from sonar.models.detector import build_detector

    detector = build_detector(3, backbone="mobilenet", pretrained=False, min_size=64, max_size=64)
    model = DomainAdaptiveDetector(detector, AdaptationConfig(), mode="dccan")
    checkpoint = tmp_path / "dccan.pt"
    save_checkpoint(checkpoint, model, TrainConfig())
    out = tmp_path / "preds.csv"

    code = main(predict_argv(checkpoint, voc_root, out))

    assert code == 0
    predictions = read_predictions(out, require_raw=True)
    assert set(predictions) == set(read_split(voc_root, "test"))
    assert sum(entry["scores"].numel() for entry in predictions.values()) > 0


def test_predict_reports_a_missing_checkpoint_as_an_exit_code(voc_root, tmp_path, capsys):
    code = main(predict_argv(tmp_path / "absent.pt", voc_root, tmp_path / "preds.csv"))

    assert code == 1
    assert "checkpoint not found" in capsys.readouterr().err


def test_train_runs_one_epoch_and_writes_a_checkpoint(voc_root, tmp_path, capsys):
    from sonar.engine.train import TrainConfig, load_checkpoint
    from sonar.models.detector import build_detector

    config = tmp_path / "smoke.yaml"
    checkpoint = tmp_path / "smoke.pt"
    config.write_text(
        "name: smoke\n"
        "model:\n"
        "  backbone: mobilenet\n"
        "  num_classes: 3\n"
        "  pretrained: false\n"
        "  min_size: 64\n"
        "  max_size: 64\n"
        "data:\n"
        f"  root: {voc_root}\n"
        "  train_split: train\n"
        "  hflip: 0.5\n"
        "train:\n"
        "  epochs: 1\n"
        "  batch_size: 4\n"
        "  num_workers: 0\n"
        "  seed: 0\n"
    )

    code = main(["train", "--config", str(config), "--out", str(checkpoint)])
    printed = capsys.readouterr().out

    assert code == 0
    line = next(line for line in printed.splitlines() if line.startswith("epoch "))
    assert re.fullmatch(r"epoch 1: (?:\S+=-?\d+\.\d{4} )+\(\d+\.\ds\)", line)
    assert "total=" in line  # the summed loss that was actually backwarded

    meta = load_checkpoint(
        checkpoint, build_detector(3, backbone="mobilenet", pretrained=False, min_size=64)
    )
    assert meta["config"] == asdict(TrainConfig(epochs=1, batch_size=4, num_workers=0, seed=0))
    assert meta["extra"]["mode"] == "baseline"


def test_train_adaptive_reads_both_roots_and_validates_the_detector_inside_the_wrapper(
    voc_root_factory, tmp_path, capsys
):
    """`mode: dann` has to take the adaptive branch, and `val_root` has to score each epoch."""
    source = voc_root_factory("source")
    target = voc_root_factory("target")
    config = tmp_path / "dann.yaml"
    checkpoint = tmp_path / "dann.pt"
    config.write_text(
        "name: dann\n"
        "mode: dann\n"
        "model:\n"
        "  backbone: mobilenet\n"
        "  pretrained: false\n"
        "  min_size: 64\n"
        "  max_size: 64\n"
        "data:\n"
        f"  source_root: {source}\n"
        f"  target_root: {target}\n"
        f"  val_root: {target}\n"
        "  val_split: test\n"
        "adaptation:\n"
        "  dann_lambda: 0.1\n"
        "train:\n"
        "  epochs: 1\n"
        "  batch_size: 4\n"
        "  num_workers: 0\n"
        "  seed: 0\n"
    )

    code = main(["train", "--config", str(config), "--out", str(checkpoint)])
    printed = capsys.readouterr().out

    assert code == 0
    assert "loss_dann=" in printed  # the baseline loop would print detection losses only
    assert "AP50=" in printed  # val_fn unwrapped the detector and scored it
    state = torch.load(checkpoint, map_location="cpu")["model"]
    assert any(key.startswith("detector.") for key in state)


def test_train_rejects_a_key_that_nothing_reads(voc_root, tmp_path, capsys):
    """A typo used to be silent: a misspelled val_root simply turned validation off."""
    config = tmp_path / "typo.yaml"
    data = f"data:\n  root: {voc_root}\n"

    for text, needle in (
        (data + "train:\n  learning_rate: 0.1\n", "learning_rate"),
        (f"data:\n  root: {voc_root}\n  val_roots: {voc_root}\n", "val_roots"),
        (data + "model:\n  backbones: mobilenet\n", "backbones"),
        (data + "trian:\n  epochs: 1\n", "trian"),
    ):
        config.write_text(text)
        assert main(["train", "--config", str(config)]) == 1
        assert needle in capsys.readouterr().err


def test_train_names_the_data_key_a_config_leaves_out(tmp_path, capsys):
    config = tmp_path / "rootless.yaml"
    config.write_text("train:\n  epochs: 1\n  num_workers: 0\n")

    assert main(["train", "--config", str(config)]) == 1
    assert "'root'" in capsys.readouterr().err


def test_convert_writes_a_voc_root_from_yolo_labels(tmp_path, capsys):
    images, labels, out = tmp_path / "images", tmp_path / "labels", tmp_path / "voc"
    images.mkdir()
    labels.mkdir()
    Image.fromarray(np.full((100, 100, 3), 40, dtype=np.uint8)).save(images / "tile.jpg")
    (labels / "tile.txt").write_text("0 0.5 0.5 0.5 0.5\n")

    code = main(
        [
            "convert",
            "--images",
            str(images),
            "--labels",
            str(labels),
            "--out",
            str(out),
            "--workers",
            "2",
        ]
    )
    printed = capsys.readouterr().out

    assert code == 0
    assert "converted 1 images" in printed
    assert (out / "JPEGImages" / "tile.jpg").is_file()
    assert "<xmin>25</xmin>" in (out / "Annotations" / "tile.xml").read_text()


def test_convert_keep_empty_keeps_a_background_only_tile(tmp_path, capsys):
    """The original dropped 1,676 background tiles; --keep-empty is how they are kept."""
    images, labels = tmp_path / "images", tmp_path / "labels"
    images.mkdir()
    labels.mkdir()
    Image.fromarray(np.full((100, 100, 3), 40, dtype=np.uint8)).save(images / "blank.jpg")
    (labels / "blank.txt").write_text("")

    argv = ["convert", "--images", str(images), "--labels", str(labels), "--out"]
    dropped, kept = tmp_path / "dropped", tmp_path / "kept"
    assert main([*argv, str(dropped)]) == 0
    assert main([*argv, str(kept), "--keep-empty"]) == 0
    capsys.readouterr()

    assert not (dropped / "Annotations" / "blank.xml").exists()
    annotation = (kept / "Annotations" / "blank.xml").read_text()
    assert "<object>" not in annotation  # kept as background, never as a [0,0,1,1] box (B8)


def test_preprocess_writes_one_annotation_per_image(voc_root, tmp_path, capsys):
    destination = tmp_path / "denoised"

    code = main(
        ["preprocess", "--src", str(voc_root), "--dst", str(destination), "--mode", "denoised"]
    )
    printed = capsys.readouterr().out

    assert code == 0
    assert "wrote 12 images" in printed
    assert len(list((destination / "JPEGImages").glob("*.jpg"))) == 12
    assert len(list((destination / "Annotations").glob("*.xml"))) == 12


def test_preprocess_clahe_augmented_follows_the_seed(voc_root, tmp_path, capsys):
    def variant(name: str, seed: str) -> list[bytes]:
        destination = tmp_path / name
        argv = ["preprocess", "--src", str(voc_root), "--dst", str(destination)]
        assert main([*argv, "--mode", "clahe_augmented", "--seed", seed]) == 0
        capsys.readouterr()
        return [path.read_bytes() for path in sorted((destination / "JPEGImages").glob("*.jpg"))]

    first = variant("aug_a", "3")
    assert len(first) == 12
    assert variant("aug_b", "3") == first
    assert variant("aug_c", "4") != first  # an unread --seed would give the same augmentation


def test_named_path_options_reject_a_bare_path(voc_root):
    with pytest.raises(SystemExit) as excinfo:
        main(["audit", "leakage", "--root", str(voc_root)])
    assert excinfo.value.code == 2
