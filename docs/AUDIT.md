# Audit of the original pipeline

This project was my MSc dissertation, submitted in September 2025 and awarded with Distinction. A
year later I rebuilt the research code as a tested package, and writing tests for it is how the
issues below came to light — most of them are invisible to anyone reading the code, and only show up
when something independent checks that two things still agree.

The substance: the comparison that concluded DCCAN outperformed every baseline was affected by three
implementation defects, each of which happened to favour DCCAN, and together larger than the 0.011
AP50 margin involved. That comparison therefore needs re-running before the conclusion can be
restated either way.

This is a record of what the code did, not a retraction of the research. The design, the analysis
and the reasoning were assessed on their own terms. What follows is engineering: each defect below
has a regression test that fails if it ever comes back.

## Reproducing these checks

Everything here is measured from the artifacts in this repository, not recalled from memory.

```bash
make setup
make test                 # regression tests for every defect listed here
make audit                # the split-overlap report reproduced in finding 2
python experiments/reproduce_audit.py   # findings 1, 5 and 7 recomputed from outputs/
```

## Findings

Ordered by effect on the conclusion, not by how embarrassing they are.

### 1. The headline comparison mixed two different export protocols

The cross-model comparison read prediction CSVs that had been produced by two different export
paths with two different score thresholds.

| Model | Export cell | Score threshold applied |
|---|---|---|
| Raw baseline | `Faster_RCNN_Baseline_Model_20_Epoch.ipynb` code cell 5 | `SCORE_THRESH = 0.5` |
| Denoised baseline | same notebook, code cell 9 | `SCORE_THRESH = 0.5` |
| CLAHE+Aug baseline | same notebook, code cell 14 | `SCORE_THRESH = 0.5` |
| DANN | `dann_dccan_20+25_epoch_tuned.py`, RAW mode | `roi_heads.score_thresh = 0.0` |
| DCCAN | same script, RAW mode | `roi_heads.score_thresh = 0.0` |

The threshold is applied inside the loop that builds the exported rows, not only the one that draws
the figures:

```python
for b, s, l in zip(boxes, scores, labels):
    if float(s) < SCORE_THRESH:      # SCORE_THRESH = 0.5
        continue
    results.append({...})            # -> preds_baseline_frcnn_20epoch.csv
```

and `preds_baseline_frcnn_20epoch.csv` is exactly what the comparison cell reads for the
"Raw (20ep)" row.

COCO average precision integrates precision over the whole recall range. Discarding every
detection below 0.5 removes the low-confidence tail, which is precisely the region that extends
recall and therefore contributes the upper part of the AP integral. The three baselines were
scored with that tail deleted. DANN and DCCAN were scored with it intact.

This is a one-directional advantage to the models I was proposing, and it is not small. The
reported margin of DCCAN over the strongest baseline was 0.1633 − 0.1520 = **0.0113 AP50**.

Section 5.1 of the dissertation states the opposite of what the code does: "the raw model outputs
were exported as CSV without any confidence thresholding". That was true of the adaptation models
and false of the baselines.

**Fix.** `sonar.engine.predict` has one postprocessing type, `PostprocessConfig`, with two named
instances: `RAW` (nothing removed, the default) and `VIS` (for figures only). Every prediction file
is written with a `.meta.json` sidecar recording the config that produced it, and
`read_predictions(..., require_raw=True)` refuses a file that was score-filtered. A thresholded
export can no longer reach an evaluator by accident.

### 2. 90% of the validation set was in the training set

Splits were generated independently for each dataset root by `convert_yolo_to_voc.ipynb`, using
`random.shuffle` with no seed set anywhere in the file. The three roots therefore disagreed about
which images were held out.

```
                 raw/val    raw/test     pre/val    pre/test     aug/val    aug/test
raw/train              0           0         156           0         156           0
pre/train            219           0           0           0           0           0
aug/train            219           0           0           0           0           0
```

DANN and DCCAN take `line2voc_preprocessed/train` as the **labelled** source domain and were
evaluated on `line2voc/val`. 219 of those 242 validation images — 90.5% — were in the labelled
training set as median-filtered copies of the same scenes, with the same boxes. The denoised and
CLAHE baselines were trained on the same leaking split.

The raw baseline is the only row in the results table that was not affected, which means the
comparison was between one clean model and four contaminated ones.

The test split escaped: all three roots carry the *same* 180 ids, and no training split touches it.
That is luck rather than design, but it means a clean evaluation set exists.

**Fix.** `sonar.data.splits` generates one seeded stratified split and `propagate_split` writes the
identical split to every root. `sonar.audit.leakage` reports the matrix above and exits non-zero
when any training split intersects any evaluation split; it runs in CI on every push.

### 3. Horizontal flips moved the pixels but not the boxes

The training transform was built as

```python
train_tfms = T2.Compose([T2.ToImage(), T2.RandomHorizontalFlip(0.5), weights.transforms()])
```

and applied in the dataset as `img = self.transforms(img)` — to the image alone. The boxes were
never passed to the transform, so they were never flipped. Roughly half of every training epoch
consisted of mirrored images paired with unmirrored boxes.

This affected the three baselines and DANN. DCCAN's transform contains no flip
(`ToImage` + `ToDtype` only), so **DCCAN was the only model in the comparison trained on
uncorrupted labels**.

That is the second one-directional advantage, and it is independent of the first.

**Fix.** `sonar.data.transforms.DetectionTransform` flips the image and rewrites the box
coordinates in the same call. The regression test constructs the old behaviour inline and asserts
that the box no longer covers the object, while the new transform keeps it on target.

### 4. The class-conditional branch conditioned on an untrained layer

DCCAN's contribution is the combination of three alignment paths, of which the class-conditional
(CDAN-style) path is the novel part. It was built like this:

```python
proxy_cls = nn.Linear(F_DIM, NUM_CLASSES).to(device)   # line 922
...
g_s = proxy_cls(f_s_pool)                              # line 1136
p_s = F.softmax(g_s / temp, dim=1).detach()            # line 991
```

`proxy_cls` has exactly one consumer, and that consumer detaches. No supervised loss touches it.
SGD skips parameters whose `.grad` is `None`, so the layer never moved from its random
initialisation for any of the 20 epochs.

The outer product that defines CDAN conditioning was therefore taken against a fixed random
projection of the pooled features rather than against class posteriors. The temperature sharpening
(T = 0.6) and the confidence gate (0.40 decaying to 0.20) were both operating on random logits.

Detaching the softmax is correct — it stops the domain loss from dragging the classifier. What was
missing is the other half: something to train the classifier in the first place.

**Fix.** `sonar.models.adaptation` offers two conditioning modes. `proxy` keeps the image-level
proxy classifier but trains it with an auxiliary multi-label loss over the classes present in the
source targets, so the conditioning is real while the detach still protects it. `roi` conditions on
the detector's own box-head logits at ROI level, which is closer to CDAN as published. The
regression test asserts the proxy classifier has a non-zero gradient after `backward()` and that
its weights change across two optimiser steps.

### 5. Different models were scored against different ground truth

The comparison cell assigned CLAHE+Aug its own dataset root:

```python
("CLAHE+Aug (20ep)", .../"preds_claheaug_frcnn_20epoch.csv",
 PROJECT_ROOT/"data"/"line2voc_preprocessed_augmented", "val.txt"),
```

That root's validation split holds 179 ids and shares only 23 of them with the 242-image raw
validation split used for the other four rows. The denoised baseline was likewise scored against
the denoised validation set in its own notebook. The resulting table was presented under a single
heading, "Final Evaluation (Raw Target Domain — Validation Split)".

Scoring the CLAHE model against CLAHE ground truth is defensible on its own terms — the augmented
annotations were correctly transformed along with the pixels. The error is tabulating it beside
four models measured on a different image set and reading the column as a ranking.

**Fix.** `sonar.engine.evaluate.GroundTruth` is bound to exactly one root and one split, and
`compare()` takes a single `GroundTruth` for all models. Mixing is a type error rather than a
convention.

### 6. DANN's domain features skipped the detector's preprocessing

```python
f_s = model.backbone(torch.stack(src_imgs))["0"]     # line 343
```

`model.transform` is where torchvision applies ImageNet normalisation and resizing. Bypassing it
means the domain discriminator was aligning features drawn from a different input distribution than
the one the detector was trained on. DCCAN does not have this bug — it calls `model.transform`
first (line 1128).

So the DANN-versus-DCCAN comparison, which the dissertation reads as evidence that global alignment
alone is insufficient, is partly a comparison between a correct implementation and an incorrect
one. This is the third one-directional advantage.

Both paths also ran the backbone twice per step: once inside `model(images, targets)` and again for
the domain features.

**Fix.** `DomainAdaptiveDetector` transforms and runs the backbone once per domain and feeds the
same feature dict to the RPN, the ROI heads and the domain heads. The regression test counts
backbone invocations and asserts exactly two per forward pass.

### 7. The committed artifacts do not reproduce the reported table

The prediction files the comparison cell reads — `preds_baseline_frcnn_20epoch.csv`,
`preds_denoised_frcnn_20epoch.csv`, `preds_claheaug_frcnn_20epoch.csv`,
`eval_dann_allinone/preds_dann_20epoch_val_RAW.csv` — were never committed. What is in `outputs/`
is a differently-named, earlier set. Every one of them has a hard score floor at 0.50.

Recomputing the dissertation's own protocol on the committed files gives:

| Model | AP50 on raw/val | reported | AP50 on raw/test |
|---|---|---|---|
| Raw baseline | 0.1375 | 0.1493 | 0.1623 |
| Denoised baseline | 0.2285 | 0.1520 | 0.1793 |
| CLAHE+Aug | 0.0674 | 0.1166 | 0.0039 |
| DANN | 0.0376 | 0.0912 | 0.0532 |
| DCCAN | 0.0914 | 0.1633 | 0.0956 |

No row matches, and the ordering inverts: on these files the denoised baseline is the strongest
model and DCCAN is roughly half of it. These numbers are not a corrected result — they come from
score-floored files and are depressed for every model — but they establish that the repository
could not reproduce its own headline claim.

The `.pth` checkpoints were kept only in Google Drive and are gone, so re-exporting unfiltered
predictions from the original weights is not possible.

Two further artifacts are inconsistent with the code that supposedly wrote them:
`outputs/dccan_loss_curve.csv` has columns `det_loss, dann_loss, cdan_loss, lambda1, lambda2` with
both lambdas fixed at 1.0 for all 20 epochs — a two-path model with no ramp schedule, whereas the
committed DCCAN is a three-path model with a logistic ramp. `outputs/dann_loss_curve.csv` has
column names the committed code does not write either.

### 8. Smaller defects

| | Finding | Fix |
|---|---|---|
| 8.1 | The stated objective `L = L_det + λ_DANN·L_DANN + …` does not match the code, which applies λ only as the GRL coefficient (`loss = det_loss + dann_l + cdan_l + rpn_l`, line 1148). The GRL formulation is standard; the printed equation is wrong. | Documented correctly in `docs/ARCHITECTURE.md` |
| 8.2 | `freeze_backbone_bn` never matched anything: torchvision's pretrained ResNet-50 FPN uses `FrozenBatchNorm2d`, which is not an `nn.BatchNorm2d` subclass. It also set `requires_grad=False` without ever restoring it. | `freeze_backbone` is reversible and tested for it |
| 8.3 | `model.backbone(...)["0"]` is FPN level P2 (stride 4). The dissertation describes it as P3. | Configurable `feature_level`, documented |
| 8.4 | Empty or unparseable annotations fell back to `boxes=[[0,0,1,1]], labels=[0]`, injecting a ground-truth box with the background class reserved by torchvision. | Empty annotations yield genuinely empty tensors; tested |
| 8.5 | 1,676 of 3,464 images — every tile containing no object — were dropped at conversion. No background-only tile was ever trained or evaluated on, which removes the negatives from the false-positive analysis. The dissertation quotes the dataset as 3,465 images; the experiments used 1,788. | `convert_yolo_to_voc(..., keep_empty=True)` |
| 8.6 | `convert_box` truncated with `int()` instead of rounding, biasing every box toward the top-left by up to a pixel. | `yolo_box_to_voc` rounds and clamps |
| 8.7 | The converter's stratification buckets were inverted relative to its own class list (`class_ids == {0}` labelled `only_shadow`, while `CLASS_NAMES[0]` is `"object"`). Emitted labels were correct; only the printed summary was wrong. | Single `group_key` function, tested |
| 8.8 | `torch.use_deterministic_algorithms(False)` with `cudnn.benchmark = True`, no NumPy seed, and no best-checkpoint selection — the reported number is whatever epoch 20 happened to produce. | `sonar.utils.seed.set_seed`, per-epoch validation hook |
| 8.9 | Three annotations contain degenerate boxes (zero width or height). | `sonar.audit.annotations` reports them; loaders drop them |

## What this adds up to

Three defects each gave the proposed model an advantage over the models it was being compared
against:

1. the baselines were evaluated with their low-confidence detections deleted and DCCAN's were not;
2. the baselines and DANN were trained on labels that were wrong half the time and DCCAN's were not;
3. DANN's alignment ran on unnormalised features and DCCAN's did not.

On top of that, four of the five models were trained on 90% of their own validation set.

The reported margin was 0.011 AP50. I do not think any conclusion about DCCAN survives this, in
either direction — the honest position is that the experiment has not been run. What I can still
defend from the original work is narrower and I still believe it: the three-path architecture
trains stably under automatic mixed precision, where a standalone CDAN outer-product mapping did
not.

## Measured effects

`experiments/ablation.py` prices findings 1, 2 and 3 directly. It has not been run yet: the
checkpoints are gone and a ResNet-50 run is about forty hours on CPU, so the script uses a
MobileNetV3 backbone at 320px and needs a few hours. Numbers will be added here when it has run.

## What I changed about how I work

The defects have a shape in common. Every one of them is a place where two things that had to
agree were kept in separate places and allowed to drift: an image and its boxes, a split file and
its sibling roots, an export threshold and the evaluator reading it, a layer and its gradient.
None would have survived a test that asserted the two agreed.

That is why the fixes in this repository are mostly not clever code. They are a type that binds a
ground truth to one root, a sidecar that records how a file was produced, a transform that owns
both the pixels and the coordinates, and a CI job that fails when splits overlap.
