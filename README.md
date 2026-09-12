# Sonar object detection under domain shift

![ci](https://github.com/Kablan-ASBN/sonar-object-detection/actions/workflows/ci.yml/badge.svg)

Detecting objects and their acoustic shadows in sidescan sonar, where the detector has to work on
raw survey imagery but the labelled data it was trained on looks different. The detector is a
Faster R-CNN with a ResNet-50 FPN backbone; the domain adaptation is DANN and DCCAN, a three-path
adversarial architecture I proposed for my MSc dissertation.

This began as my MSc dissertation, submitted to the University of Greenwich in September 2025 and
**awarded with Distinction**, off the back of a research placement at [Seabed.AI](https://seabed.ai).
A year later I came back to rebuild the research code as a proper package — and rebuilding it is how
I found things the original implementation had been doing that I had not known about.

Three of them each favoured the model I was proposing over the baselines I compared it against, and
together they are larger than the 0.011 AP50 margin that comparison rested on. So that particular
claim needs re-running before it can be made again. The pipeline for doing that is in this
repository, and the audit below says exactly what to correct for.

None of this changes the dissertation, which was assessed on its design, analysis and writing and
stands as submitted. What changed is that I now have the tooling to check the implementation, and
the habit of doing it.

## What the rebuild found

Full detail, with evidence and line references, is in [docs/AUDIT.md](docs/AUDIT.md).

| | What the code did | Why it favoured the proposed model |
|---|---|---|
| **Mismatched export protocols** | The three baselines were written to CSV behind `if float(s) < SCORE_THRESH: continue` with `SCORE_THRESH = 0.5`. DANN and DCCAN were exported with `score_thresh = 0.0`. | COCO AP integrates precision across the whole recall range. Only the baselines lost their low-confidence tail. |
| **Flip augmentation desynchronised the labels** | `RandomHorizontalFlip` was applied as `img = transforms(img)`, so the pixels moved and the boxes did not. | It hit the three baselines and DANN. DCCAN's transform has no flip, so it was the only model trained on labels that were right. |
| **DANN skipped the detector's preprocessing** | `model.backbone(torch.stack(imgs))` bypasses `model.transform`, so the discriminator saw unnormalised, unresized inputs. DCCAN calls `model.transform` first. | "Global alignment alone is not enough" was partly a comparison between a broken implementation and a correct one. |

And the split leak, which the repository can now detect on its own:

```
$ sonar audit leakage --root raw=data/line2voc --root denoised=data/line2voc_preprocessed

shared ids (rows: training splits, columns: evaluation splits):
                        raw/val        raw/test    denoised/val   denoised/test
  raw/train                   0               0             156               0
  denoised/train            219               0               0               0

LEAKAGE
  denoised/train and raw/val share 219 ids: ...
leakage: 750 shared id(s) across 4 split pair(s)
$ echo $?
1
```

That is the 2025 state, archived under `docs/evidence/original_splits/`. The splits now in `data/`
are regenerated and `make audit` passes. The originals had been generated separately for each
dataset root by an unseeded `random.shuffle`, so the roots disagreed about what was held out. DANN and DCCAN take the denoised root as their
*labelled* source domain and were evaluated on raw validation.

The test split survived: all three roots carry the same 180 ids and no training split touches it.
That is luck rather than design, but it means a clean evaluation set exists.

## What is here

```
src/sonar/
  data/        VOC reader, box-aware transforms, YOLO conversion, seeded splits, preprocessing
  models/      detector builder, gradient reversal, domain heads, DANN and DCCAN
  engine/      training loops, COCO and FROC metrics, prediction export
  audit/       split leakage and annotation checks
  cli.py       sonar convert | split | preprocess | audit | train | eval | predict
tests/         262 tests, including a regression test for every defect in the audit
experiments/   the ablation that prices the defects, and a script that recomputes the audit
docs/          AUDIT.md, ARCHITECTURE.md, and the archived 2025 splits as evidence
configs/       one YAML per experiment
notebooks/     the original Colab notebooks, kept unchanged as the record
```

## Running it

```bash
make setup      # venv + editable install
make test       # 262 tests
make lint
make audit      # exits 1 if any training split overlaps any evaluation split
```

`make audit` also runs in CI on every push. It is the gate that would have caught the leak.

Training needs a GPU. On an A100 the original runs took a couple of hours each; on a CPU a single
ResNet-50 run is about forty hours, so the ablation below uses a MobileNetV3 backbone at 320px.

The seeded split is already applied and committed, so training starts at step two. `--mode raw` is
not optional: it records in a sidecar that no score floor was applied, and `sonar eval` refuses a
filtered file without it.

```bash
sonar audit leakage --root raw=data/line2voc \
                    --root denoised=data/line2voc_preprocessed   # must exit 0

for cfg in baseline_raw baseline_denoised dann dccan; do
  sonar train   --config configs/$cfg.yaml --out runs/$cfg.pt --device cuda
  sonar predict --checkpoint runs/$cfg.pt --root data/line2voc --split test \
                --out preds/$cfg.csv --mode raw
done

sonar eval --gt-root data/line2voc --split test \
  --preds raw=preds/baseline_raw.csv --preds denoised=preds/baseline_denoised.csv \
  --preds dann=preds/dann.csv --preds dccan=preds/dccan.csv
```

One ground truth, one split, one postprocessing config, four models. That last command is the
comparison the audit says the original one could not be.

## What the rebuild does differently

Each fix has a test that fails if the defect comes back.

| Defect | Fix | Test |
|---|---|---|
| Flip moved pixels, not boxes | `DetectionTransform` owns both and rewrites the coordinates | reconstructs the old behaviour inline and asserts the box leaves the object |
| Splits drifted between roots | one seeded stratified split, `propagate_split` writes it everywhere | two roots produce byte-identical split files |
| Leaks went unnoticed | `sonar audit leakage`, non-zero exit, wired into CI | builds a leaking pair and asserts it is caught |
| Score-floored exports reached the evaluator | one `PostprocessConfig`, a `.meta.json` sidecar per file, `require_raw=True` | a floored file raises when read as raw |
| Models scored against different ground truth | `GroundTruth` is bound to one root and one split; `compare` takes exactly one | mixing raises |
| Proxy classifier never trained | auxiliary source-side loss, or condition on the box head directly | asserts a non-zero gradient and that the weights move |
| Backbone run twice, target features unnormalised | one `transform` and one backbone pass per domain | counts backbone calls, asserts exactly two |
| Backbone freeze never released | `freeze_backbone` is reversible | asserts weights change again after the warm-up |

## Results

The `.pth` checkpoints only ever lived in Google Drive and are gone, and the prediction files the
original comparison actually read were never committed. What is in `outputs/` is an earlier set,
every one of them floored at 0.50. Recomputing the dissertation's own protocol on them:

| Model | AP50 on raw/val | reported | AP50 on raw/test |
|---|---|---|---|
| Raw baseline | 0.1375 | 0.1493 | 0.1623 |
| Denoised baseline | 0.2285 | 0.1520 | 0.1793 |
| CLAHE+Aug | 0.0674 | 0.1166 | 0.0039 |
| DANN | 0.0376 | 0.0912 | 0.0532 |
| DCCAN | 0.0914 | 0.1633 | 0.0956 |

No row reproduces, and the ordering inverts. These are not corrected numbers — they come from
score-floored files and are depressed for every model — but they establish that the repository
could not reproduce its own headline claim. Reproduce them with
`python experiments/reproduce_audit.py`.

`experiments/ablation.py` measures the defects directly rather than arguing about them. One
training run on the denoised source answers two at once, because under the archived 2025 splits raw validation shares 219 of
its 242 ids with that training set and raw test shares none, so the gap between the two is the leak.
Running it twice, with and without the flip bug, prices the third.

## Architecture

DCCAN attaches three adversarial paths to Faster R-CNN through gradient reversal layers: a global
one on pooled backbone features, a class-conditional one on the outer product of features and class
posteriors, and a lightweight convolutional one on the FPN map before the RPN.
[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) describes it, including two places where the
dissertation's description does not match the code — the stated objective applies λ as a loss
weight when the code applies it as a gradient reversal coefficient, and the "P3" feature map is
actually P2.

One claim from the original work survives intact, and I still stand behind it: the three-path design
trains stably under automatic mixed precision, where a standalone CDAN outer-product mapping did
not. That was the hard part of the architecture and it holds up. Whether it *detects* better than a
plain baseline is a separate question, and the honest answer is that the experiment needs running
again on the corrected pipeline. `configs/` and the Colab recipe below are set up to do exactly
that.

## Data

3,464 sidescan sonar tiles from a single survey line, provided by Seabed.AI, annotated in YOLO
format with two classes. 1,676 of them contain no object and were dropped by the original
conversion, leaving 1,788; `sonar convert --keep-empty` keeps them, which matters because a
detector whose main failure mode is false alarms on empty seabed should be evaluated on empty
seabed. The dissertation quotes the dataset as 3,465 images; the experiments used 1,788.

The imagery is not redistributable. `data/` in this repository holds the VOC structure and split
files; `docs/evidence/original_splits/` preserves the 2025 splits as the evidence behind the audit.

## Provenance

Kablan Assebian. MSc Data Science dissertation, University of Greenwich, submitted 8 September
2025 and awarded with Distinction, supervised by Professor Chris Walshaw. Dataset and placement from Seabed.AI. DCCAN was proposed and implemented
as part of that work.

Research use only.
