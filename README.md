# Sonar object detection under domain shift

![ci](https://github.com/Kablan-ASBN/sonar-object-detection/actions/workflows/ci.yml/badge.svg)

Finding objects and their acoustic shadows in sidescan sonar, where the detector has to work on raw
survey imagery but the labelled data it was trained on looks different. A Faster R-CNN with a
ResNet-50 FPN backbone, adapted across that gap with DANN and with DCCAN, a three-path adversarial
architecture I proposed for my MSc dissertation.

The dissertation was submitted to the University of Greenwich in September 2025 and **awarded with
Distinction**, off the back of a research placement at [Seabed.AI](https://seabed.ai). A year later
I rebuilt the research code as a tested, installable package and re-ran every experiment on a
stricter protocol. This repository is that rebuild, and the re-run turned up something more useful
than a better number: the run-to-run variation on this benchmark is larger than any of the
differences being measured, including the margin the dissertation reported.

## Result

Four models, one seeded split shared by every dataset variant, all scored on the same 180-image test
split with unfiltered predictions. Produced by [`retrain_colab.ipynb`](retrain_colab.ipynb); both
runs are recorded under [`experiments/results/`](experiments/results/).

The headline is not which model won. It is that **two runs of identical code, identical config and
the same seed produced different orderings.**

| Model | trained on | run 1 AP50 | run 2 AP50 | mean | spread |
|---|---|---|---|---|---|
| *raw baseline (in-domain ceiling)* | *raw* | *0.1619* | *0.1448* | *0.1534* | *0.0171* |
| DCCAN | denoised | 0.1299 | 0.1279 | 0.1289 | 0.0020 |
| source-only baseline | denoised | 0.1141 | 0.1310 | 0.1226 | 0.0169 |
| DANN | denoised | 0.1131 | 0.1141 | 0.1136 | 0.0010 |

The three adaptation rows train on the denoised source domain and are tested on raw, so they are
doing real transfer. The raw baseline trains on the target domain itself, which makes it a ceiling
rather than a competitor: it shows what you would get if you already had target labels.

In run 1, DCCAN sat 0.0158 AP50 above source-only training. In run 2 it sat 0.0031 below it. Nothing
changed between the two except nondeterminism inside the runs themselves.

**The noise floor is therefore about 0.017 AP50, and every effect in this table is smaller than
that.** The 0.011 margin the dissertation reported is smaller still. Single-run comparisons at this
scale cannot separate these models, and that holds independently of the three implementation defects
in [docs/AUDIT.md](docs/AUDIT.md): even a defect-free single run could not have supported the
original conclusion.

Two things are stable across both runs and worth stating:

- The in-domain ceiling is highest in both. Training on the target domain beats adapting to it,
  which is what a ceiling is for.
- DANN is last in both and beats source-only training in neither. Global alignment alone does not
  help here, and that is the one comparative claim the data currently supports.

DCCAN against source-only is **undetermined**. Separating them needs several seeds per model with
`deterministic: true`, reporting mean and spread rather than a single number: twelve runs at roughly
twenty minutes each. Quoting the run-1 figure on its own would have been a smaller version of the
mistake the audit is about.

Why identical seeds diverge: [`utils/seed.py`](src/sonar/utils/seed.py) seeds Python, NumPy and
torch, but the configs leave `deterministic` off, so cuDNN picks algorithms by timing and
non-deterministic GPU atomics in the detection heads accumulate differently. That file's own
docstring says any run whose numbers get reported should turn it on. These did not, and the table
above is what that costs.

Absolute numbers are low because the task is hard: 500x500 sonar tiles, heavy speckle, and objects
that are often a few dozen pixels of slightly brighter return. Shadows score about twice what objects
do, which matches how a human reads a sonar waterfall.

## What this repository demonstrates

| | Where to look |
|---|---|
| Object detection in PyTorch: Faster R-CNN, ResNet-50 FPN, custom heads, mixed-precision training | [`src/sonar/models/`](src/sonar/models/) |
| Adversarial domain adaptation: gradient reversal, DANN, CDAN-style class conditioning, multi-level alignment | [`models/adaptation.py`](src/sonar/models/adaptation.py) |
| Test engineering: 262 tests, a regression test per defect, each checked by breaking the code to confirm the test fails | [`tests/`](tests/) |
| CI that gates on data integrity, not just code: lint, tests, and a split-leakage check that exits non-zero | [`.github/workflows/ci.yml`](.github/workflows/ci.yml) |
| Packaging and tooling: installable package, seven-subcommand CLI, YAML-driven experiments | [`pyproject.toml`](pyproject.toml), [`cli.py`](src/sonar/cli.py) |
| Evaluation you can trust: COCO metrics, FROC, and prediction files that record their own postprocessing | [`engine/evaluate.py`](src/sonar/engine/evaluate.py) |
| Data engineering: YOLO to Pascal VOC conversion, seeded stratified splits, annotation and leakage audits | [`data/`](src/sonar/data/), [`audit/`](src/sonar/audit/) |
| Debugging and root-cause analysis on someone else's code, where that someone was me a year ago | [`docs/AUDIT.md`](docs/AUDIT.md) |
| Knowing when not to report a result: measured the run-to-run noise floor and found it larger than the effect | [`experiments/results/`](experiments/results/) |

Built with PyTorch, torchvision, torchmetrics, pycocotools, NumPy, OpenCV, pytest, ruff and GitHub
Actions. Trained on an A100 through Colab.

## Running it

```bash
make setup      # virtualenv and editable install
make test       # 262 tests
make lint
make audit      # exits 1 if any training split overlaps any evaluation split
```

`make audit` also runs in CI on every push. It is the check that would have caught the leak described
below, and it is why a broken split can no longer reach a model unnoticed.

Training needs a GPU. All four models take about 90 minutes on an A100, roughly 35 seconds per epoch
for the baselines and 52 for the adaptation runs. The seeded split is already applied and committed,
so a re-run starts at training:

```bash
sonar audit leakage --root raw=data/line2voc \
                    --root denoised=data/line2voc_preprocessed   # must exit 0

for cfg in baseline_raw baseline_denoised dann dccan dccan_no_adapt; do
  sonar train   --config configs/$cfg.yaml --out runs/$cfg.pt --device cuda
  sonar predict --checkpoint runs/$cfg.pt --root data/line2voc --split test \
                --out preds/$cfg.csv --mode raw
done

sonar eval --gt-root data/line2voc --split test \
  --preds raw=preds/baseline_raw.csv --preds denoised=preds/baseline_denoised.csv \
  --preds dann=preds/dann.csv --preds dccan=preds/dccan.csv
```

One ground truth, one split, one postprocessing config, four models. `--mode raw` is not optional: it
writes a sidecar recording that no score floor was applied, and `sonar eval` refuses a filtered file
without it.

## Why the pipeline was rebuilt

Writing tests for research code tends to surface things that reading it does not. Three
implementation defects turned up, each of which happened to favour the model I had proposed over the
baselines I was comparing it against. Full detail, with line references, is in
[docs/AUDIT.md](docs/AUDIT.md).

| | What the code did | Why it mattered |
|---|---|---|
| **Two export protocols in one table** | The baselines were written to CSV behind `if float(s) < SCORE_THRESH: continue` with `SCORE_THRESH = 0.5`. DANN and DCCAN were exported at `score_thresh = 0.0`. | COCO AP integrates precision over the whole recall range. Only the baselines lost their low-confidence tail, so only the baselines were penalised. |
| **Flips moved pixels, not boxes** | `RandomHorizontalFlip` was applied as `img = transforms(img)`, so the image mirrored and the annotation stayed put. | Measured across the test split, 78.7% of affected labels no longer contained any part of their object. It hit the baselines and DANN; DCCAN had no flip. |
| **DANN skipped the detector's preprocessing** | `model.backbone(torch.stack(imgs))` bypasses `model.transform`, so the discriminator saw unnormalised, unresized input. DCCAN called `model.transform` first. | "Global alignment alone is not enough" was partly a correct implementation being compared with a broken one. |

Underneath all three, the dataset splits had been generated separately for each variant with an
unseeded `random.shuffle`, so the variants disagreed about what was held out. The repository now
detects that itself:

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

219 of the 242 validation images had been in the labelled training set. That is the 2025 state,
preserved under `docs/evidence/original_splits/`; the splits in `data/` are regenerated and the gate
passes. The test split escaped untouched, which is why a clean evaluation set existed at all.

These are ordinary research-code defects. A flip that moves pixels without boxes is one of the most
common bugs in object detection, and unseeded splits are endemic in academic pipelines. All three are
invisible to code review and obvious to a test.

All three are fixed, each one pinned by a test, and every experiment was re-run from scratch on the
corrected pipeline. The result of that re-run is the table at the top of this page: the ordering the
dissertation reported survived the stricter protocol.

### Every fix is held in place by a test

| Defect | Fix | The test |
|---|---|---|
| Flip moved pixels, not boxes | `DetectionTransform` owns the image and the coordinates together | reconstructs the old behaviour inline and asserts the box leaves the object |
| Splits drifted between variants | one seeded stratified split, propagated to every root | two roots produce byte-identical split files |
| Leaks went unnoticed | `sonar audit leakage`, non-zero exit, wired into CI | builds a leaking pair and asserts it is caught |
| Score-floored exports reached the evaluator | one `PostprocessConfig`, a `.meta.json` sidecar per file, `require_raw=True` | a floored file raises when read as raw |
| Models scored against different ground truth | `GroundTruth` binds to one root and one split; `compare` takes exactly one | mixing raises |
| Class-conditional branch was never trained | auxiliary source-side loss, or condition on the box head directly | asserts a non-zero gradient and that the weights move |
| Backbone run twice, target features unnormalised | one `transform` and one backbone pass per domain | counts backbone calls, asserts exactly two |
| Backbone freeze never released | `freeze_backbone` is reversible | asserts weights change again after the warm-up |

Several of those tests were themselves found wanting. An independent pass broke the code deliberately
to check each test failed as it claimed; nineteen did not, and were rewritten until they did.

## Architecture

DCCAN attaches three adversarial paths to Faster R-CNN, each through a gradient reversal layer: a
global one on pooled backbone features, a class-conditional one on the outer product of features and
class posteriors, and a lightweight convolutional one on the FPN map before the RPN. The components
are established work, DANN from Ganin and Lempitsky and CDAN from Long et al., applied at multiple
levels in the manner of Domain Adaptive Faster R-CNN. The particular combination, the ramp schedule
and the stabilisation around the conditional branch are mine.

[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) describes it in full, including two places where the
dissertation's prose does not match the code: the stated objective applies lambda as a loss weight
where the code applies it as a gradient reversal coefficient, and the feature map described as P3 is
in fact P2.

The claim I am most confident in is a practical one. A standalone CDAN outer-product mapping was
numerically unstable under automatic mixed precision in this setting; the three-path arrangement,
with the conditional branch weighted low behind a confidence gate, trains stably. That was the
difficult part of the architecture and it holds up.

### An open question

Across all twenty DCCAN epochs `loss_cdan` sits at 0.6931 to four decimal places, with `loss_dann`
and `loss_proposal` hovering near the same value. ln(2) = 0.6931 is exactly the loss of a domain
discriminator that cannot separate the two domains at all. That is either perfect domain confusion or
a discriminator that never learned anything, and the loss alone cannot tell them apart. The training
loop does not yet log discriminator accuracy, which would settle it in one number: 0.50 means the
paths are inert, higher means the gradient reversal is doing its job. The ramp starts the reversal
coefficient at exactly zero, so early on the discriminator trains essentially unopposed; one that
still cannot beat chance under those conditions is more likely not learning than at equilibrium.

If the paths are inert, DCCAN is not doing adaptation at all and its margin must come from the
training recipe, which is the same confound flagged in the Result section. Both questions are settled
by the same two runs: the matched control, and one forward pass over the saved checkpoint to read
discriminator accuracy off the domain heads, which `save_checkpoint` already stores.

## Data

3,464 sidescan sonar tiles from a single survey line, provided by Seabed.AI, annotated in YOLO format
with two classes. 1,676 of them contain no object and were dropped during the original conversion,
leaving 1,788. `sonar convert --keep-empty` retains them, which matters: a detector whose main
failure mode is false alarms on empty seabed ought to be evaluated on empty seabed.

The imagery is not redistributable beyond this repository. `data/` holds the VOC structure and split
files; `docs/evidence/original_splits/` preserves the 2025 splits as the evidence behind the audit.

## Provenance

Kablan Assebian. MSc Data Science dissertation, University of Greenwich, submitted 8 September 2025
and awarded with Distinction, supervised by Professor Chris Walshaw. Dataset and placement from
Seabed.AI. DCCAN was proposed and implemented as part of that work.

Research use only.
