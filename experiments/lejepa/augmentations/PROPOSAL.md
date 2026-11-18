# Controlled augmentation ablation — primary pair submitted

Primary pair submitted as job180196 after job180053 stopped during temporary-file
cleanup in preflight, before training. Optional follow-ups remain unsubmitted.
Initial matrix: one historical control and two new
200-epoch runs (`local-min10`, `jitter-strong`), all with seed 42. `local-min50`
and `more-views` have separate optional specs and are excluded from the default
launch. Epoch 200 is the primary comparison; retained epochs 50/100/150 show
trajectories. No automatic extensions or combinations.

## Exact existing policy

Verified against `data.py`, baseline/tuned configs, the sweep source snapshot,
screen `resolved.json`, continuation `resume.json`, and evaluation artifacts.
Before this change, current `data.py`, `main.py`, and `model.py` were byte-identical
to the frozen sweep source. The longer seed-43 run uses the same augmentations.
The default `config.json` still has the old LR/lambda; it is **not** the tuned control.

Each RGB source produces independent views, using the same transform chain:

1. Two global crops resized to **96×96**, area fraction **[0.6, 1.0]**;
   two local crops resized to **48×48**, area fraction **[0.3, 0.6]**.
   Both use aspect ratio **[0.9, 1.1]**, bilinear interpolation and antialiasing.
   Torchvision samples proposed area uniformly and aspect ratio logarithmically,
   with finite rejection attempts and its center-crop fallback. Bounds describe
   proposals, not guaranteed realized areas for every input geometry.
2. Horizontal flip **p=0.5**, vertical flip **p=0.0**.
3. Color jitter, applied with **p=0.5**: brightness, contrast, saturation
   magnitude **0.2** each (factors [0.8, 1.2]); hue magnitude **0.02**
   (offset [-0.02, 0.02]). Operation order is randomly permuted by torchvision.
4. Grayscale **p=0.05**, output remains three channels.
5. Tensor conversion, [0,1] scaling, ImageNet normalization:
   mean [0.485,0.456,0.406], std [0.229,0.224,0.225].

No Gaussian blur, solarization, rotation, affine/perspective transform, erasing,
or other geometry is present. Crops are sampled independently; local crops need
not be contained in a sampled global crop. View counts/resolutions stay fixed
in the primary comparisons; the optional `more-views` changes only local count.

## Reference comparison and source versions

The [original LeJEPA README recipe](https://github.com/galilai-group/lejepa/blob/c293d291ca87cd4fddee9d3fffe4e914c7272052/README.md)
and [stable-pretraining benchmark code](https://github.com/galilai-group/stable-pretraining/blob/ab836bf699a2be712dfcf980a5eb70d36391e876/benchmarks/imagenet10/lejepa-vit-small.py)
are separate references, inspected at the pinned revisions in these links.

| Parameter | This experiment | Original README | stable-pretraining benchmark |
|---|---|---|---|
| Global count / output | 2 / 96² | 2 / 224² | 2 / 224² |
| Local count / output | 2 / 48² | 6 / **98²** | 6 / **96²** |
| Global / local area | [.6,1] / [.3,.6] | [.3,1] / [.05,.3] | [.3,1] / [.05,.3] |
| Crop aspect ratio | [.9,1.1] | Unspecified in table | [3/4,4/3] wrapper default |
| Jitter B/C/S/H | .2/.2/.2/.02 | .4/.4/.2/.1 | .4/.4/.2/.1 |
| Jitter probability | .5 | .8 | .8 |
| Grayscale probability | .05 | .2 | .2 |
| Gaussian blur | absent | p=.5; kernel/sigma unspecified | p=.5, kernel 23, sigma [.1,2] |
| Solarization | absent | p=.2, threshold128 | p=.2, threshold128 |
| Horizontal / vertical flip | .5 / 0 | .5 / absent | .5 / absent |
| Rotation | absent | absent | absent |

The benchmark's transform calls use uniform blur p=.5 for both view types,
despite a stale comment describing asymmetric global blur. Its
[crop wrapper](https://github.com/galilai-group/stable-pretraining/blob/ab836bf699a2be712dfcf980a5eb70d36391e876/stable_pretraining/data/transforms.py)
supplies the aspect-ratio default. Both normalize with ImageNet statistics.
The README's 98-pixel local output is recorded as written, without silently
substituting the benchmark's 96.

There is also an [original minimal example](https://github.com/galilai-group/lejepa/blob/c293d291ca87cd4fddee9d3fffe4e914c7272052/MINIMAL.md)
with a different policy: recommended V=4 equal 128² views, crop [.08,1],
torchvision default ratio, jitter .8/.8/.8/.2 at p=.8, grayscale .2,
blur kernel7/sigma[.1,2] inside default-p=.5 RandomApply, and horizontal flip .5.
Its solarization is nested: RandomApply(p=.2) around RandomSolarize(default p=.5),
giving effective p=.1 under those API defaults. There is no single universal
"original recipe" that matches all these examples.

Our locked installed stable-pretraining **0.1.8** provides the loss and backbone;
it does not supply our data augmentations. The upstream benchmark above is an
explicit pinned reference, not a claim that this checkout ran that policy or
that today's upstream source is identical to the installed release. No dependency
upgrade or reference training data is used.

## Proposed matrix

Full explicit JSON configurations are next to this document. All values not
listed as changes below match `control.json`.

| ID | Priority | Local crop area | B/C/S jitter | Hue | Jitter probability | Global/local count |
|---|---|---|---|---|---|---|
| `control` | Historical reference | [.3,.6] | .2 | .02 | .5 | 2 / 2 |
| `local-min10` | **Initial spatial experiment** | **[.1,.6]** | .2 | .02 | .5 | 2 / 2 |
| `jitter-strong` | **Initial photometric experiment** | [.3,.6] | **.4** | **.1** | **.8** | 2 / 2 |
| `local-min50` | Optional follow-up | **[.5,.6]** | .2 | .02 | .5 | 2 / 2 |
| `more-views` | Optional follow-up | [.3,.6] | .2 | .02 | .5 | **2 / 6** |

`local-min10` changes only the minimum crop area, from30% to10%, keeping the
upper bound60%. Nominal minimum square side on a96² image falls from about53
to30 pixels, before resizing to48². Our existing policy already retains substantial
context; smaller crops may encourage transferable local features rather than
global appearance/layout shortcuts. They may also remove class-defining objects
or galaxy morphology. This tests insufficient spatial augmentation diversity
without presuming the stronger policy is better.

`local-min50` remains prepared to test the opposite hypothesis: preserving more
context may help object extent/morphology. Its minimum side is about68 pixels.
It is not included in `study.json`; `study-local-min50.json` explicitly opts into
a control-versus-local-min50 comparison after review.

`more-views` changes only local_views2→6 in `more-views.json`, retaining48²
local output, area[.3,.6], and all control photometric/geometric parameters.
`study-more-views.json` is a separate optional comparison. More independently
sampled within-image views may help low-data representations, but this comparison
also changes compute, batch statistics and objective aggregation as detailed below.

The jitter intervention is a bundled change **within the color-jitter family**;
it cannot distinguish brightness, contrast, saturation, hue, and frequency effects.
Grayscale stays .05; solarization and blur remain disabled. It is intentionally
not presented as a reproduction of the reference photometric recipe.

If informative, separately review the prepared optional comparisons, reduced
jitter, grayscale-only change, solarization-only change, a combination of winning
axes, or 90-degree rotations. Rotations could help some
orientations and harm upright objects, so require their own comparison. All
training policies apply uniformly to every image; labels and inferred domains
never select transforms. No Cartesian sweep.

## Optional more-views: exact loss aggregation and compute interpretation

Inspected the installed **stable-pretraining0.1.8** implementation in
`.venv/lib/python3.12/site-packages/stable_pretraining/methods/lejepa.py`:
`LeJEPA.forward`, `_compute_loss`, `SlicedEppsPulley.forward`, and
`EppsPulley.forward`. Preparation records this file's SHA256. The default
projector uses BatchNorm (`norm_layer="batch_norm"`); no loss code is changed.

Let `z` have shape[V,B,K], `G=2`, and `c = mean(z[:G], axis=0)`.
The actual objective is:

```text
invariance = mean((z - c[None])**2)                 # over V, B, K
SIGReg     = mean_slices(EP(z.reshape(V*B, K)))
loss       = invariance + 0.005 * SIGReg
EP         = (integrated empirical CF error) * (V*B)  # one GPU
```

The center still uses only two global projections, with gradients through the
center. Invariance is a mean, not a sum: going from4 to8 total views halves each
view's explicit averaging weight. Global/local terms shift from50%/50% to25%/75%
of the averaged residuals; center-dependent gradients mean these are not direct
fractions of the backbone gradient norm. There is no compensating loss rescale.

At batch128, SIGReg receives512→1024 projected embeddings in one pooled empirical
distribution, with1024 slices and17 quadrature points unchanged. Its explicit
sample-count multiplier doubles **at a fixed empirical CF error**. The observed
SIGReg value need not double: the empirical error changes with sample count,
correlation and global/local mixture. These are correlated views of128 source
images, not1024 independent training images. Preserve lambda=.005 and the standard
objective; do not divide SIGReg by two or reweight local residuals in this first test.

The backbone processes global and local batches separately: global batch256 stays
fixed, local batch256→768. The projector sees512→1024 embeddings together, altering
its BatchNorm statistics as well as SIGReg. Backbone local BatchNorm statistics
also change. Optimizer steps stay58/epoch and11,600/run, but total views processed
double. Conclusions must describe this as a standard more-views recipe comparison,
not an isolated measure of diversity at equal compute or unchanged loss statistics.

## Controls and provenance

Fixed: random ResNet-9 construction (1024-dimensional output), LeJEPA/projector,
seed42, all 7,500 provided images, preprocessing, float32, workers0, batch128,
drop_last, AdamW LR **.003**, weight decay .0001, default betas [.9,.999],
SIGReg lambda **.005**, 1024 slices, 10-epoch warmup followed by the existing
epoch-stepped cosine with **200-epoch horizon**, deterministic settings, and folds.
No new weights are initialized from another run. Each variant starts from scratch;
periodic interruption/resume restores that variant's own complete state.

Reusable historical run (verified against actual logged rates, not just config):

- Screen: `experiment_outputs/sweeps/lejepa-lr-sigreg-seed42/lr-0.003_lambda-0.005`.
  Stopped at100, configured epochs200, retained checkpoints50/100.
- Continuation: the same root's `continued-200/lr-0.003_lambda-0.005`.
  Restored the screen optimizer/scheduler/SIGReg state, retained150/200.
- At200: **73.07% five-fold, 69.64% macro, 72.40% official local LOO**.
  Fold accuracies: 71.33,75.33,73.33,70.67,74.67 percent.
- Original seed42 baseline has different LR/lambda and is contextual only.
  Seeds43/44 and the seed43 400-epoch run are contextual only. In particular,
  epoch200 of the 400-epoch schedule is not a matched control.

The two `metrics/metrics.csv` files contain5,800 LR rows each: steps0–5799
and5800–11599, with no missing or duplicate steps. With7,500 images, batch128
and drop_last, this is58 optimizer steps/epoch. Every rate matches the existing
10-epoch warmup and cosine **planned for200 epochs from initialization**, with
maximum absolute discrepancy **0.0 at recorded floating-point precision** using
the training scheduler's operation order. The resumed segment starts at epoch
index100 with the continuing cosine rate, not a new warmup or a100-epoch cosine.

| Training epoch (1-based) | Actual optimizer LR |
|---|---:|
| 1 | 0.00030000000000000003 |
| 10 / 11 | 0.003 |
| 50 | 0.002698781995342387 |
| 100 (last before continuation) | 0.0016485682953756944 |
| 101 (first after continuation) | 0.0016238690182084987 |
| 150 | 0.0005024631418292275 |
| 200 | 0.00000020504251129649376 |

`lr_history.verify_lr_history` checks all logged steps and rejects missing rows,
restarted warmup, a100-epoch initial cosine, or changed LR values. Each preparation
saves `control_lr_audit.json`, including both CSV hashes, rate samples and error.
This verifies the observed optimizer-rate trajectory; it does not claim a bitwise
counterfactual comparison with an uninterrupted run. The existing control is
appropriate; no replacement control training is required.

Data SHA256: `a5da94e1e90b6c62ce325706c8e9046bb3bd94231a621754d6eb76952fe367e4`.
Fold manifest SHA256: `7ca078f5a0c8938979aa689628b3a39efa72735e32ed81eb4e62d1dab45876cb`.
The manifest fixes 750 labeled rows, five 600/150 training/validation splits.
Historical versions: Torch2.14.1+cu130, torchvision0.29.1, Lightning2.6.6,
timm1.0.30, stable-pretraining0.1.8. Use the existing frozen environment.

Preparation freezes config/source/data/fold/reference/checkpoint hashes in
`plan.json`, copies source and folds, and writes per-run `augmentation.json`.
New checkpoints contain the full explicit config; training `resolved.json`
also records ordered transform metadata, package versions and dataset hash.
Historical checkpoints remain untouched; implicit historical defaults are resolved
in the study manifest. A changed plan/source/reference is rejected on restart.

## Evaluation and diagnostics

Use existing `main`, `checkpoint_eval`, `export`, `cross_validate`, and `diagnostics`.
The runner is orchestration, not a second train/probe implementation. Stop/resume
at50/100/150/200 preserves the full 200-epoch LR history and evaluates periodically.
Backfill historical control diagnostics into the new study directory only.

At every milestone: five-fold and macro accuracy, official local LOO, per-fold
metrics, per-class scores/supports, predictions/confusion, spatial PCA, nearest
neighbors, mean embedding variance and effective rank. Endpoint comparison
reports paired class/fold changes, probe learning curves and separate SSL loss
curves, plus side-by-side PCA panels. The control loss curve joins its two segments.
The unchanged official ridge uses analytic LOO internally to select regularization.

PCA freezes count12/seed42 across the full dataset; selection is independent of
labels and model. Same row IDs, shared-within-model basis mode, sampling budget,
sign convention, percentile scaling and overlay. Bases are fitted separately
per model, so colors are not aligned semantic meanings. The 12-image rank and
neighbor diagnostics are qualitative and sample-limited.

Invariance freezes 128 full-dataset images with seed42 and per-row jitter seeds.
Compare identity features against jitter(.4,.4,.4,.1, applied always), three-channel
grayscale, solarization128, and centered square crops of nominal area .3/.6/.9
resized to96. Save per-image cosine and relative L2, aggregate summaries, variance,
and mismatched-image cosine. These are diagnosis only. High similarity can also
indicate collapse or anisotropy. Central crop diagnostics do not estimate the
entire random training crop distribution. Solarization/grayscale measure responses
even though they are not changed by the initial interventions.

`CHALLENGE.md` explicitly says domain information is absent. No documented numeric
class-to-domain mapping was found; report class IDs without guessing. SSL saw
the development images; both outer-fold and local LOO operate on the same labeled
population. Repeated model selection on them does not test unseen classes/domains
or imply leaderboard improvement. Look for consistent trajectory changes and
class tradeoffs; do not select a lucky intermediate epoch as independent evidence.
If promising, request separate 400-epoch-from-scratch runs and additional seeds.

## Cost, storage, and launch commands (prepared, not executed)

Read-only `sacct` evidence: job161602 ran four 200-epoch repeats and evaluations
in **22:08:09** on gpu1907 (~5.54 allocated GPU-hours/run); job161245 ran two
100-epoch continuations/evaluations in5:42:27; job171065 ran400 epochs plus
milestone evaluations in10:53:54 on gpu2001. These are allocation elapsed times,
including overhead, not isolated kernel timing. Budget **~11 hours for two new
training recipes**, and **~13–16 allocated GPU-hours total** including twelve
checkpoint evaluations/diagnostic sets and preflight; diagnostic overhead is an
estimate and has not been timed. Request24h margin. Queue wait is additional.
Output sizes and view counts are unchanged, so GPU compute/memory should remain
close to control; stronger jitter can add CPU augmentation time.

Request one GPU with >=24GiB nominal VRAM, four CPUs,24GiB host RAM. Prior full
batch memory preflight on TITAN RTX:10.01GiB allocated,11.75GiB reserved,23.47GiB
total. Pascal remains excluded under the locked CUDA13 environment. A later
exact-batch test established that2080 Ti fits with tight headroom (see below).
TITAN RTX matches the seed42 control; RTX A6000, L40 and L40S are also eligible
to broaden access. Different cards may introduce numeric variation. Reserve roughly2GiB new storage plus plot/log margin; historical
checkpoints are referenced rather than copied. All outputs/caches stay in the repo.

For optional `local-min50`, expect roughly5.5 hours for one new training recipe
and7–9 allocated GPU-hours including its control/milestone diagnostics; use the
same24-GB GPU/24-GB host RAM/four CPUs and a12-hour limit.

For optional `more-views`, output pixels per source increase from
`2*96² + 2*48² = 23,040` to `2*96² + 6*48² = 32,256`, a **1.4× backbone
spatial-work proxy**. This is not a measured FLOP count or runtime ratio.
Local backbone work is approximately3×; projector and SIGReg sample work are2×;
CPU view generation is2× in count. Estimate **~1.4–2× training cost**, about
**8–11 hours for the new200-epoch run**, or **9–14 allocated GPU-hours** including
milestone/control diagnostics. Equal epochs imply unequal training compute;
report both epoch and measured allocation/training time when results exist.

Memory increases for local activations, projector batches and SIGReg tensors.
Neither peak memory nor runtime can be inferred by multiplying11.75GiB by1.4.
Reserve a **48-GB GPU**,48GB host RAM and four CPUs for this optional run, with
a24-hour limit, until measured. The launcher exercises an actual batch128,
two-global/six-local forward/backward/optimizer step using the existing memory
preflight before training, saving `more-views/memory-preflight.json`. It aborts
on OOM; it does not silently reduce batch size or change the objective. All
variants receive their own exact-config memory preflight. The48-GB choice is
headroom, not a measured minimum. Different optional GPU hardware is a numeric
confound versus the historical TITAN RTX control; document the actual device.

Pure metadata preparation, safe on the login host (no Torch import/training):

```bash
python3 -m experiments.lejepa.augmentation_study --prepare-only \
  --output experiment_outputs/augmentation-study/seed42-200-diversity-retry
```

The reviewed primary pair uses **one** training allocation from repository root.
After the original preflight cleanup failure and successful short2080 Ti fit
job180194, replacement job180196 started on gpu2003 at17:15:59 Eastern. It
accepts2080 Ti, TITAN RTX, RTX A6000, L40 and L40S. The request fits live partition
and node resources. GTX1080/1080Ti remain incompatible with the locked CUDA13
environment. The reproducible resource request is:

```bash
mkdir -p experiment_outputs/augmentation-study/seed42-200-diversity-retry
sbatch --partition=gpus --gres=gpu:1 --constraint='gtx_2080_ti|titan_rtx|rtx_a6000|l40|l40s' \
  --nodes=1 --ntasks=1 --cpus-per-task=4 --mem=24G --time=1-00:00:00 \
  --job-name=lejepa-aug42 --export=ALL \
  --output=experiment_outputs/augmentation-study/seed42-200-diversity-retry/slurm-%j.log \
  experiments/lejepa/augmentation_slurm.sh \
  --output experiment_outputs/augmentation-study/seed42-200-diversity-retry
```

This default command includes only control, `local-min10`, and `jitter-strong`.
The older `seed42-200` metadata directory belongs to the superseded proposal;
it is preserved, and the revised matrix uses a fresh immutable plan/output.

Optional commands are prepared for **separate subsequent review**, not executed
by the primary launcher. Each uses the same evaluation/report runner with the
historical control and exactly one new policy.

```bash
# Optional local-min50 only, after deciding to run this follow-up:
python3 -m experiments.lejepa.augmentation_study --prepare-only \
  --spec experiments/lejepa/augmentations/study-local-min50.json \
  --output experiment_outputs/augmentation-study/seed42-200-local-min50
sbatch --partition=gpus --gres=gpu:1 --constraint=titan_rtx \
  --nodes=1 --ntasks=1 --cpus-per-task=4 --mem=24G --time=12:00:00 \
  --job-name=lejepa-min50 --export=ALL \
  --output=experiment_outputs/augmentation-study/seed42-200-local-min50/slurm-%j.log \
  experiments/lejepa/augmentation_slurm.sh \
  --spec experiments/lejepa/augmentations/study-local-min50.json \
  --output experiment_outputs/augmentation-study/seed42-200-local-min50
```

```bash
# Optional more-views only, after deciding to run this follow-up:
python3 -m experiments.lejepa.augmentation_study --prepare-only \
  --spec experiments/lejepa/augmentations/study-more-views.json \
  --output experiment_outputs/augmentation-study/seed42-200-more-views
sbatch --partition=gpus --gres=gpu:1 --constraint='rtx_a6000|l40|l40s' \
  --nodes=1 --ntasks=1 --cpus-per-task=4 --mem=48G --time=1-00:00:00 \
  --job-name=lejepa-views --export=ALL \
  --output=experiment_outputs/augmentation-study/seed42-200-more-views/slurm-%j.log \
  experiments/lejepa/augmentation_slurm.sh \
  --spec experiments/lejepa/augmentations/study-more-views.json \
  --output experiment_outputs/augmentation-study/seed42-200-more-views
```

No child jobs/arrays are submitted. Preflight checks legacy augmentation tensor
and RNG parity, train/export/GPU memory, and fixed-fold isolation before execution.
After interruption, use the same command/output only after the prior allocation
has stopped. A failure stops this study, preserving its checkpoints/logs; restart
continues its own state and reuses hash-matched completed evaluations.
Repeated phase launches consume some startup time but do not restart optimization.

To regenerate the comparison later without training, inside an existing suitable
Slurm allocation:

```bash
.venv/bin/python -m experiments.lejepa.augmentation_report \
  --output experiment_outputs/augmentation-study/seed42-200-diversity-retry
```

Validation before review: six stdlib protocol tests passed; one transform/RNG
parity test intentionally skipped outside Slurm. Python syntax, shell syntax,
and whitespace checks passed. GPU execution and end-to-end plots are prepared
but remain unexecuted until review; no ablation results are claimed yet.

## Executed 2080 Ti fit test and replacement launch

Job180194 completed on gpu2003 in3m21s. The exact batch128, FP32, two96² global
and two48² local setup completed13 forward/backward/AdamW steps (3 warmup,
10 timed). Peak live allocation10.0817GiB, peak reserved10.3301GiB, device
total10.5814GiB. Median synthetic GPU step0.7549s; mean0.7177s. All7 protocol
tests passed, including transform/RNG parity and the temporary CSV cleanup fix.

This is evidence of fit, with tight peak headroom; CPU image generation, data
loading, Lightning and evaluation were excluded from the timing. It does not
establish a measured full-training speed ratio against TITAN RTX. The earlier
11.75GiB reservation on TITAN was not a minimum VRAM requirement: allocator
reclamation allowed this workload to fit on2080 Ti. Full training retains the
exact-config preflight and must still be monitored for memory failures.

Job180053 failed only on a shared-filesystem temporary-directory cleanup error
in one protocol test; the other26 checks passed and no training started. Test
artifacts now live under ignored experiment_outputs/protocol-tests, and cleanup
errors cannot override successful protocol assertions. Replacement job180196
uses the new immutable output seed42-200-diversity-retry and permits2080 Ti in
addition to TITAN RTX/A6000/L40/L40S. No batch, precision or objective changed.
Raw fit evidence: experiment_outputs/fit-tests/2080ti/result-180194.json.
