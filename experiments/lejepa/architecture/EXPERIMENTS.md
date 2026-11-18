# ResNet18 and frozen pooling study —2026-11-17

No pretrained weights, external images, label-conditioned augmentations, optional
training runs, or public model uploads are used. All compute runs through Slurm;
all new outputs are in `experiment_outputs/backbone-pooling/`.

## Primary architecture

The teammate's actual backbone was located and pinned in
`../references/dino_backbones.py` (commit e718fe83b85a6ce3e1ea10dfc39785e3baaf6307).
The implementation matches its random initialization and residual path exactly:

- RGB3→40,3x3 stride1 padding1 convolution without bias, BatchNorm, in-place ReLU.
- Four BasicBlock stages, widths[40,80,160,320], depths[2,2,2,2].
- Stage strides[1,2,2,2]. First blocks at channel/stride transitions use1x1
  stride2 convolution plus BN in the identity path; remaining identity paths
  are unchanged. Each BasicBlock contains two3x3 convolutions and post-add ReLU.
- No7x7 stride2 ImageNet stem or stem max pool. At96px, stage maps are
  96/48/24/12; at48px they are48/24/12/6. This avoids the standard stem reducing
  final maps to3x3 or2x2 on these inputs.
- Final GAP produces320 native features. The LeJEPA projector begins320→512
  and otherwise uses the same pinned default projector/SIGReg as before.
- Zero padding320→1024 happens only in the existing ONNX export utility.
  There is no learned1x1 reduction, no2x2 max readout and no padded SSL features.
- Convolution Kaiming-normal fan_out initialization, BN weight1/bias0, as in
  the teammate's implementation. This differs from the old ResNet9 constructor.

Backbone parameters: **4,365,480**. Export size and full-batch memory/step timing
are measured in `preflight/preflight.json` before the primary training submission.
Preflight checks exact seeded equality to the teammate, forward/backward, input320
to the projector, population-statistic readouts, unchanged frozen weights,
dynamic-batch export parity, and the official submission validator.

## Primary training and controls

`resnet18-tuned.json`: seed42,200 epochs planned from initialization,10-epoch
warmup plus existing cosine; LR.003, SIGReg lambda.005,1024 slices, AdamW
weight decay.0001/default betas, batch128,workers0,float32, strict determinism.
Two96px global views area[.6,1] and two48px local views area[.3,.6], ratio[.9,1.1],
horizontal flip.5, vertical flip0, jitter B/C/S.2,H.02,p.5, grayscale.05; no
blur/solarization/rotation. Same7,500 images, normalization, folds and ridge probes.

Matched reference: existing tuned seed42 ResNet9 at200 epochs:73.07% five-fold,
69.64% macro,72.40% LOO. Its actual200-epoch LR history was previously verified
across all11,600 steps; it was not a restarted or extended100-epoch cosine.
Architecture comparison changes depth/width, readout, initialization procedure
and embedding dimension together to match the teammate. It is not depth alone.
Frozen pooling helps evaluate the readout contribution separately. Device-type
variation may introduce numeric differences; actual hardware is recorded.

The primary runner stops/resumes its own optimizer/scheduler/SIGReg state at
50/100/150/200, retains checkpoints and performs existing export, fixed-fold,
macro, LOO, per-class/per-fold comparison and shared spatial PCA diagnostics.
Its SPT metrics also record allocated/reserved/peak GPU memory and epoch throughput.
`resources.json` reports training wall seconds and epoch images/sec; phase wall
times include evaluation/startup in `phase_times.json`. Use Slurm accounting for
total allocated GPU time. The resource callback is diagnostic only.

`resnet18-original-hparams-optional.json` changes only LR to.001 and lambda to.02
from the primary architecture configuration. It is prepared but not launched.

Initial resource estimate: one>=24GB GPU,4 CPUs,24GB host RAM,24-hour allocation.
Use only48GB GPUs if measured full-batch preflight requires them. Do not reduce
batch size or change precision to fit. Runtime estimates must be refined from
the synthetic-step preflight and first real epochs; synthetic timing excludes
CPU image augmentation, startup, exports and probes.

## Frozen readout experiment

Four checkpoints are included: original200/seed42, tuned200/seed42,
tuned400/seed43, and stronger-jitter200/seed42. The primary comparison is within
each checkpoint; the400-epoch checkpoint differs in seed/schedule.

| Readout ID | Operation | Native dimension |
|---|---|---:|
| max2 | Adaptive2x2 max, existing bin/order convention | 1024 |
| avg2 | Adaptive2x2 average | 1024 |
| gap | Global mean | 256 |
| gmp | Global max | 256 |
| avgmax | Concatenate global mean,max | 512 |
| avgmaxstd | Concatenate global mean,max,population std | 768 |

Maps immediately before pooling are `[B,256,12,12]`. Extract one cache per
checkpoint for the750 labeled rows plus the union with12 fixed PCA rows
(seed42); this avoids extracting all7,500 maps unnecessarily. Store unmodified
float32 maps in `cache/maps.npy`, with checkpoint/data/map hashes and row IDs.
Each readout operates on that same cache, then pads to1024 before the unchanged
official ridge. Preprocessing matches `evaluate.embed` exactly. No learned
readout, convolution/BN update, gradient step or SSL retraining takes place.

The existing max2 must match saved ONNX embeddings within rtol1e-3/atol1e-4 and
reproduce the historical outer and official LOO predictions **exactly**, including
aggregate metrics, before alternatives are interpreted. Failure stops the study.
The shared `cross_validate.probe_embeddings` calls the existing fixed-fold and
official Ridge code, so there is no second probe implementation.

Per-checkpoint reports save metrics, paired fold/class changes, prediction fixes,
confusion, standardized nearest-neighbor IDs, variance and effective rank. A
single spatial PCA panel per checkpoint suffices: changing a parameter-free
readout cannot change frozen spatial maps. Existing PCA code reuses the trained
map cache; its untrained reference is still independently computed. Color bases
are separately fitted per encoder, so colors are not semantic labels.

Export max2 plus readouts improving five-fold or LOO within a checkpoint, using
the same padding/export utilities. Check dynamic batches1/3/4, size<20MiB,
and compare ONNX embeddings against8 cached-image representations numerically.
Small extra inference for export parity is deliberate; no complete dataset
is re-embedded for every pooling variant. Saved convolution weights are checked
unchanged after export. No models are uploaded.

Prepare an optional from-scratch ResNet9 readout configuration only if a readout
gains at least0.5pp mean matched five-fold accuracy across the four checkpoints.
This is an explicit local selection rule, not a significance threshold or a
claim that these four checkpoints are independent training seeds. The config
is saved under the pooling output and is never launched automatically.

CPU pooling estimate:4 CPUs,16GB host RAM,roughly30–90 minutes,4-hour limit.
Caches use about110MiB/checkpoint (~0.45GB for four); embeddings, exports,
checkpoints and reports add storage. Allow~3GB total for both studies. CPU
pooling runs independently alongside primary GPU training after preflight.

## Launch and monitoring

Commands from the repository root (existing environment already provisioned):

```bash
# GPU preflight, job185752:
sbatch --partition=gpus --gres=gpu:1 --constraint='titan_rtx|rtx_a6000|l40|l40s' \
  --nodes=1 --ntasks=1 --cpus-per-task=4 --mem=24G --time=00:30:00 \
  --output=experiment_outputs/backbone-pooling/preflight/slurm-%j.log \
  experiments/lejepa/architecture_preflight_slurm.sh

# Only after successful preflight; exact primary GPU constraint is recorded at launch:
sbatch --partition=gpus --gres=gpu:1 --constraint='titan_rtx|rtx_a6000|l40|l40s' \
  --nodes=1 --ntasks=1 --cpus-per-task=4 --mem=24G --time=1-00:00:00 \
  --output=experiment_outputs/backbone-pooling/resnet18-tuned-seed42/slurm-%j.log \
  experiments/lejepa/architecture_slurm.sh

# Independent CPU frozen study:
sbatch --partition=compute --nodes=1 --ntasks=1 --cpus-per-task=4 \
  --mem=16G --time=04:00:00 \
  --output=experiment_outputs/backbone-pooling/frozen-readouts/slurm-%j.log \
  experiments/lejepa/pooling_slurm.sh
```

Create output directories first. The runner refuses changed source/config/data
plans on restart. Re-submit only after a stopped allocation; no concurrent writers
to the same output. Neither script submits child jobs or arrays.

```bash
squeue -u "$USER"
cat experiment_outputs/backbone-pooling/resnet18-tuned-seed42/progress.json
tail -f experiment_outputs/backbone-pooling/resnet18-tuned-seed42/training.log
cat experiment_outputs/backbone-pooling/frozen-readouts/progress.json
cat experiment_outputs/backbone-pooling/frozen-readouts/REPORT.md
```

Final reports are generated by the jobs. Local superior scores do not imply
better holdout transfer; the previous leaderboard history is in SUBMISSIONS.md.

## Executed preflight and submitted runs

GPU preflight185752 completed successfully on gpu2001 (TITAN RTX), elapsed5m27s:
23 tests passed, all6 official submission-validator checks passed, and8 strict
FP32 full-batch optimizer steps completed. Measured encoder parameter count
4,365,480; SSL parameters including the non-exported projector10,834,088. Actual
ONNX size17,455,594 bytes (16.65MiB). Peak GPU allocation8.3246GiB, reservation
8.8926GiB. Median synthetic GPU step0.6133s over5 post-warmup steps. No batch or
precision reduction is required. The kernel-only200-epoch estimate is about2h;
allow roughly5–8h for actual training plus milestone evaluation/startup overhead,
with a24h allocation limit. This is an estimate until real epoch timing exists.

Primary training: job185780, initially running on gpu2001, one TITAN RTX,4 CPUs,
24GB RAM,24h limit. Frozen pooling: job185781, initially running on echidna,
4 CPUs,16GB RAM,4h limit. Both launched after successful preflight and run
independently. Optional configurations and public uploads remain unlaunched.

```bash
squeue -j 185780,185781
sacct -j 185752,185780,185781 --format=JobID,State,Elapsed,AllocTRES,ExitCode
tail -f experiment_outputs/backbone-pooling/resnet18-tuned-seed42/slurm-185780.log
tail -f experiment_outputs/backbone-pooling/frozen-readouts/slurm-185781.log
```

Launch metadata: experiment_outputs/backbone-pooling/launch.json.

Initial real training measurement: first two epochs approximately107.2s each,
69.2 images/s, peak allocation8.3243GiB and reservation8.9141GiB. This projects
to about6h of training plus milestone evaluation/startup, not a final runtime.
The first frozen checkpoint (original200/seed42) reproduced historical max2
outer and LOO predictions exactly:70.00% five-fold,68.40% LOO. Remaining
checkpoint reproduction gates are enforced by the CPU job before comparison.

## Statistical pooling and broader frozen evaluation follow-up

The authorized next round uses the prepared statistical ResNet9 configuration and
extends this same runner/cache/probe/export infrastructure. See
[downstream protocol](../downstream/PROTOCOL.md) and
[launch record](../downstream/LAUNCH.md). Statistical training job199245 starts from
scratch with the planned200-epoch schedule; no optional second SSL variant or
leaderboard upload is launched. ResNet18 frozen pooling job199242 completed with
exact historical GAP reproduction; broader frozen tasks run separately from SSL.
