# Launch record — 2026-11-17

Implemented with existing uncommitted experiment work preserved. Repository/ancestor
AGENTS.md checks found no additional files; the shared-server instructions supplied
in the session apply. No subagents, additional SSL variants, or uploads were used.

## Submitted allocations

| Job | Purpose | Resources / limit | Dependency / status at launch |
|---:|---|---|---|
|199240|Exact statistical SSL preflight|1 TITAN RTX,4 CPU,24G /30min|Passed,2m31s|
|199242|ResNet18 frozen pooling|4 CPU,16G /4h|Completed, original GAP predictions reproduced exactly|
|199245|Statistical ResNet9 SSL200/42|1 TITAN RTX,4 CPU,24G /24h|Running; from scratch, milestones50/100/150/200|
|199247|Downstream/architecture/probe smoke tests|4 CPU /20min|22 passed,8 GPU-only skips; GPU SSL/export checks separately passed199240|
|199249|Pet archive verification/extraction|2 CPU,8G /30min|Completed; official3680/3669 split|
|199258|Existing encoders: cached extraction + correspondence|1 compatible GPU,4 CPU,24G /4h|Running; replaces cancelled199250|
|199251|Existing encoders: global tasks + RGB regression|4 CPU,24G /4h|afterany199258; evaluates successfully cached models even if another extraction fails|
|199252|Existing encoders: Pet linear probes|1 compatible GPU,4 CPU,24G /12h|afterok199249; running|
|199253|New statistical encoder: extraction + correspondence|1 compatible GPU,4 CPU,24G /1h|afterok199245|
|199254|New statistical encoder: CPU tasks|4 CPU,24G /1h|afterok199253|
|199255|New statistical encoder: Pet probes|1 compatible GPU,4 CPU,24G /3h|afterok199245 and199249|
|199292|Appearance-only float64 numerical correction|4 CPU,24G /2h|afterany199251; reuse caches and semantic metrics|
|199293|Float64 appearance regression fixture|2 CPU,4G /10min|Smoke check|
|199263|Fixed label-independent K=30 semantic regression fixtures|2 CPU,4G /10min|2 passed|
|199261|Official validators for all ResNet18 exports; report refresh|2 CPU,8G /20min|Independent CPU audit|

Numerical preflight measured2,437,696 encoder parameters,9,135,680 SSL parameters,
768-dimensional features and projector768→512→2048→2048→512. Peak allocation10.121GiB,
reservation11.773GiB, median synthetic step.3905s over5 post-warmup steps.
Full FP32 batch128 with2×96 and2×48 views completed8 optimizer steps with finite
encoder/projector gradients. Constant-channel std backward passed on CPU and GPU.
ONNX9,750,148 bytes, dynamic batch1/3/4 parity and all6 official validator checks passed.
First real epochs take~93s (roughly80 images/s), peak allocation10.12GiB and
reservation13.69GiB; allow~5–6h plus variable startup/evaluation overhead.

## Completed ResNet18 pooling endpoint

| Readout | Native dim | Five-fold | Macro | Official LOO |
|---|---:|---:|---:|---:|
|GAP|320|72.93%|69.61%|73.47%|
|GMP|320|73.47%|70.23%|72.93%|
|Mean+max|640|75.07%|71.78%|74.53%|
|Mean+max+std|960|75.33%|72.29%|74.80%|

Historical GAP embeddings and both saved prediction arrays reproduced before any
alternative was interpreted. Mean+max+std gains2.40pp five-fold,2.68pp macro and
1.33pp LOO within the same frozen encoder. GMP shows that improving one metric
can worsen another. These development gains do not establish hidden transfer.
All four exports passed shared dynamic-batch/cached-feature parity checks; job199261
runs the additional official validator for each. Detailed results and paired changes:
`experiment_outputs/green-area/resnet18-pooling/`.

## Deviations and isolated failures

- Pet download jobs199241(torchvision default URL) and199244(official mirror via
  curl) timed out on compute-node outbound connections. With explicit user approval
  for the~800MB download, only archive transfer ran on the login host; verification,
  archive extraction and all image processing remained in Slurm. Both official
  archives were checked against torchvision MD5s. No package/environment changes.
- Initial downstream extraction199250 landed on GTX1080Ti(sm61), unsupported by
  installed PyTorch2.14.1+cu130(sm75+). It was cancelled before producing metrics,
  and its attempt plan retained. Replacement199258 and future GPU jobs use
  `titan_rtx|rtx_a6000|l40|l40s|gtx_2080_ti`. No other user's jobs were modified.
- Correspondence uses192px because final ResNet18 RF109 cannot be fully visible
  at96px. Segmentation/global tasks remain96px. Full RF-valid overlap, native-grid
  quantization and raw RGB/random CNN controls are reported.
- Pet caches store frozen FP32 features in FP16 to bound disk usage; probes train
  in FP32. Challenge caches/readout comparisons remainFP32. This approximation is
  documented for every segmentation endpoint.
- Initial RGB ridge candidates emitted float32 ill-conditioning warnings. The final implementation standardizes and solves in float64. Appearance-only job199292 preserves the initial JSON and reuses semantic results/maps/embeddings; the split and alpha grid are unchanged. New-encoder CPU evaluations use float64 directly.
- Dense label-efficiency ablation is deferred to bound costs; no extra datasets
  launched. The reusable probe functions accept any maps/masks/split-ID adapter.
- Reporting files were extended during implementation; numerical task modules and
  source/cache identities are recorded per stage. Incompatible restarts fail closed.

## Commands

All commands run from the repository root. The launcher
`experiments/lejepa/downstream/slurm.sh` sets existing offline SPT/HF/XDG/Matplotlib
caches, deterministic CUDA workspace, and allocation-bounded BLAS threads, then
executes `.venv/bin/python -u -m MODULE ARGS`. No child jobs or arrays are submitted.
The exact training module invocation is:

```bash
sbatch --partition=gpus --gres=gpu:1 \
  --constraint='titan_rtx|rtx_a6000|l40|l40s' \
  --nodes=1 --ntasks=1 --cpus-per-task=4 --mem=24G --time=1-00:00:00 \
  --output=experiment_outputs/green-area/statpool/slurm-%j.log \
  experiments/lejepa/downstream/slurm.sh experiments.lejepa.architecture_run \
  --config experiments/lejepa/architecture/resnet9-statpool.json \
  --preflight experiment_outputs/green-area/preflight/preflight.json \
  --output experiment_outputs/green-area/statpool
```

Other module arguments, with resources/dependencies in the table above:

```text
experiments.lejepa.pooling_study --backbone resnet18 --output experiment_outputs/green-area/resnet18-pooling
experiments.lejepa.downstream.run extract
experiments.lejepa.downstream.run cpu
experiments.lejepa.downstream.run appearance
experiments.lejepa.downstream.run segmentation
experiments.lejepa.downstream.run extract --include-new --only statpool-200-seed42
experiments.lejepa.downstream.run cpu --include-new --only statpool-200-seed42
experiments.lejepa.downstream.run segmentation --include-new --only statpool-200-seed42
```

Monitoring (do not resubmit active outputs or release failed dependencies blindly):

```bash
squeue -j 199245,199251,199252,199253,199254,199255,199258,199261,199292,199293
sacct -j 199240,199242,199245,199247,199249,199251,199252,199253,199254,199255,199258,199261,199292,199293 \
  --format=JobID,State,Elapsed,AllocTRES,MaxRSS,ExitCode
cat experiment_outputs/green-area/statpool/progress.json
cat experiment_outputs/green-area/statpool/resources.json
tail -f experiment_outputs/green-area/statpool/training.log
tail -f experiment_outputs/green-area/downstream/extract-199258.log
tail -f experiment_outputs/green-area/downstream/cpu-199251.log
tail -f experiment_outputs/green-area/downstream/pet-199252.log
cat experiment_outputs/green-area/downstream/*-status.json
cat experiment_outputs/green-area/downstream/REPORT.md
```

The consolidated report and model-by-task matrix update after each model/stage;
missing results are explicitly pending. Raw episodes, regressions, per-transform
correspondences, masks, candidate validation curves, and native metrics remain
available. Resource estimates must be refined from stage-status JSON and final
Slurm accounting. No final SSL/downstream results are claimed before completion.

## First downstream functional results

All four ResNet18 readouts subsequently passed every official validator check
(job199261 completed successfully); metadata is in
`experiment_outputs/green-area/resnet18-pooling/official-validation.json`.
The first real Pet encoder evaluation (original200/42) completed both layers:
mid-layer test mIoU51.75%, foreground IoU56.69%, pixel accuracy76.14%;
final-layer test mIoU51.66%, foreground IoU61.39%, pixel accuracy78.18%.
These are functional endpoints, not an encoder comparison. Both selected probe
checkpoints used only validation; test masks were scored afterward. Exact-overlap
screening excluded0 images for this dataset/challenge pair (near-duplicates remain
an unresolved limitation). The first model took12m15s including import/cache
creation/probe fits, with peak GPU allocation.316GiB; other models remain running.

## Completed downstream stages at handoff

All8 existing encoders completed cached extraction/correspondence199258 in18m52s.
Global semantic/initial appearance199251 completed in5m56s. The appearance-only
float64 correction199292 completed all8 encoders in1m09s with no conditioning
warnings; regression fixture199293 passed. The initial float32 JSON is retained
per model and excluded from the primary appearance endpoint. Official export audit
199261 completed in13m29s: all4 ResNet18 readouts passed all6 validator checks.
Report refresh199332 writes the consolidated matrix, paired changes and native-scale
profile. Segmentation199252 has completed original200 and tuned200 probes and
continues through the remaining encoders. Training199245 remains active; its3
follow-up evaluation jobs retain their success dependencies.

Initial native-max2 jitter/control results: five-shot62.25%/60.46%, retrieval
mAP.4733/.4548, global RGB R².8522/.9036, final local RGB R².7356/.7995.
This supports a semantic/appearance tradeoff in this matched single-seed comparison;
no hidden leaderboard conclusion is drawn. Exact values, repeated-episode SDs,
per-transform correspondence scores, class IoUs and remaining endpoints are in the
consolidated report and raw matrix.

## Final completion

Training199245 and follow-ups199253/199254/199255 all completed with exit0, as did
all9 encoders' semantic, correspondence, float64 appearance and Pet evaluations.
Statistical SSL used7h12m28s allocation (6.61h epoch wall time), rather than the
initial5–6h estimate. Peak allocation/reservation were10.12/13.75GiB.
Final parent-allocation accounting:9.20 GPU-hours and38.96 CPU-hours (CPU-hours
include CPUs allocated alongside GPUs; these are separate resource measures).
This includes preflight, recovered attempts, tests and completed report refreshes;
final formatting job202821 is recorded separately. All artifacts remain local.
The final report/matrix show the completed semantic/spatial/appearance tradeoffs;
no additional SSL variants or leaderboard submissions were launched.
