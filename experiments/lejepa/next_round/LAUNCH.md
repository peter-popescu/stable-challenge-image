# Next-round launch record — 2026-11-17

CLEVR scope correction (2026-11-18): the user requested a lightweight evaluation.
Full85k extraction203433 and its pending children203434–203440 were stopped.
The original encoder's completed full embedding cache is preserved; incomplete
tuned200 arrays remain isolated. Replacement CLEVR-light fixes1000 training,
500 validation and2000 test images by seed42, independent of labels/predictions,
within the existing partition roles. Both tasks and all encoders share rows.
Completed full features are sliced only after provenance/content verification;
other encoders extract3500 images. This reduces image inference~24.3-fold.
Full and lightweight protocols have distinct outputs; no full CLEVR probe results
were used to choose this subset. DTD/CIFAR and SSL training are unchanged.
Replacement launch details are in `external/clevr-light-jobs.json` and EXTERNAL.md.
Replacement prepare204017→existing extraction204018→CPU probes204019;
new ResNet18 extraction/probes204020/204021, original400204022/204023,
six-view200204024/204025 retain successful-training and subset-preparation
dependencies. Independent review and fixtures203996/204000 passed before launch.
Probe selection uses1000 training/500 validation; final fixed-lambda refit uses
their1500-image union before scoring the2000 untouched test images.

Shared storage activation: `/nbu/people/ppopescu` was provisioned and verified
from a Slurm compute node. Successful migration203401 moved6,194 completed
external files (7,585,151,437 bytes) to
`/nbu/people/ppopescu/stable-challenge-image-next-round/external`, preserving
repository paths with a symlink. Future `external-new` outputs also point to
that task directory. Every file's hash/size/metadata was checked before removing
its verified home duplicate. Audit:
`experiment_outputs/next-round/reporting/external-storage.json` and the full
`reporting/storage-migration-203401/` manifests. Home has approximately29GiB
available. Three active SSL runs remain in place; none was interrupted or moved.
The share resolves the CLEVR capacity block; its transfer and dependent evaluation
launches are recorded in `external/clevr-jobs.json` and [EXTERNAL.md](EXTERNAL.md).
Earlier storage-blocked statuses below are historical launch records.

Latest recovery: authorized cache/test/optimizer cleanup reclaimed~28.1GiB.
See [CLEANUP.md](CLEANUP.md). Original preflight203186 failed its local-space
guard before writes; replacement203228 runs against guarded repository storage.
Original held training203199/203200/203201 were replaced by203253/203254/203255
after Slurm denied release following a hold/dependency state conversion. All
children were rewired before the obsolete pending jobs were cancelled. Replacement
training has afterok203228 dependencies and no manual holds. DTD preparation203231
completed; extraction203232 is running, CPU203233 depends on it. CIFAR archive
transfer and its preparation/extraction/probe chain203240/203241/203242 are staged.
Per-dataset jobs JSON records all new-encoder follow-ons. CLEVR still awaits bulk
storage. The table below preserves the original launch IDs/history.

Existing uncommitted work is preserved. No additional repository/ancestor
AGENTS.md files were found in the checked scope; session shared-server rules
apply. The user explicitly authorized concurrent experiment branches and multiple
Slurm submissions. All neural computation, probes, tests and reporting run in
Slurm. No leaderboard submission, system modification or environment update.

## Verified provenance and choices

Checkpoint metadata and saved run configurations agree for all11 audited trained
encoders. The historical tuned400 checkpoint is **seed43**, epoch400, planned400;
it is not a controlled duration comparison with tuned200/seed42. Original200 is
seed42, LR.001, SIGReg.02. Tuned recipes use LR.003/SIGReg.005. Independent baseline
and tuned200 checkpoints at seeds43/44 also exist. See the authoritative
[manifest](../../../experiment_outputs/next-round/provenance/MANIFEST.md) and
[complete configurations, hashes, scheduler states and evidence](../../../experiment_outputs/next-round/provenance/checkpoints.json).

ResNet18 retains its exact stride1/no-maxpool stem,2/2/2/2 depths,
40/80/160/320 widths and GAP320 training. Original400 retains historical max2
training. Both400 schedules start from scratch. Six-view200 uses the prepared
configuration, two96px globals, six48px locals and batch128. No smaller crops.
Full details and pinned-objective accounting are in [TRAINING.md](TRAINING.md).

Frozen ResNet18 dimensions600/960/920/960 match the proposal. ResNet9 actual
meaningful stages have64/128/256/256 channels: all-stage means704;
last-two mean+max1024; intermediate mean+final mean+max768; intermediate mean+
final mean+max+std1024. Native reference readouts and both random architectures
are retained. Mid/final caches are identity/hash checked; earlier stages are
extracted once. Concatenation never aligns or averages encoder coordinates.

The ensemble matrix includes all five requested cross-recipe pairs, independently
trained tuned200 seeds43+44, and duplicated tuned200/42. Held-out predictions,
per-class/per-fold error partitions and diagnostic oracle accuracy are saved.
Selected ONNX exports reserve original+statistical as a scientifically distinct
pair alongside the strongest local combinations. Selection uses challenge
development data, never external test scores. See [FROZEN.md](FROZEN.md).

## Allocations

| Job | Purpose | Resources / limit | Dependency |
|---:|---|---|---|
|203177|11-checkpoint and pinned-objective audit|2CPU8G/20min|Completed0|
|203186|All3 exact-batch memory/numerical/ONNX preflights and existing tests|1×48GiB GPU,4CPU32G/1h|afterok203177; queued for resources at launch|
|203187|Frozen stage/readout/duplicate/oracle fixtures|4CPU8G/20min|Passed3 tests|
|203189|Main frozen feature extraction|1 compatible GPU,4CPU24G/2h|afterok203187; completed0|
|203198|Supplemental tuned200/43 extraction|1 compatible GPU,4CPU16G/30min|Independent|
|203190|Cached global suite and ensemble complementarity|4CPU24G/6h|afterok203189:203198|
|203191|Selected ONNX export, official checks, CPU latency/memory|4CPU16G/1h|afterok203190|
|203199|ResNet18 tuned400 from scratch|1×24+GiB GPU,4CPU24G/24h|afterok203186|
|203200|ResNet9 original400 from scratch|1×24+GiB GPU,4CPU24G/24h|afterok203186|
|203201|ResNet9 six-local200 from scratch|1×48GiB GPU,4CPU32G/24h|afterok203186|
|203202/203205|New ResNet18 frozen extraction / global suite|1GPU4CPU24G/2h;4CPU24G/4h|203199→203202→203205|
|203203/203206|New original400 frozen extraction / global suite|Same|203200→203203→203206|
|203204/203207|New six-view200 frozen extraction / global suite|Same|203201→203204→203207|
|203192/203194/203196/203208|External labels, splits, leakage, primal/dual, DTD path fixtures|CPU allocations|Passed|
|203193/203195|Initial consolidated/external reports|CPU allocations|Completed0|
|203209|New-module syntax/report integration|2CPU4G/10min|Passed|
|203210|Consolidated report after frozen/export branch|2CPU4G/10min|afterany203191|
|203211|Consolidated report after new global evaluations|2CPU4G/10min|afterany203205:203206:203207|

Training allocations have successful-preflight dependencies **and** exact config/
source/hash guards. Submission of their allocations is not a claim that preflight
has passed or training has begun. If preflight fails, training must remain blocked;
do not release its dependencies without fixing and repeating preflight.

Current estimates from prior TITAN RTX runs:12–14h per400-epoch model,
10–14h for six-view200 (unmeasured), plus preflight/evaluation overhead. All reserve
24h. The six-view48GiB allocation preserves batch128. Actual memory/step timing
will appear in each run's `preflight/preflight.json`; no memory success is claimed
while preflight is resource-pending. Resource files and cumulative-time curves
separate training seconds from allocated GPU time. Retain every50 epochs through
400; intermediate200 checkpoints preserve their400-epoch cosine horizon.

## External status and deviations

External suite is implemented and CPU fixtures pass. The fixed matrix covers
eight existing encoders,31 unique single readouts and two prespecified ensembles;
DTD official splits1–3, CIFAR100 full/few-shot, CLEVR count/distance. CLEVR follows
VTAB labels and partition roles but explicitly uses filename ordering for its
training/validation subdivision. Scalable primal ridge matches the dual solution
numerically and selects hyperparameters only on validation. See
[EXTERNAL.md](EXTERNAL.md) for precise preprocessing, splits and acquisition chain.

**The user approved all three downloads.** A subsequent capacity check found
only1.9GiB free in home. The user authorized scratch; shared scratch location is
being resolved before any archive transfer. Brown's [filesystem documentation](https://cs.brown.edu/about/system/services/files/filesystem/)
identifies `/nbu` as networked scratch and `/ltmp` as machine-local. Personal
`/nbu/people/ppopescu` is absent and its parent is not writable; a concrete shared
alternative has been presented to the user. Do not use per-host `/ltmp` as if it
were shared across the queued jobs.

Training203199/203200/203201 are temporarily **held for storage**. Preflight203186
was held, then safely released with node-local `/ltmp` staging: full tests,
temporary checkpoints, ONNX and caches remain on its allocated GPU node. Only
small JSON, validator logs and bounded test logs are copied home. The exact path
and hostname are recorded in `preflight-staging-203186.json`. Preflight requires
8GiB local free space and256MiB home for metadata; training requires8GiB at its
target before writes, including milestones. External acquisition/cache guards
reserve more. No prior datasets/checkpoints/caches were deleted. After shared
storage relocation, release only the three remaining task-owned training holds;
successful-preflight dependencies must remain.

Bounded compute-node network probe203213 timed out after10 seconds. Acquisition
therefore has a prepared transfer-only login fallback; decompression, verification,
preprocessing, extraction and probes still run in Slurm. No primary external
jobs have started without data. External test values remain pending. NYU masked
linear probe and protocol are prepared in [NYU.md](NYU.md); synthetic valid-mask,
split-isolation and test-leakage fixtures203212 passed. No real depth experiment.

Supplement203198 was necessary because the main extractor imported its model list
before adding tuned43. It has its own extraction plan/status and completes the
independent-seed control without re-extracting completed encoders. No checkpoint
or prior result was relabeled. Current results are incremental, not final findings.

## Monitoring

From the repository root:

```bash
squeue -u "$USER"
sacct -j 203177,203186,203187,203189,203198,203190,203191,203199,203200,203201,203202,203203,203204,203205,203206,203207 --format=JobID,State,Elapsed,AllocTRES,MaxRSS,ExitCode
tail -f experiment_outputs/next-round/training/preflight-203186.log
tail -f experiment_outputs/next-round/training/resnet18-tuned-400/training.log
tail -f experiment_outputs/next-round/training/resnet9-original-400/training.log
tail -f experiment_outputs/next-round/training/resnet9-more-views-200/training.log
tail -f experiment_outputs/next-round/frozen/cpu-203190.log
```

Do not resubmit active output directories. Follow-on runners require exact final
checkpoint config/epoch before extraction. Consolidated artifacts are
`experiment_outputs/next-round/{REPORT.md,performance_matrix.csv,spatial_matrix.json,
dense_matrix.json,external_matrix.json,artifacts.json,candidate_shortlist.json}`.
Candidate shortlist entries require completed official checks and cached numerical
parity. File sizes alone do not establish submission readiness. No hidden gain is
inferred, and no heterogeneous task metrics are collapsed into one scalar.
