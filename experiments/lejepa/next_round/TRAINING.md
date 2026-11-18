# Next-round SSL training

Replacement training jobs203253/203254/203255 are submitted with
`afterok:203228`; they cannot start until successful exact preflight.
User-authorized cleanup and archival restored30GiB free on home.
Replacement preflight203228 runs with `--repository-storage`, retaining full
artifacts under `experiment_outputs/next-round/training/preflight-work-203228`.
The original203186 attempt failed before writes: its allocated node had only
5.03GiB free in `/ltmp`, below the8GiB safety floor. Its failure log is preserved.
The original203199/203200/203201 never started: updating their dependencies while
priority0 held changed their scheduler state from user-held to admin-held, and
ordinary release was denied. Identical replacements preserve the configurations,
output directories and preflight guards. No privileged release or priority
override was attempted. At replacement submission30GiB was free, exceeding the
13GiB planned budget (8GiB safety floor plus5GiB for the three concurrent runs).
Each run also checks the8GiB floor immediately before writes.
`training/launch.json` records the complete submission/failure/replacement history.

Replacement preflight203228 completed successfully in11m23s. Existing targeted
tests passed30/30; all three exact FP32 batch128 forward/backward/AdamW checks,
finite gradients, dynamic-batch ONNX parity and official validators passed.
Measured on an NVIDIA RTX A6000:

| Recipe | Peak allocated / reserved GiB | Median synthetic step seconds | Encoder parameters | ONNX bytes |
|---|---:|---:|---:|---:|
| ResNet18 tuned400 |8.325 /8.346|.344|4,365,480|17,455,594|
| ResNet9 original400 |10.082 /10.252|.254|2,437,696|9,771,297|
| ResNet9 six-local200 |14.180 /14.252|.354|2,437,696|9,771,297|

Synthetic step timing excludes augmentation/data loading and is not a measured
end-to-end training ETA. Raw reports remain in each recipe's `preflight` directory.

The default node-local staged preflight requires8GiB free locally and256MiB free in the repository
for small reports. Tests, their checkpoints, all full ONNX files and temporary
caches stay on node-local scratch. Home receives `preflight.json`, validator
logs, a bounded test log, test XML and staging metadata. Logs copied home are
limited to1MiB each; full logs remain local. The staging metadata records hostname
and the full scratch path; each recipe's preflight JSON includes artifact paths.
With `--repository-storage`, that same staging directory is on the repository
filesystem and requires8GiB free; full test/checkpoint/ONNX artifacts stay there.
`train.py` requires only the copied JSON configuration, measurements and source
hashes; it never opens the staged ONNX file. Node-local artifacts are temporary
cluster storage rather than durable experiment results.

Completed preflight test checkpoint copies were removed under explicit user
authorization; their test logs/configurations remain. Historical retained
milestones are archived separately by removing optimizer states only, preserving
schedulers, full SSL state, config, epoch and SIGReg. Distinct `.weights.ckpt`
artifacts and `.ckpt.archived.json` mappings preserve original identities. The
central provenance manifest records old/new paths and hashes; adjacent milestone
provenance gains an explicit archive mapping while retaining the historical hash.
All final `last.ckpt` reference files remain unchanged.

`storage_guard.require_free_space` normally requires at least8GiB available on the actual
destination filesystem, following symlinks and checking the nearest existing
ancestor for a new output path. It runs before preflight tests, before individual
preflight model imports/output creation, and before training output writes and
each training/evaluation/diagnostic subprocess. Insufficient capacity raises a
clear error. The guard source hash is bound in both preflight and training plans.
This minimum is a per-process safety floor, not a reservation of shared capacity;
storage must also accommodate all concurrent jobs and their retained artifacts.

Three from-scratch recipes reuse `main`, `checkpoint_eval`, diagnostics, resource
logging and the standard pinned stable-pretraining LeJEPA method. No continuation
from historical weights is used. `train.py` retains and evaluates every 50 epochs,
including 250/300/350/400 for the longer recipes. Configuration `epochs` always
sets the complete cosine horizon, including at intermediate milestones. Epoch200
from a 400-epoch schedule is not a completed 200-epoch cosine run.

The original400 configuration is copied from the original checkpoint's historical
`resolved.json`, changing only epochs and resource logging. LR=.001, SIGReg=.02,
batch128, seed42, AdamW decay=.0001, 10-epoch warmup. ResNet18 changes only epochs
from its existing configuration: widths40/80/160/320, depths2/2/2/2, stride1
3x3 stem without max pooling, GAP320, LR=.003, SIGReg=.005, seed42. More-views is
the prepared optional configuration, plus resource logging: seed42, LR=.003,
SIGReg=.005, 200-epoch horizon, two96px globals and six48px locals with original
crop areas/photometric transforms. The exact historical config and checkpoint
metadata are cross-checked in the CPU provenance audit and preflight driver.

## What six local views changes

The pinned objective computes centers using the two global-view projections,
then averages squared deviations across **all** views, samples and projector
coordinates. Increasing locals2→6 changes global versus local coefficient totals
from 1/2 versus1/2 to1/4 versus3/4. Each global has weight1/8 rather than1/4;
the global center still depends on both global projections and retains gradients.
There is no separate local-loss multiplier or normalization introduced here.

SIGReg flattens all views: batch128 gives512 projection samples in the standard
recipe and1024 with six locals. The pinned Epps–Pulley statistic multiplies the
integrated empirical characteristic-function error by N (512→1024). Lambda stays
.005. The realized SIGReg loss need not double because empirical errors and view
correlations also change; increasing views changes finite-sample loss behavior.
The total objective remains invariance + lambda×SIGReg.

The backbone sees two forward calls per step: concatenated globals256×96², then
concatenated locals256×48² or768×48². Thus each backbone BatchNorm still updates
twice per optimizer step, but the second batch has three times as many local
images. The projector sees512 or1024 concatenated embeddings in one call; its
BatchNorm updates once per step with changed sample composition. These changes
make this a broader recipe comparison, not a pure view-diversity ablation.

Backbone input-pixel work increases from2.5 to3.5 units of128×96² per step (7/5);
projector/SIGReg samples double. These are workload proxies, not measured FLOPs
or latency. Exact-configuration preflight uses all eight views, FP32, strict
determinism, full batch128, AdamW updates and finite-gradient checks, recording
step latency and peak allocated/reserved CUDA memory. All three recipes receive
the official ONNX validator before training launch.

Final-audit correction: the original launch note incorrectly stated a5/3 pixel
work ratio. The correct ratio is `(2×96²+6×48²)/(2×96²+2×48²)=7/5=1.4`.
This documentation error did not affect any training configuration or loss.

`metrics/metrics.csv` records actual optimizer LR every step and all three loss
components; `resources.json` records per-epoch wall time, images/s, GPU identity
and peak memory; `phase_times.json` includes milestone-evaluation overhead. Use
`sacct` parent allocation time for GPU-hours, which includes setup/evaluation.
Compare learning curves versus cumulative measured epoch wall time and retain
allocation time as a separate accounting metric. Single-seed comparisons and
reused development data do not establish hidden leaderboard improvements.
