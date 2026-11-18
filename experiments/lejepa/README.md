LeJEPA + ResNet-9 baseline
=========================

The completed [next round](next_round/LAUNCH.md) covers longer ResNet9/ResNet18
training, six-local-view training, frozen multiscale and multi-encoder readouts,
and DTD/CIFAR-100/[lightweight CLEVR](next_round/CLEVR_LIGHT.md) probes.
The [consolidated results](../../experiment_outputs/next-round/FINAL_REPORT.md)
and external matrices are generated artifacts under `experiment_outputs/` and
remain excluded from Git. External datasets and caches use the repository's
shared-storage symlinks; checkpoints and submission files are also excluded.

The [ResNet18 and frozen pooling study](architecture/EXPERIMENTS.md) matches the
teammate's width40 BasicBlock backbone with GAP320 and export-only padding,
and compares six parameter-free readouts on four frozen ResNet9 checkpoints.
It records architecture/resource preflight, reproducible launches and reports.

The prepared [augmentation ablation proposal](augmentations/PROPOSAL.md) documents
the exact current policy, pinned reference comparisons, primary local-min10 and
jitter-strong interventions, optional local-min50/more-views specs, actual control
LR-history verification, resource estimates, and one-allocation launch commands.
The primary pair is submitted as replacement job180196; optional follow-ups remain
unsubmitted. Job180053 stopped before training on temporary-file cleanup. A short
2080 Ti fit test passed, so the replacement accepts 2080 Ti, TITAN RTX, RTX A6000,
L40 or L40S with batch128 and FP32 unchanged.

Run commands from the repository root. Training, sweeps, seed repeats and
checkpoint evaluations have been exercised through Brown CS Hydra Slurm.
This implementation targets the existing lockfile: stable-pretraining 0.1.8,
with its installed `global_views`/`local_views` interface. No dependencies were
added: Lightning and timm are already locked transitive baseline dependencies.
Use `uv sync --frozen` to provision the existing project environment on a host
where dependency installation is permitted, with UV_CACHE_DIR on approved storage.
An offline evaluator needs that environment provisioned in advance; runtime uses
no pretrained weights, remote logging, downloads, or other training datasets.


Current results and tracked artifacts
-------------------------------------

All experiments train from random initialization on the provided 7,500 images.
Labels are removed from SSL batches and used only for probes/analysis. The
current longer-duration configuration is `config_400_seed43.json`; the original
baseline remains `config.json`.

| Recipe | Training seeds | Five-fold accuracy | Official local LOO |
|---|---|---:|---:|
| Original 200-epoch baseline | 42, 43, 44 | 69.47% mean, 0.81 pp seed SD | 68.44% mean |
| Tuned 200 epochs (LR .003, SIGReg .005) | 42, 43, 44 | 72.27% mean, 0.71 pp seed SD | 71.69% mean |
| Tuned 400-epoch schedule | 43 | 76.00% | 73.47% |

These are development scores using the same fixed folds, not held-out or
new-class test results. Seed 42 was used for tuning. The 400-epoch run changes
the entire cosine schedule and is a single-seed comparison. Checkpoint choice
by these scores is additional model selection. No leaderboard score is inferred.

This branch tracks code, configurations, launch scripts, tests and this guide.
Datasets, model weights/ONNX files, caches and raw plots/logs/results under
`experiment_outputs/` are deliberately not committed. Supply the training
parquet and provision the locked environment to reproduce the experiments.
The remaining sections document runnable workflows and historical launch IDs;
the IDs and existing artifact paths are specific to the development checkout.

Cluster setup
-------------

The available global instructions require all training through Slurm and prohibit
unapproved writes outside this repository. Hydra partition and GPU feature names in the commands below were checked
against live Slurm settings. Storage remains repository-local under
experiment_outputs/. Recheck availability and policies before launching on
another system. `run_slurm.sh` leaves resource requests to sbatch arguments. Set UV_BIN to the
installed uv executable (not present on this session's PATH), and
EXPERIMENT_STORAGE to an approved directory. For this task it must be inside the
repository unless you explicitly authorize external storage. Set HF_HOME,
XDG_CACHE_HOME and UV_CACHE_DIR there before any Python/uv command as well.
The launcher sets these variables for training. Do not put checkpoints on a
login node's temporary filesystem. Keep caches and outputs out of version control.

Example environment (replace the storage choice if cluster policy requires it):

```bash
export EXPERIMENT_STORAGE="$PWD/experiment_outputs"
export UV_BIN=/cs/home/ppopescu/.local/bin/uv
export UV_CACHE_DIR="$EXPERIMENT_STORAGE/cache/uv"
export SPT_CACHE_DIR="$EXPERIMENT_STORAGE/cache/stable-pretraining"
export HF_HOME="$EXPERIMENT_STORAGE/cache/huggingface"
export XDG_CACHE_HOME="$EXPERIMENT_STORAGE/cache/xdg"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
```

Use an existing local copy of train.parquet; the entrypoint requires `--data`
and never triggers the loader's implicit download. The full dataset is ignored
by Git and must be supplied separately on a fresh clone. On this shared cluster,
ask before downloading datasets; the local copy was downloaded with permission.

Smoke and tests (inside a CPU Slurm allocation)
----------------------------------------------

Use the site's approved `salloc`/`srun` CPU resource flags, then:

```bash
uv run --frozen --offline python -m pytest tests/test_lejepa.py -q --basetemp="$EXPERIMENT_STORAGE/pytest"
uv run --frozen --offline python -m experiments.lejepa.main --smoke --accelerator cpu --data data/sample.parquet --output "$EXPERIMENT_STORAGE/smoke"
uv run --frozen --offline python -m experiments.lejepa.export "$EXPERIMENT_STORAGE/smoke/checkpoints/last.ckpt" "$EXPERIMENT_STORAGE/smoke/model.onnx"
uv run --frozen --offline test_submission.py "$EXPERIMENT_STORAGE/smoke/model.onnx"
uv run --frozen --offline python -m experiments.lejepa.diagnostics "$EXPERIMENT_STORAGE/smoke/checkpoints/last.ckpt" --data data/sample.parquet --output "$EXPERIMENT_STORAGE/smoke/plots"
```

The tiny sample has three images, so drop_last gives one batch per epoch;
`--smoke` limits to at most two batches, uses batch 2, eight SIGReg slices,
one epoch and CPU when requested. This is a plumbing check, never a performance
experiment. Use the full training parquet to exercise two batches.
Compute tests skip outside Slurm. Existing repository tests can be run with
`uv run --frozen --offline python -m pytest tests --basetemp="$EXPERIMENT_STORAGE/pytest"`
inside the same allocation.

Full training and resume
------------------------

Copy `config.json` to a versioned experiment config to change hyperparameters.
Defaults: 200 epochs, batch 128, AdamW lr 0.001, weight decay 0.0001,
10-epoch linear warmup followed by cosine decay, float32, seed 42,
SIGReg lambda 0.02, 1024 slices, upstream default projector and quadrature.
Two independent global views at 96 pixels use area scales [0.6, 1.0]; two
48-pixel local views use [0.3, 0.6]. Aspect ratio [0.9, 1.1], horizontal flip
0.5, vertical flip 0, brightness/contrast/saturation jitter 0.2 and hue 0.02
with probability 0.5, grayscale probability 0.05. These moderate starting
settings still need domain-aware ablations: crops and colors may remove useful
astronomical or satellite information. ImageNet normalization occurs after
augmentation, never inside the encoder.

Supply site-approved single-GPU flags in place of the placeholder below:

```bash
sbatch <approved single-GPU resource flags> --export=ALL experiments/lejepa/run_slurm.sh --data data/train.parquet --output "$EXPERIMENT_STORAGE/baseline"
# Resume with the identical config and total epoch budget:
sbatch <approved single-GPU resource flags> --export=ALL experiments/lejepa/run_slurm.sh --data data/train.parquet --output "$EXPERIMENT_STORAGE/baseline" --resume "$EXPERIMENT_STORAGE/baseline/checkpoints/last.ckpt"
```

CLI accepts `--config`, `--data`, `--output`, `--resume`, `--accelerator`,
and `--smoke`. Single-device training is deliberate. Lightning stores model,
optimizer, scheduler and SIGReg projection counter in epoch checkpoints.
Epoch-based seeding restarts shuffle/augmentations reproducibly at epoch
boundaries; bitwise mid-epoch recovery is not guaranteed. Deterministic kernels
are requested; reproducibility across hardware/software releases is not promised.
CSV logs contain step/epoch total, invariance and SIGReg losses and LR.
`resolved.json` records configuration, dataset SHA256, package versions, Python,
Torch, git revision and tracked diff. A source/config/lock snapshot is saved.
Resume requires matching config; use a separate output directory for new runs.

Export and official evaluation
------------------------------

Run these in an appropriate Slurm CPU allocation:

```bash
uv run --frozen --offline python -m experiments.lejepa.export "$EXPERIMENT_STORAGE/baseline/checkpoints/last.ckpt" "$EXPERIMENT_STORAGE/baseline/model.onnx"
uv run --frozen --offline test_submission.py "$EXPERIMENT_STORAGE/baseline/model.onnx"
uv run --frozen --offline evaluate.py "$EXPERIMENT_STORAGE/baseline/model.onnx" --data data/train.parquet --out "$EXPERIMENT_STORAGE/baseline/development_probe.json"
```

The timm registry adapter subclasses the installed Resnet9, replacing `fc`
with Identity and exposing embed_dim=1024. Its original conv/flatten forward,
including adaptive **max** pooling to 2x2, is preserved. Training calls the
actual upstream LeJEPA forward/loss without reimplementation. Only the learned
backbone is exported using the repository utility, with no projector or extra
normalization. Export asserts <20 MiB and compares CPU ONNX/PyTorch results
for dynamic batches 1, 3, 4. The official validator also evaluates sample images.
The encoder has roughly 2.4 million parameters; actual file size must be checked
by export/validator. Labels are removed from the training dataset before batching.
The unchanged official evaluator supplies the ridge metric. A score on labeled
training images is a development score, especially if used to select configs;
it is not unbiased held-out or new-class performance.

Diagnostics (no retraining)
--------------------------

```bash
uv run --frozen --offline python -m experiments.lejepa.diagnostics "$EXPERIMENT_STORAGE/baseline/checkpoints/last.ckpt" --data data/train.parquet --output "$EXPERIMENT_STORAGE/baseline/plots"
# Optional common basis and common 1st/99th percentile scaling across images:
uv run --frozen --offline python -m experiments.lejepa.diagnostics "$EXPERIMENT_STORAGE/baseline/checkpoints/last.ckpt" --data data/train.parquet --output "$EXPERIMENT_STORAGE/baseline/shared" --shared
```

Default selection is 12 seeded random rows. Domain IDs are absent; inspect the
contact sheet and use `--indices 1,23,456,...` to freeze a visually diverse sample
covering domains; no automatic domain coverage is claimed. `--count` and `--seed`
are configurable. Spatial maps come immediately before adaptive pooling,
[B,256,12,12]. Per-image PCA is the default. Shared mode fits up to 10,000
seeded spatial vectors. Both resolve PCA signs deterministically and save
explained variance ratios and source row IDs in diagnostics.json. spatial.png
shows original, trained features, overlay, seeded untrained features and raw
RGB PCA. Learned/random models each have their own PCA: colors are not aligned
semantic labels. Percentile scaling can exaggerate tiny differences. These are
qualitative features, not segmentation masks. neighbors.png shows each query
followed by three cosine neighbors excluding itself. Embedding mean variance,
entropy effective rank of centered singular values and embeddings.npz help
spot collapse; rank is limited by the small diagnostic sample. Increase count
for a more representative estimate in a compute allocation.

Verification status
-------------------

Syntax compilation, shell syntax and whitespace checks passed. Slurm job 156147
passed all 57 tests (including encoder/loss/backward and ONNX tests) before
reaching its 20-minute limit. Job 156318 completed the remaining pipeline in
21m47s: CPU smoke training, encoder export, dynamic-batch numerical parity,
official submission validator, sample probe evaluation, per-image and shared
PCA, retrieval, and checkpoint restoration. Export size was approximately
9.3 MiB. Resume restored state at the completed epoch budget; it did not
exercise continued optimization. The smoke dataset contained only three images;
no meaningful model performance or full-data evaluation is claimed.

The logs revealed that stable-pretraining uses SPT_CACHE_DIR independently of
XDG_CACHE_HOME and wrote environment metadata to its default home cache during
verification. Launchers and the model import now set a repository-local SPT cache
for subsequent runs. Existing external cache files were left untouched.
This paragraph describes the initial smoke verification. Later training and
analysis procedures are documented below; leaderboard uploads are performed
by the user, not by these scripts.

Live training dashboard
-----------------------

Job180196 also has a separate SPT telemetry run, `gpu-memory-job-180196`,
under `experiment_outputs/telemetry/`. It plots allocated-device VRAM used/free,
total capacity, memory percentage and GPU utilization every10 seconds. The x-axis
step is elapsed wall seconds since monitoring started. Select this run in SPT
or filter metrics by `gpu_memory`. The collector is
`monitoring/gpu_memory.py`, running as overlapping step180196.1 inside the existing
allocation, without importing Torch or editing the frozen training pipeline.
These are sampled nvidia-smi device measurements, not PyTorch allocated/reserved
values or guaranteed transient peaks. Earlier memory history cannot be recovered.

For a future existing GPU allocation, start one monitor with a fresh output path:

```bash
srun --jobid=JOB_ID --overlap --exact --nodes=1 --ntasks=1 \
  --cpus-per-task=1 --gres=gpu:1 --immediate=10 \
  .venv/bin/python -u monitoring/gpu_memory.py \
  --output experiment_outputs/telemetry/gpu-memory-job-JOB_ID --interval 10
```

The collector refuses multiple GPUs and duplicate CSV writers. It exits with the
allocation; a terminated monitor retains its recorded history. A low usage trace
during checkpoint CPU evaluation is expected.

New training runs use stable-pretraining's RegistryLogger. The run directory
`<output>/metrics/` contains metrics.csv, sidecar.json, summary.json and heartbeat.
Total/invariance/SIGReg losses, LR, epoch and config are shown by `spt web`.
CSV flush cadence is 10 steps plus epoch boundaries; the viewer polls every
five seconds. Resume appends to the same metrics directory and uses the same
output-derived run ID. Keep a distinct output directory for each experiment.
The completed smoke CSV was registered so the dashboard has a real example;
its single data point does not describe a learning curve.

Start the viewer on aramis (no training is performed; initial package imports
can take several minutes on the shared filesystem):

```bash
.venv/bin/python -m experiments.lejepa.dashboard
# Equivalent viewer through the standard CLI:
# uv run --frozen --offline spt web experiment_outputs --host 127.0.0.1 --port 4242
```

The helper binds only to 127.0.0.1:4242. On your own computer, open a terminal:

```bash
ssh -N -J ppopescu@fastx-cluster.cs.brown.edu -L 4242:127.0.0.1:4242 ppopescu@aramis.cs.brown.edu
```

Keep that SSH connection open, then visit http://localhost:4242 in your browser.
Use your usual university SSH route/VPN if required; no SSH config changes are
needed. If local port 4242 is busy, forward `4243:127.0.0.1:4242` and browse
http://localhost:4243. Training can run on a Slurm GPU node while this viewer
reads the shared repository filesystem from aramis. The tunnel destination must
match the host on which you started the viewer.

Use a detached tmux session if the viewer should survive terminal disconnection.
For a repository-local socket and no tmux configuration changes:

```bash
mkdir -p experiment_outputs/dashboard
tmux -S "$PWD/experiment_outputs/dashboard/tmux.sock" -f /dev/null new-session -d -s spt-dashboard -c "$PWD" 'exec .venv/bin/python -u -m experiments.lejepa.dashboard >> experiment_outputs/dashboard/server.log 2>&1'
```

Restart the viewer if the host or process is terminated.
Spatial PCA/retrieval PNGs remain standalone diagnostics; they are not currently
published to the dashboard's media tab. No external service or account is used.


First full training launch
--------------------------

After authorization to download the official dataset and start training,
job 156567 was submitted on the observed `gpus` partition with
`--gres=gpu:nvidia_titan_rtx:1 --cpus-per-task=4 --mem=32G --time=12:00:00`.
Slurm placed it on gpu1907. Config is the unchanged 200-epoch baseline;
output is `experiment_outputs/lejepa-resnet9-seed42`. This is a launch record,
not a claim of successful completion. The launcher uses the tested existing
.venv through `uv run --frozen --offline --no-sync`; provision it with
`uv sync --frozen` before reproducing on another machine. CUBLAS_WORKSPACE_CONFIG
is set to support deterministic GPU matrix multiplication. Downloaded official
train.parquet size: 122,020,238 bytes. The full run records its SHA256 and row
count in resolved.json. Logs and checkpoints stay in the run directory;
the dashboard discovers its metrics directory automatically.


GPU determinism fix and replacement launch
------------------------------------------

Job 156567 failed on its first backward pass: PyTorch's CUDA implementation
of adaptive_max_pool2d backward rejects strict deterministic algorithms.
The encoder now uses exactly the same adaptive 2x2 bins, with indexed max
reductions instead of the unsupported CUDA pooling backward kernel. First-max
tie handling and odd-size overlapping bins are retained. There are no learned
parameters or checkpoint keys in either pooling implementation. Strict
determinism remains enabled; the LeJEPA loss is unchanged.

Replacement job 156576 uses the same config/resources and a fresh output
directory `experiment_outputs/lejepa-resnet9-seed42-retry`, preserving the failed
run. Within this one allocation it first runs `tests/test_lejepa.py`, including
strict CPU/GPU LeJEPA backward, export parity and six pooling comparisons of
GPU values/gradients against CPU adaptive pooling (even and odd shapes,
random data and tied maxima). Training starts only if these checks pass.
This historical launch later completed 200 epochs; its local official LOO
score was 68.4% and fixed five-fold score was 70.0%.
Future launches can set LEJEPA_PREFLIGHT=1 to run these checks before training.


Error analysis
--------------

The error-analysis module reuses evaluate.embed and evaluate.Ridge without
changing the official probe. It computes per-class support/accuracy, macro
accuracy, raw and row-normalized confusion, largest directed confusions,
paired fixes/regressions relative to an untrained model, and standardized
cosine retrieval among all labeled images excluding the query itself.
A third probe uses six global RGB channel means/standard deviations. This
is a low-level reference, not proof that the trained encoder ignores color.

```bash
LEJEPA_ANALYSIS_MODULE=experiments.lejepa.error_analysis sbatch --partition=compute --nodes=1 --ntasks=1 --cpus-per-task=2 --mem=16G --time=02:00:00 --export=ALL --output=experiment_outputs/lejepa-resnet9-seed42-retry/error_analysis/slurm-%j.log experiments/lejepa/report_slurm.sh --analysis experiment_outputs/lejepa-resnet9-seed42-retry/analysis --checkpoint experiment_outputs/lejepa-resnet9-seed42-retry/checkpoints/last.ckpt --data data/train.parquet --output experiment_outputs/lejepa-resnet9-seed42-retry/error_analysis
```

Create the output directory before submitting. Results include REPORT.md,
index.html, summary.json, predictions.csv, per_class.csv, confusion_pairs.csv,
embeddings.npz and confusion matrices. Seeded error/fix/regression and
per-confusion galleries include class IDs, source row IDs and nearest neighbors.
Per-class contact sheets support manual identification of domains; numeric
class IDs are not assumed to map to domains. Focused per-image/shared spatial
PCA panels cover examples from the largest confusion pairs. Their untrained
reference is initialized explicitly with the same seed as the exported random
baseline. These are development analyses, not outer cross-validation or
held-out scores. The Slurm wrapper first runs targeted accounting/retrieval
tests and starts analysis only if those pass. No model is retrained.


Fixed five-fold probe validation
--------------------------------

The new cross_validate entrypoint evaluates a frozen ONNX encoder using the
unchanged evaluate.Ridge. It does not train the encoder. For each outer fold,
Ridge.fit sees only the outer training embeddings/labels; it fits their
standardization and selects regularization using the original analytic LOO.
Ridge.predict then predicts the outer validation embeddings. In the 750-image
baseline there are five 600/150 train/validation splits. Inner standardization
still follows the official implementation; only the outer validation split is
fully excluded from fitting. The official local full-data LOO score is also
reported separately, without modifying evaluate.py.

The stratifier shuffles sorted row IDs independently within each class using
seed 42, then assigns folds by a rotating round robin. Total and per-class fold
sizes differ by at most one. Every observed class must have at least five
labeled examples. The manifest records the algorithm, seed, full parquet SHA256,
and exact row ID/label/fold assignments. Existing manifests are validated and
never overwritten. Use the identical --folds file for future SSL configurations;
do not search fold assignments or seeds based on performance.

```bash
mkdir -p experiment_outputs/lejepa-resnet9-seed42-retry/cross_validation
LEJEPA_ANALYSIS_MODULE=experiments.lejepa.cross_validate sbatch --partition=compute --nodes=1 --ntasks=1 --cpus-per-task=2 --mem=16G --time=02:00:00 --export=ALL --output=experiment_outputs/lejepa-resnet9-seed42-retry/cross_validation/slurm-%j.log experiments/lejepa/report_slurm.sh --model experiment_outputs/lejepa-resnet9-seed42-retry/analysis/model.onnx --untrained-model experiment_outputs/lejepa-resnet9-seed42-retry/analysis/untrained.onnx --data data/train.parquet --folds experiment_outputs/validation/stratified-5fold-seed42.json --output experiment_outputs/lejepa-resnet9-seed42-retry/cross_validation
```

The exported model is the already validated encoder from last.ckpt. No encoder
weights are changed. Optional --untrained-model supplies a matched comparison;
six RGB means/stds are evaluated as a low-level reference. ONNX extraction uses
the official preprocessing and bounded CPU threads. Results include REPORT.md,
results.json, fold_metrics.csv (including selected lambda per fold), per_class.csv,
predictions.csv (one outer prediction per image), embeddings.npz and confusion
matrices. Model, data and fold-manifest hashes are recorded. Fold SD is descriptive,
not a confidence interval. Tests cover split balance, row-order independence,
manifest immutability, per-fold training-row isolation, and unchanged fitted
standardization/lambda when held-out features and labels are perturbed.

The encoder has seen all these images without labels during SSL. This measures
probe generalization, not generalization to images excluded from SSL or new
classes. Once reused for selecting SSL configurations, the folds are development
data, not an unbiased final test. The outer score also uses fewer probe training
images than official LOO, so differences cannot be attributed only to tuning bias.
No additional dependencies or changes to the supplied evaluator are required.


Sequential LR/SIGReg screening sweep
------------------------------------

`sweep.json` defines nine LR/weight combinations: learning rates 0.0003,
0.001, 0.003 crossed with SIGReg weights 0.005, 0.02, 0.08. The baseline
control runs first. All configurations use seed 42, batch 128, identical views
and augmentations, 1024 SIGReg slices and the same 200-epoch schedule with
10-epoch warmup. Each run stops at 100 epochs; --stop-after-epochs limits
Trainer execution without changing the configured LR horizon. --retain-every 50
keeps milestone checkpoints at completed epochs 50 and 100 in addition to the
latest/last checkpoints. Checkpoint filenames use zero-based epoch numbers,
so epoch 049 is the checkpoint after 50 complete epochs.

Create the output directory and launch exactly one sequential allocation:

```bash
mkdir -p experiment_outputs/sweeps/lejepa-lr-sigreg-seed42
EXPERIMENT_STORAGE="$PWD/experiment_outputs" sbatch --partition=gpus --gres=gpu:nvidia_titan_rtx:1 --nodes=1 --ntasks=1 --cpus-per-task=4 --mem=32G --time=2-00:00:00 --job-name=lejepa-sweep --export=ALL --output=experiment_outputs/sweeps/lejepa-lr-sigreg-seed42/slurm-%j.log experiments/lejepa/sweep_slurm.sh --data data/train.parquet --folds experiment_outputs/validation/stratified-5fold-seed42.json --output experiment_outputs/sweeps/lejepa-lr-sigreg-seed42
```

The existing environment must already be provisioned with `uv sync --frozen`;
this launcher uses its .venv directly and never installs dependencies/downloads.
Preflight tests run inside the allocation before any screen. Tests include real
CPU checkpoint continuation with a nonzero LR at the screening stop and the
original schedule preserved, milestone retention, fixed-fold isolation and
GPU forward/backward/export checks. Training, CPU export and fold evaluation
are sequential in the same allocation, one configuration at a time. Each run
has its own dashboard metrics, source/config snapshot and training/export/
validation logs. No Slurm arrays or child jobs are submitted.

At the sweep root:

- plan.json freezes configs, source hashes, dataset hash and fold-manifest hash.
- progress.json records each configuration's current phase/completion/failure.
- comparison.csv and REPORT.md are updated after each phase; completed runs are
  ranked by outer accuracy, with macro accuracy as a secondary metric, and
  paired per-fold differences against the matched 100-epoch control.
- source/ and folds.json preserve the implementation and fold assignments.

An individual failed subprocess is logged and the runner proceeds to remaining
configurations; the final job returns nonzero if any configuration failed.
Nonfinite SSL loss stops that run. If interrupted, resubmit the same command
and output directory after the old job has stopped. The runner validates its
immutable plan, restores incomplete runs from last.ckpt, verifies completed
checkpoint budgets, and reuses exports/CV only with matching checkpoint/model/
data/fold hashes. Changed code or plans require a new output directory. Do not
run two instances against the same sweep directory concurrently.

Review the screening table before extending candidates. To extend a selected
run use its config.json (still epochs=200), --resume its last.ckpt, and omit
--stop-after-epochs. Preserve --retain-every 50. This restores optimizer,
scheduler and SIGReg state and continues to epoch 200. The sweep does not
extend configurations or submit anything to the leaderboard automatically.
Fold SD is descriptive and reused folds are development data. One-seed screening
is not evidence of a seed-robust improvement.


Sweep resource broadening (job 159667)
--------------------------------------

While still pending, this job was updated in place to `--gres=gpu:1`,
`--constraint='gtx_2080_ti|titan_rtx|rtx_a6000|l40|l40s'`, and `--mem=24G`.
CPU count, 48-hour wall time, batch size and scientific configuration are unchanged.
Live node features were inspected before specifying this constraint. The wider
request includes idle 2080 Ti nodes. Pascal GTX 1080/1080 Ti nodes are excluded
because the installed PyTorch uses CUDA 13, which removed support for that
architecture. A new full batch-128 multiview forward/backward/optimizer-step
preflight runs before the sweep. It records GPU VRAM usage in
`experiment_outputs/gpu-memory-preflight-159667.json` on success; if the batch
does not fit, preflight fails without starting the nine training configurations.
The update did not submit a second job. Different GPU types can produce numeric
variation; all nine configurations still share one allocation and GPU type.


Continue the two selected screens to epoch 200
----------------------------------------------

`extend.py` resumes selected completed screening runs into a separate output
root. The 100-epoch parent checkpoint/export/CV scores are kept untouched.
Configuration, source checkpoint hash and screening-score provenance are
validated. Optimizer, scheduler, SIGReg counter and loop state are restored;
no warmup or training is restarted. The configured schedule stays at 200 epochs.
New dashboard runs show epochs 100 through 199, with parent curves preserved.

```bash
mkdir -p experiment_outputs/sweeps/lejepa-lr-sigreg-seed42/continued-200
EXPERIMENT_STORAGE="$PWD/experiment_outputs" LEJEPA_SWEEP_MODULE=experiments.lejepa.extend sbatch --partition=gpus --gres=gpu:1 --constraint='gtx_2080_ti|titan_rtx|rtx_a6000|l40|l40s' --nodes=1 --ntasks=1 --cpus-per-task=4 --mem=24G --time=12:00:00 --job-name=lejepa-extend --export=ALL --output=experiment_outputs/sweeps/lejepa-lr-sigreg-seed42/continued-200/slurm-%j.log experiments/lejepa/sweep_slurm.sh --sweep experiment_outputs/sweeps/lejepa-lr-sigreg-seed42 --runs lr-0.003_lambda-0.005 lr-0.001_lambda-0.005 --data data/train.parquet --folds experiment_outputs/validation/stratified-5fold-seed42.json --reference-results experiment_outputs/lejepa-resnet9-seed42-retry/cross_validation/results.json --output experiment_outputs/sweeps/lejepa-lr-sigreg-seed42/continued-200
```

Only one allocation runs the two continuations sequentially, with preflight
tests before training. Checkpoints are retained every 50 epochs (150/200 in
this segment). Each final encoder is exported with numeric parity checks and
scored on the same fixed folds. REPORT.md/comparison.json compare 100 and 200
epochs and the original 200-epoch baseline, including paired fold differences.
Partial work can resume by resubmitting the same command/output after the old
job has stopped. A changed plan/source/data is rejected. No seed repetitions,
automatic further training, or leaderboard submissions are performed.


Matched training-seed confirmation
-----------------------------------

`seeds.py` trains the original baseline (LR 0.001, lambda 0.02) and selected
configuration (LR 0.003, lambda 0.005) from scratch with seeds 43 and 44.
Everything else remains fixed, including 200 epochs, batch 128, augmentation,
projector, optimizer schedule and fold manifest. Existing seed-42 scores are
included as references without retraining. A matched seed uses the same seed
for both configurations. These runs do not resume any seed-42 weights.

```bash
mkdir -p experiment_outputs/seed-repeats/baseline-vs-tuned
EXPERIMENT_STORAGE="$PWD/experiment_outputs" LEJEPA_SWEEP_MODULE=experiments.lejepa.seeds sbatch --partition=gpus --gres=gpu:1 --constraint='titan_rtx|rtx_a6000|l40|l40s' --nodes=1 --ntasks=1 --cpus-per-task=4 --mem=24G --time=1-12:00:00 --job-name=lejepa-seeds --export=ALL --output=experiment_outputs/seed-repeats/baseline-vs-tuned/slurm-%j.log experiments/lejepa/sweep_slurm.sh --baseline-config experiments/lejepa/config.json --winner-config experiment_outputs/sweeps/lejepa-lr-sigreg-seed42/continued-200/lr-0.003_lambda-0.005/config.json --baseline-reference experiment_outputs/lejepa-resnet9-seed42-retry/cross_validation/results.json --winner-reference experiment_outputs/sweeps/lejepa-lr-sigreg-seed42/continued-200/lr-0.003_lambda-0.005/validation/results.json --seeds 43 44 --data data/train.parquet --folds experiment_outputs/validation/stratified-5fold-seed42.json --output experiment_outputs/seed-repeats/baseline-vs-tuned
```

One allocation runs all four new configurations sequentially; no child jobs or
arrays are submitted. The 36-hour limit provides margin over the roughly
22-24 GPU-hour estimate. Eligible GPU types have at least 24 GiB VRAM; the
previous full-batch preflight peaked at 10.0 GiB active/11.75 GiB reserved, so
11 GiB cards are excluded for this confirmation batch. Preflight tests, including
matched-seed aggregation and real continuation checks, run before training.
All training outputs, caches and test artifacts remain in the repository.

Each run has its own config/checkpoints, dashboard entry, export and fixed-fold
validation. Milestones at epochs 50/100/150/200 are retained. Progress and hashes
allow a stopped job to resume with the same command/output; a changed plan is
rejected. The source, reference CV reports and folds are preserved at the root.
Individual subprocess failures are logged and remaining repeats continue.

REPORT.md/comparison.json show per-seed outer/macro/official LOO scores, mean
and sample SD across matched training seeds, and winner-minus-baseline paired
differences. Aggregates use only seeds completed for both configurations. Seed
42 was used for hyperparameter selection, so a separate confirmation-only gain
for seeds 43/44 is reported. Fold SD is distinct from training-seed SD; 15 folds
from three seeds are not 15 independent training experiments. Three seeds are
a small descriptive comparison, and all scores remain development results.
No new hyperparameter search or automatic leaderboard submission occurs.


Assess the final 50 epochs without retraining
---------------------------------------------

`milestones.py` exports each available retained epoch-149 checkpoint (150
completed epochs), evaluates it on the fixed folds, and compares against the
already saved epoch-200 scores. It covers all three tuned seeds and baseline
seeds 43/44. The original baseline seed 42 did not retain epoch 150; it is
explicitly unavailable and excluded from paired aggregates. No missing
checkpoint is reconstructed or retrained.

```bash
mkdir -p experiment_outputs/milestone-evaluation/epoch150-vs200
LEJEPA_ANALYSIS_MODULE=experiments.lejepa.milestones sbatch --partition=compute --nodes=1 --ntasks=1 --cpus-per-task=2 --mem=16G --time=02:00:00 --job-name=lejepa-150-v-200 --export=ALL --output=experiment_outputs/milestone-evaluation/epoch150-vs200/slurm-%j.log experiments/lejepa/report_slurm.sh --data data/train.parquet --folds experiment_outputs/validation/stratified-5fold-seed42.json --output experiment_outputs/milestone-evaluation/epoch150-vs200
```

Root arguments --baseline-root, --winner-root and --repeat-root can override
the documented experiment directories. The analysis validates source checkpoint
epoch/seed, the 200-epoch schedule, dataset and fold hashes; exports include
numeric parity checks. Cached exports/CV are reused only with matching hashes.
Outputs are REPORT.md/results.json with paired per-seed/fold changes, aggregate
means and seed SD, macro/official LOO changes, and loss windows near epochs
150/200. Separate per-configuration artifacts retain epoch150.onnx and complete
fixed-fold predictions/metrics. Baseline and tuned means have different seed
counts, so emphasize within-seed gains. Improvement before epoch 200 does not
establish improvement beyond it. No additional training or jobs are launched
from the analysis allocation.


400-epoch duration experiment
-----------------------------

Config `config_400_seed43.json` changes only epochs=400 from the tuned seed-43
configuration (LR .003, SIGReg .005, batch 128, 10-epoch warmup). Seed 43 was
fixed before this experiment for comparison with its existing 200-epoch run,
not selected for its score. The first launch starts from random initialization
without a resume argument. Longer cosine decay changes the LR trajectory
throughout training; this tests a longer training recipe, not only extra steps
appended to the old zero-LR endpoint.

```bash
mkdir -p experiment_outputs/duration/tuned-seed43-400
sbatch --partition=gpus --gres=gpu:1 --constraint='titan_rtx|rtx_a6000|l40|l40s' --nodes=1 --ntasks=1 --cpus-per-task=4 --mem=24G --time=1-00:00:00 --job-name=lejepa-400 --export=ALL --output=experiment_outputs/duration/tuned-seed43-400/slurm-%j.log experiments/lejepa/duration_slurm.sh
```

Job 171065 is the initial submission. The one allocation runs GPU/validation
preflight tests, trains with checkpoints retained every 50 epochs, and exports
and evaluates epochs 200/300/400 on the fixed folds afterward. REPORT.md and
comparison.json compare against the original seed-43 200-epoch recipe. The
final encoder export is model.onnx at the run root; intermediate exports are
in epoch200/ and epoch300/. No leaderboard submission is made.
Resubmitting the script after a stopped allocation automatically restores this
run's own last.ckpt; it does not load another experiment's weights. Override
LEJEPA_DURATION_OUTPUT to use a different output directory. This experiment
is a single-seed development comparison; broader conclusions require repeats.


Evaluate the retained 350-epoch duration checkpoint
---------------------------------------------------

Job 173642 runs checkpoint_eval.py on retained-epoch-349-step-20300.ckpt,
verifies it contains 350 completed epochs, exports only the encoder with
numeric parity checks, and runs the unchanged official probe and fixed outer
folds. Inputs are the existing train.parquet and fixed fold manifest. No
training is performed. Output is duration/tuned-seed43-400/epoch350/ within
experiment_outputs. provenance.json records checkpoint/config/model hashes;
validation/ contains scores and predictions. MILESTONES.md and
comparison_with350.json at the duration root compare epochs 200/300/350/400
without overwriting the original duration report.

The generic entrypoint is `python -m experiments.lejepa.checkpoint_eval
--checkpoint PATH --expected-epoch N --data data/train.parquet --folds
experiment_outputs/validation/stratified-5fold-seed42.json --output DIRECTORY`.
Run it through report_slurm.sh with
LEJEPA_ANALYSIS_MODULE=experiments.lejepa.checkpoint_eval in a CPU allocation.
Optional --compare-duration-root adds the documented duration comparison.

The statistical-pooling training and broader frozen evaluation suite are documented
in [downstream/PROTOCOL.md](downstream/PROTOCOL.md), with jobs, preflight results and
monitoring in [downstream/LAUNCH.md](downstream/LAUNCH.md). The incremental report is
`experiment_outputs/green-area/downstream/REPORT.md`. External evaluation data and
probe weights remain isolated under `experiment_outputs/external-eval/`.
