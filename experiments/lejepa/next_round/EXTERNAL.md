# External evaluation, next round

All data, cached features, fitted probe weights and predictions live under
`experiment_outputs/next-round/external`, separate from SSL and ONNX artifacts.
No external images enter training. All processing/inference/probing runs in Slurm.
Acquisition requires explicit approval before using `external_download`.

Primary sources:
- [DTD official data and partitions](https://www.robots.ox.ac.uk/~vgg/data/dtd/): use official partitions 1,2,3 of ten; 40 images/class in each train, validation, test. Three fixed splits bound repeated probe cost. Report every split and mean/SD, which are overlapping dataset splits, not independent training seeds.
- [CIFAR official data](https://www.cs.toronto.edu/~kriz/cifar.html): official 50,000 training and 10,000 test images, 100 fine classes. Seed42 classwise permutation reserves50/class validation and450/class training. Select lambda on validation, refit on full50k for test. Few-shot1/5/20 per class, seeds42/43/44, selected from the450/class training bank; select lambda on the same validation bank, and do not add validation to few-shot training.
- [VTAB CLEVR definitions](https://github.com/google-research/task_adaptation/blob/master/task_adaptation/data/clevr.py) and [TFDS adapter](https://github.com/tensorflow/datasets/blob/master/tensorflow_datasets/datasets/clevr/clevr_dataset_builder.py). Count=nobjects−3, eight classes. Distance=min float32 object pixel_coords z, class=max index whose threshold is strictly less than distance, thresholds0/8/8.5/9/9.5/10/100; six classes. Match exact boundary behavior. Official CLEVR train70k becomes63k training/7k validation; official val15k becomes evaluation test, since original test lacks scene annotations. **Deviation:** deterministic filename order replaces TFDS hashed-record order; these are VTAB task definitions and partition roles, not an exact TFDS or VTAB1k split reproduction. No reasoning/nonlinear decoder. Count and distance share extraction.

Every image is converted to RGB, PIL bilinear resized directly to96×96,
converted float32/255 and normalized by challenge mean/std. Aspect ratio is
not preserved, consistent across tasks/models. Decode-level SHA256 detects exact
challenge overlap before external probing and excludes matching images from every
partition. This does not detect near-duplicates. Source archives/annotation hashes,
row IDs and canonical image-sequence hash are retained; checkpoint and adapter
hashes bind feature caches. Feature vectors are stored FP32; no spatial maps are
saved for external global tasks. Encoder state including BatchNorm must remain
unchanged. Extract once/checkpoint/dataset; all readouts and both CLEVR labels reuse
that pass. Restart rejects incompatible completed caches.

Matrix fixed before external test access: original200/42, tuned200/42, tuned400/43,
jitter200/42, statistical200/42, ResNet18/200/42, random ResNet9/42,
random ResNet18/42. Per model: native, mean+max, mean+max+std, all-stage means,
with duplicates removed. Two ensemble pairs: original+tuned400 and
jitter+statistical,512 mean+max features per component. This is31 single conditions
plus2 ensembles per task. New SSL checkpoints use `--manifest` only after successful
training. Native/stat readouts receive CIFAR few-shot probes; avoid expanding every
model/readout/task/budget combination.

Probe: linear one-hot ridge with intercept, float64 training-only population
standardization using official epsilon1e−6 and relative lambda grid10^−4…10²
(13 values). Solve primal D×D eigensystem because official N×N dual would be
impractical at50–70k examples. Lambda scale=trace(X'X)/N, exactly the official mean
dual eigenvalue convention. For **external tasks**, lambda uses validation micro
accuracy; ties favor smaller lambda. Selected relative lambda is fixed before
train+validation refit, whose normalization and absolute scale are recomputed.
Test labels never select lambda, encoders, readouts or matrix membership. Save all
candidate validation scores, final probe weights, confusion and test predictions.
This deliberately uses explicit external validation instead of challenge analytic
LOO selection. No arbitrary block weights. If these validation results later guide
SSL/readout development, they become development data and must be called such;
the initial test endpoints remain evaluation only. Test results should not select
leaderboard candidates in this initial round.

CPU resource estimate:4–8 CPUs,16–32GiB, up to24h per dataset. GPU extraction:
1 supported GPU,4 CPUs,24GiB, up to8h per dataset. Pooled cache upper bound at8
encoders is~6GB CIFAR/~8GB CLEVR/~.6GB DTD; no feature-map expansion. Full CLEVR
archive18GB plus train/val images can require tens of GB beyond caches. Acquisition
and preprocessing have their own CPU jobs. Estimates require actual accounting.

NYU Depth V2 is **prepared protocol only**, pending primary four tasks: require
licensed/access-authorized official labeled RGB/depth and official train/test
indices; reserve image-disjoint validation from training, retain finite positive
valid depth masks, resize RGB bilinear and masks nearest neighbor, fit existing
frozen spatial per-location linear regression head, select ridge on validation
RMSE. Report masked RMSE, absolute-relative error and delta1, retaining metric depth
units and official validity conventions. No large download or depth run is launched.

Launch after acquisition:

```bash
sbatch --partition=gpus --gres=gpu:1 --constraint='titan_rtx|rtx_a6000|l40|l40s|gtx_2080_ti' --cpus-per-task=4 --mem=24G --time=08:00:00 --output=experiment_outputs/next-round/external/DTYPE-extract-%j.log experiments/lejepa/downstream/slurm.sh experiments.lejepa.next_round.external_run extract --dataset DTYPE
sbatch --partition=compute --cpus-per-task=8 --mem=32G --time=24:00:00 --dependency=afterok:EXTRACT_JOB --output=experiment_outputs/next-round/external/DTYPE-cpu-%j.log experiments/lejepa/downstream/slurm.sh experiments.lejepa.next_round.external_run cpu --dataset DTYPE
```

Use separate data output for changed plans; do not overwrite completed results.
Status: functional fixtures203192 passed. Acquisition approval is pending at initial
implementation; no external test results or successful extraction are yet claimed.

## Ready acquisition chain (requires outstanding approval)

Do not run acquisition until the user approves the selected archive size. No
acquisition or primary evaluation jobs were submitted at this preparation step.
Existing CPU fixtures203192/203194 passed. Report refresh203195 creates pending
endpoint tables. Use one invocation per approved dataset; these are independent
jobs, not an unbounded sweep. The launcher does not automatically submit children.

```bash
# After explicit download approval, replace DTYPE with dtd/cifar100/clevr.
# CLEVR needs up to12h for18GB transfer; DTD/CIFAR normally much less.
sbatch --partition=compute --cpus-per-task=2 --mem=8G --time=12:00:00 --job-name=external-DTYPE-data --output=experiment_outputs/next-round/external/DTYPE-data-%j.log experiments/lejepa/downstream/slurm.sh experiments.lejepa.next_round.external_download --dataset DTYPE --approved-large-download
# Image validation, exact overlap screening, frozen extraction share GPU job.
sbatch --partition=gpus --gres=gpu:1 --constraint='titan_rtx|rtx_a6000|l40|l40s|gtx_2080_ti' --cpus-per-task=4 --mem=24G --time=08:00:00 --dependency=afterok:DOWNLOAD_JOB --job-name=external-DTYPE-extract --output=experiment_outputs/next-round/external/DTYPE-extract-%j.log experiments/lejepa/downstream/slurm.sh experiments.lejepa.next_round.external_run extract --dataset DTYPE
sbatch --partition=compute --cpus-per-task=8 --mem=32G --time=24:00:00 --dependency=afterok:EXTRACT_JOB --job-name=external-DTYPE-probe --output=experiment_outputs/next-round/external/DTYPE-cpu-%j.log experiments/lejepa/downstream/slurm.sh experiments.lejepa.next_round.external_run cpu --dataset DTYPE
```

If approved acquisition fails because compute-node outbound access is unavailable,
retain the failure log and use an approved transfer-only fallback; verification,
extraction, exact-overlap screening and feature inference still belong in Slurm.
Do not silently use synthetic or alternate datasets. For a supplied local CLEVR
path, arrange a repository-local read-only symlink under `data/clevr/CLEVR_v1.0`
only after confirming access and provenance, or supply `--data-root` with the
expected adapter directory layout. No external source is modified.

New-checkpoint invocation requires explicit `--manifest FILE.json` of successfully
finished checkpoint specs. Give the run a distinct `--output` to avoid replacing
old dataset/model identity metadata. Depend extraction on `afterok:TRAIN_JOB` and
successful dataset preparation; depend CPU evaluation on that extraction job.
Use identical dataset manifests and split conventions; caches can be copied only
when their identity/content checks match.

## Acquisition approval and capacity block

User approved all three archive downloads. Bounded compute-network job203213
failed after10s with curl28, confirming the previously observed outbound timeout.
A transfer-only fallback script `external_transfer.sh` is ready; no archives have
been downloaded. Repository filesystem capacity was checked before transfer:
100GB total, approximately99GB used,1.9GB available. **Actual acquisition and
external primary evaluation are blocked on a larger authorized storage target.**
No old caches, datasets or experiment artifacts were removed. Transfer script
checks conservative free-space minima before network writes (DTD3GiB,
CIFAR2GiB,CLEVR60GiB) to cover archives, expansion and pooled evaluation caches.
No acquisition/extraction/probe job IDs are claimed while storage remains blocked.
NYU preparation and synthetic-only tests are complete; NYU203212 passed.

Both downloader and transfer-only fallback enforce acquisition free-space guards. External GPU extraction additionally requires10GiB free before any model feature-cache writes. These are minimum safety checks; scheduler concurrency and other jobs can consume shared filesystem space, so the larger scratch allocation must cover all concurrent pipelines.

## Authorized completed Pet feature-cache pruning

At the user's explicit request, Slurm inventory203220 verified all nine completed
Pet evaluations and identified27 disposable cache arrays. Prune203221 verified
all original SHA256s before unlinking only `cache/mid.npy`, `cache/final.npy`,
`cache/masks.npy`. It completed successfully in3m28s and reclaimed
23,637,209,472bytes (~22.01GiB); observed repository free space rose to26GiB.
Metrics, root/layer results, `probe.pt`, `history.json`, `examples.png`, all original
`cache.json` metadata, source identities, array hashes and split IDs remain intact.
Audit: `experiment_outputs/external-eval/probes/cache-prune-plan-applied.json`;
original plan: `cache-prune-plan.json`; each cache retains `pruned.json`.
No dataset, checkpoint or other cache files were removed. Active/pending consumers
were screened; root separately audited complete Slurm SubmitLine arguments.
Future cleanup tooling queries both scontrol and sacct SubmitLine.

`segmentation.run` now verifies completed source/checkpoint identity and returns
completed results before loading the encoder or generating maps. Reuse fixture
203226 passed all9 endpoints with encoder loading and cache regeneration forbidden.
This cleanup does not by itself authorize any external storage target or remove
the remaining full-CLEVR capacity constraint.

## DTD and CIFAR launch after authorized cleanup

Historical checkpoint optimizer archival203227 completed. Actual repository free
space reached30GiB; the authorized Pet cache pruning preserved all metrics and
weights. Training reservation is13GiB including the8GiB free-space floor and~5GiB
incremental outputs. Current+new DTD/CIFAR datasets/caches/probes budget~11GiB.
The extraction guard requires `max(10GiB, estimated pending model arrays+8GiB)`;
no uncached extraction proceeds below this budget. Full CLEVR remains blocked by
its60GiB acquisition guard and has no acquisition/extraction jobs.

Transfer-only curl ran on login because bounded compute networking203213 timed
out. DTD transfer completed successfully; prepare203231 was released afterward
for official MD5 verification and archive extraction in Slurm. CIFAR transfer is
in progress; prepare203240 remains held until successful archive completion.
No hashing, decompression, image processing or inference ran on login.

| Dataset | Verification/preparation | Existing extraction | Existing CPU probes | New ResNet18 extraction/probe | New original ResNet9 extraction/probe | New six-view extraction/probe |
|---|---:|---:|---:|---|---|---|
|DTD|203231|203232|203233|203234/203235|203236/203237|203238/203239|
|CIFAR100|203240|203241|203242|203243/203244|203245/203246|203247/203248|

New-model extraction also requires success of its exact training203199/203200/
203201 and checks saved config plus final epoch before evaluating. CPU probes
require successful extraction. Preparation jobs start only after their archive
transfer succeeds; no failed dependencies are released blindly. Exact commands,
IDs, dependencies and transfer sizes are saved in `external/dtd-jobs.json` and
`external/cifar100-jobs.json`. Logs use the same directory. Future encoder outputs
are isolated under `external-new/RUN-seed42/DATASET` and aggregate automatically.
All external test partitions remain unevaluated at this launch point.

Latest launch updates:

- Both transfer-only curl commands succeeded: DTD625,239,812bytes (~120s), CIFAR169,001,437bytes (~500s). Prepare jobs203231/203240 were released only after their respective completed archives existed. DTD203231 passed official verification/extraction in1m34s; CIFAR203240 is proceeding.
- Existing DTD extraction203232 was cancelled during startup before any features or evaluation results existed, to extend exact-overlap screening to both original decoded pixels and bilinear96px copies. Replacement203258 is running; CPU203233 now depends on203258. Original startup log is retained.
- Scheduler recovery replaced held training allocations203199/203200/203201 with203253/203254/203255. All six future external GPU jobs were rewired to the corresponding replacement training and unchanged dataset preparation success dependencies; their CPU children remain unchanged. Full dependency history is retained in both jobs JSON files.
- The shared overlap audit includes original+96px hashes and adapter/method identity; future checkpoint extractions reuse it. No external test result has been used to choose readouts or the prespecified ensemble pairs.

Monitoring:

```bash
squeue -j 203231,203258,203233,203240,203241,203242,203234,203235,203236,203237,203238,203239,203243,203244,203245,203246,203247,203248
cat experiment_outputs/next-round/external/dtd/*-status.json
cat experiment_outputs/next-round/external/cifar100/*-status.json
tail -f experiment_outputs/next-round/external/dtd-existing-extract-v2-203258.log
tail -f experiment_outputs/next-round/external/cifar100-existing-extract-203241.log
```

## CLEVR launched on the provisioned scratch filesystem

User-provisioned `/nbu/people/ppopescu/stable-challenge-image-next-round` now hosts
all external datasets, caches and results. Verified migration203401 completed:
6194 files/7,585,151,437bytes checked by SHA256/size/metadata; original repository
path strings remain symlinks to the scratch copies. Future `external-new` outputs
also resolve to scratch. Active SSL files stay in their original home paths.
Migration/storage audits are in `reporting/storage-migration-203401` and
`reporting/external-storage.json`.

CLEVR endpoint-specific network check203392 failed on compute after10s (curl28).
The approved transfer-only login fallback is now running into scratch, with more
than3.8GiB already received at the launch check. No archive processing runs on
login. Full official archive expected size19,021,600,724bytes and SHA256
`5cd61cf1096ed20944df93c9adb31e74d189b8459a94f54ba00090e5c59936d1` come from
[official TFDS checksums](https://raw.githubusercontent.com/tensorflow/datasets/master/tensorflow_datasets/datasets/clevr/checksums.tsv).
Preparation validates that exact size/hash before extracting train/val images and
scene annotations; ZIP CRCs are checked while reading members. Original unlabeled
test images and questions remain unexpanded. Partial extraction members with
incorrect byte lengths are replaced on an authorized restart.

| CLEVR stage | Job | Dependencies |
|---|---:|---|
|Official archive verification/preparation|203432|Held until successful complete transfer|
|Existing encoders, shared count/distance extraction|203433|afterok203432|
|Existing count/distance CPU probes|203434|afterok203433|
|New ResNet18 extraction/probes|203435/203436|training203253 + prepare203432; then extraction success|
|New original ResNet9 extraction/probes|203437/203438|training203254 + prepare203432; then extraction success|
|New six-view ResNet9 extraction/probes|203439/203440|training203255 + prepare203432; then extraction success|

`external_transfer_release.sh clevr 203432` performs only byte transfer plus a
small scheduler action: curl success → atomic final archive name → verify the
preparation ID against `clevr-jobs.json` → release203432 → record bytes/time.
A failed/truncated transfer retains `.partial` and leaves preparation held. The
wrapper verifies the repository external path resolves to the authorized scratch
root before downloading. Exact queued commands/dependencies are retained in
`external/clevr-jobs.json`; progress is available from
`external/data/clevr/CLEVR_v1.0.zip.partial` and `external/clevr-transfer.log`.
The transfer does not need a later assistant turn to release preparation.

Existing DTD and CIFAR evaluations completed successfully before migration:
DTD203258/203233 (18m13s extraction/2m55s CPU) and CIFAR203241/203242
(12m46s extraction/6m19s CPU). Their fixed33-condition matrix includes99 DTD
split endpoints and33 CIFAR full-training endpoints plus documented few-shot
repeats. Test metrics describe the prespecified matrix; no test result changed
encoder/readout membership or selected leaderboard artifacts. CLEVR results and
all three new SSL encoder endpoints remain pending at this launch record.
