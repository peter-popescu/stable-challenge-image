# Fixed lightweight CLEVR branch

This branch replaces the cancelled full-dataset CLEVR evaluation as the active
experiment. It uses exactly **1000 training,500 validation,2000 test images** per
encoder, shared by count and closest-object-distance tasks. No additional data is
downloaded. Full-run images, annotations, archive and all complete/partial caches
remain preserved separately; no partial feature cache is trusted.

Selection is fixed before probe results: seed42 NumPy PCG64, choose without
replacement from the existing63000/7000/15000 role partitions in train,validation,
test order, sort within each role, concatenate these three sets. No label enters
sampling. The full overlap audit and dataset identity are verified and reused;
there is no full-image overlap rescan. The metadata/scene adapter is checked, and
only selected images are decoded for encoders lacking a completed cache.
Both labels use the unchanged documented VTAB task definitions, but **this is not
an official VTAB1k split or the full VTAB protocol**. Split rows are explicitly
marked `clevr-light` in results. Existing filename-order partition conventions
remain documented in EXTERNAL.md.

The same prespecified eight existing encoders,31 individual readouts and two
ensembles cover both tasks,66 endpoint rows. There is no new model/readout
selection based on test results. Lambda selection uses1000 training/500 validation
images. The selected linear ridge probe is then **refitted using1500 train+val
labels** before scoring the2000 held-out test images. Report validation accuracy,
test accuracy/macro accuracy, class supports and final probe parameters separately.
No nonlinear decoder, encoder training or learned reduction is added.

Output isolation:

- Existing models: `experiment_outputs/next-round/external/clevr-light/clevr`.
- New endpoints: `experiment_outputs/next-round/external-new/RUN-seed42-clevr-light/clevr`.
- Full historical/partial run: `experiment_outputs/next-round/external/clevr`, retained unchanged.
- Jobs: `experiment_outputs/next-round/external/clevr-light-jobs.json`.

A CPU preparation job publishes the immutable3500-row manifest and checks every
array hash and checkpoint/code/dataset identity before slicing completed full
caches. Slicing uses **exact row-ID lookup**, preserving selected FP32 vectors.
Only full encoders with a completed `cache.json` are eligible; partially written
arrays without it are ignored. Currently the original encoder has a complete
full cache. New isolated feature-cache metadata records the original array hashes,
source metadata hash and exact reuse method. Others extract3500 images each on
GPU, once across all readouts and both labels. Baseline `external_data.py`,
`external_run.py`, their hashes and all DTD/CIFAR caches remain unchanged.

The light adapter's complete file hash binds every new dataset plan. Freeze it
before preparation; an incompatible later source change fails closed. The new
checkpoints are checked for exact saved configuration/final epoch, and evaluated
only after their training jobs203253/203254/203255 and light preparation succeed.

Expected reduction:85000→3500 images per uncached encoder,24.3× fewer image
inferences. Seven existing uncached encoders need24500 images total; the original
encoder reuses completed full vectors. Per-stage limits: CPU preparation2CPUs/8GiB,
30min; GPU extraction1compatibleGPU/4CPUs/24GiB,2h; cached probes8CPUs/16GiB,2h.
These are scheduler ceilings, not measured consumption. New-model jobs use the
same small3500-image protocol. No exhaustive variants or full CLEVR job is resumed.

Tests203996 and204000 use synthetic data only. They check deterministic
label-independent selection, disjoint role membership, true Dataset.exclude
source-index/image/label composition, shuffled full-cache row-ID mapping, corrupt
hash rejection and ignoring partial caches. CPU preparation is the end-to-end
real-data CLI check; inference cannot start unless it succeeds.

## Submitted and verified

Independent review and syntheticfixtures203996/204000 passed. The final adapter
was frozen before dataset-plan publication, SHA256
`c5bdf4787f32ef113a6698789445ee30ab1af90187d35a66c306b321e0110c4d`.
Real-data CPU preparation204017 completed successfully in46s (32.7s measured
stage time), publishing the exact3500-row plan and reusing the original encoder's
complete full vectors after all hashes/identities passed. Its isolated41MiB cache
records the original full-array hashes. No partial encoder cache was reused.

| Stage | Job | Success dependencies |
|---|---:|---|
|Fixed-manifest/complete-cache CPU preparation|204017|fixture204000; completed|
|Existing encoders:3500-image GPU extraction|204018|204017|
|Existing count/distance CPU probes|204019|204018|
|New ResNet18 extraction/probes|204020/204021|training203253+204017, then extraction|
|New original ResNet9 extraction/probes|204022/204023|training203254+204017, then extraction|
|New six-view ResNet9 extraction/probes|204024/204025|training203255+204017, then extraction|

Existing GPU extraction204018 is running. DTD/CIFAR queues and outputs were not
modified. Full CLEVR203433 was cancelled separately and its data/caches remain
preserved. See `external/clevr-light-jobs.json` for exact commands and limits.
No task score or final completion is claimed before successful CPU probes.
