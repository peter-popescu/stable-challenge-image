# Prepared NYU Depth V2 linear probe

`nyu_probe.fit` is an evaluation-only primitive. It does not acquire data, extract
features, submit jobs, train encoders, or run a depth experiment automatically.
Dataset access, the concrete dataset adapter, depth alignment/preprocessing and
original official split file are still pending. Complete the four primary external
tasks first. Never substitute a random split or infer official test membership.

Expected arrays:

- Frozen spatial features `[N,H,W,C]` in NHWC order.
- Aligned metric depth targets `[N,H,W]` in **metres**.
- Explicit aligned boolean validity masks `[N,H,W]`.
- Unique image IDs and manifest containing `official_train_ids`,
  `official_test_ids`, `train_ids`, `validation_ids`, `test_ids`,
  `official_split_source`.

Train and validation must exactly partition the documented official training
images; test must equal the documented official test images. All images must
appear exactly once in the array IDs. Partition overlap is rejected before any
fit. The data adapter must record original official split file and data hashes,
RGB/depth registration, crop/resize rules, depth units/range and original validity
conventions. RGB resizes bilinearly; validity masks use nearest neighbor. Depth
resampling must not blend missing/invalid depths into valid targets. The prepared
probe does not assert that arbitrary aligned arrays reproduce an official NYU
benchmark; this requires the pending adapter and documented preprocessing.

Every location from a training image stays in the training partition. Fit training
feature standardization and a scalar linear ridge head with intercept in float64.
Candidate alphas `.01,1,100` reuse the existing appearance-probe pattern. Validation
masked RMSE selects alpha; keep that training-only fit for test. Test depth cannot
select weights, scaling, alpha or preprocessing. Validity is the conjunction of the
explicit mask and finite positive metric target. Partitions with no valid sites
are rejected. Inputs/encoder weights remain frozen; no nonlinear decoder.

Report pixel-weighted valid-site RMSE in metres, mean absolute-relative error,
and delta1 (`max(pred/target,target/pred)<1.25`) with per-image metrics and counts.
Do not clamp linear predictions: nonpositive predictions fail delta1 and remain
visible in regression errors. Report their fraction. Save standardized linear
weights, intercept, validation candidate metrics, final test predictions/targets,
split manifest and per-image results separately from SSL/submission artifacts.

Synthetic-only Slurm fixture203212 checks ignored invalid targets, nonpositive
prediction treatment, explicit split leakage rejection, known linear regression,
and invariance of selected alpha/validation scores under changed test depths.
It does not access NYU or establish performance on real images.

Primary dataset documentation:
[NYU Depth V2 official page](https://cs.nyu.edu/~fergus/datasets/nyu_depth_v2.html).
Use its labeled depth conventions and original split artifact; the much larger raw
video collection and third-party preprocessed depth datasets are outside this
prepared lightweight evaluation. No data was acquired for the synthetic fixture.

The official labeled MAT file distinguishes `depths` (inpainted) from `rawDepths`
(projected values before filling), both in metres. The future adapter must name
which target it uses and derive/retain the corresponding sensor-valid mask;
inpainted pixels must not silently become sensor-valid ground truth. This choice
and any crop/range restriction must be fixed before validation or test scoring.
