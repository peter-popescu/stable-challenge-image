# Frozen representation evaluation protocol — 2026-11-17

All compute, neural smoke tests, extraction, preprocessing, and probes run in Slurm.
Only the authorized dataset archive transfers used the login host, after compute-node
outbound connections timed out. No sudo, system configuration changes, external SSL
inputs, additional SSL variants, or leaderboard uploads are involved.

This suite characterizes representation properties, not predictors of hidden
leaderboard performance. Current encoders have one training seed each. The tuned
400-epoch model uses seed43; duration, seed and schedule are confounded. Architectures
also differ in width/depth, capacity, initialization and native readout.

## Models and interfaces

`common.models()` lists original ResNet9/200/42, tuned ResNet9/200/42,
tuned ResNet9/400/43, local-min10 ResNet9/200/42, jitter-strong ResNet9/200/42,
tuned width40 ResNet18/200/42, and random ResNet9/ResNet18 (seed42).
The statistical-pooling ResNet9/200/42 is evaluated by dependent jobs after its
200-epoch training allocation succeeds. It is never evaluated from an early `last.ckpt`.

Global readouts: ResNet9 max2/1024, GAP/256, GMP/256, mean+max/512,
mean+max+population-std/768; ResNet18 GAP/320, GMP/320, mean+max/640,
mean+max+population-std/960. The unnecessary avg2 Cartesian expansion is omitted.
There is one spatial evaluation per distinct encoder/layer, shared by all readouts.

`--manifest FILE.json` accepts a list of model specifications. Matching teammate
VISReg ResNet18 checkpoints can use this explicit adapter (replace state key/prefix
with the actual checkpoint schema; strict loading rejects missing/extra keys):

```json
[{
  "name": "visreg-resnet18-seed42",
  "backbone": "resnet18",
  "checkpoint": "path/in/repository/checkpoint.pt",
  "loader": "experiments.lejepa.downstream.common:load_matching_checkpoint",
  "state_key": "state_dict",
  "prefix": "encoder.",
  "readout": "gap",
  "stage_widths": [40,80,160,320],
  "stage_depths": [2,2,2,2]
}]
```

A different architecture may supply `loader`, `spatial_adapter`, and explicit
`layer_info` (stride, first receptive-field center, maximum RF size). The spatial
adapter returns `mid` and `final` tensors. This is an evaluation adapter, not a new
SSL pipeline. Checkpoint hashes, source hashes, row IDs, preprocessing and cache
content hashes bind runs. The encoder is always `eval().requires_grad_(False)`;
state comparisons include BatchNorm buffers. Probe optimizers receive probe
parameters only.

## Statistical-pooling training and frozen ResNet18 pooling

Training reuses `architecture_preflight`, `architecture_run`, `main`, milestone
`checkpoint_eval`, diagnostics, resource callback and export utilities. The prepared
`architecture/resnet9-statpool.json` is the tuned original-augmentation recipe:
seed42, planned200-epoch cosine after10 warmup epochs, batch128, LR.003,
SIGReg weight.005,1024 slices, AdamW weight decay.0001, FP32, strict determinism,
workers0; two96px global and two48px local views. No continuation from old weights.
The SSL embedding is768; projector first layer is768→512 and remaining layers are
unchanged. No padding enters SSL. Milestones50/100/150/200 retain optimizer,
scheduler and SIGReg state, classification, PCA, training losses and resources.
The original official1024-wide interface alone adds zeros on export/probing.

Population std is still native `torch.std(unbiased=False)` over all spatial sites,
using its finite zero subgradient at constant channels. We did not replace this
with `sqrt(var)`, introduce epsilon bias or change frozen statistical values.
CPU/GPU constant-channel backward and finite full-batch gradients passed preflight;
a double precision gradcheck is also included. Frozen ONNX exports have no backward
path and are checked numerically including dynamic batch sizes.

The primary matched comparison separates max2 train/max2 eval, max2 train/statistical
eval, statistical train/statistical eval. Existing control/frozen metrics remain
separate from the new training endpoint. Local readout selection is not hidden-set
validation. ResNet18 is never retrained: `pooling_study --backbone resnet18` reuses one
float32 final-map cache for GAP/GMP/mean+max/mean+max+std. Original GAP embeddings must
match historical ONNX features and outer/LOO predictions exactly before alternatives
are interpreted. The unchanged official fixed five-fold, macro and analytic LOO
probe handles all readouts. Export candidates improve five-fold or LOO locally;
exports are <20MiB, dynamically batched, parity checked, and leave encoder weights
unchanged. Nothing is uploaded.

## Global semantic organization

Use all750 labeled development images at96×96, RGB/255 and challenge mean/std.
No stochastic evaluation augmentations. The same ordered row IDs and fixed
stratified5-fold manifest as previous studies are used. Existing verified
classifications are reused by checkpoint/data/fold identity; missing readouts use
`cross_validate.probe_embeddings`, including training-only outer standardization
and inner official analytic LOO regularization selection. The official inner LOO
uses its existing training-partition-wide standardization convention.

Other global tasks use per-image L2 normalization, cosine similarity/distance,
and no fitted dataset-wide normalization. kNN reports fixed k=1,5,10, uniform
voting, excluding the query itself from the labeled reference bank. Lowest class ID
breaks vote ties; stable row order breaks equal similarities. These are labeled
leave-one-out neighbor metrics, not trained probe replicates.

Few-shot classification is cosine nearest support-class centroid, a fixed linear
classifier with no probe hyperparameters. Twenty repeats use seeds42…61. Each repeat
independently permutes each class in sorted class order; support sets are nested
across1/5/10/15 shots. Classes with at most `shots` images are excluded at that shot
count so at least one query remains. Query uses every remaining eligible image.
All models/readouts share exactly the same saved `episodes.json`. Report micro and
macro mean and episode sample SD, class eligibility and query counts; different
shot counts can have different eligible class sets. Episodes are not independent
encoder training replicates or confidence intervals.

Clustering runs Euclidean KMeans on unit image features (centroids are not renormalized), K=30 fixed from the challenge documentation (class IDs0–29),
10 initializations per run and seeds42…46. Labels only evaluate ARI/NMI; they never determine K or fit cluster assignments. NMI uses the arithmetic averaging convention. Both raw initialization results and their
mean are retained. ARI can be negative; NMI is higher-is-better in[0,1].

Class retrieval ranks all other images by cosine similarity; AP averages precision
at each relevant rank divided by the number of other same-class images. Report
image-mean mAP and class-macro mAP, retaining per-query AP. Singletons, if any, get
AP0. No self-match or label-based feature fitting occurs.

## Spatial layers and dense synthetic correspondence

| Architecture/layer |96px tensor C×H×W|Stride|First center (pixel-edge coordinates)|Maximum RF width|
|---|---|---:|---:|---:|
|ResNet9 mid (conv index11, after second2x2 pool)|256×24×24|4|2|20|
|ResNet9 final (before adaptive pooling)|256×12×12|8|4|64|
|ResNet18 mid (stage index2)|160×24×24|4|.5|53|
|ResNet18 final (stage index3)|320×12×12|8|.5|109|

The ResNet9 mid layer is earlier in depth than the ResNet18 mid layer. Matching
stride does not match receptive field or depth. These differences remain explicit.

Correspondence alone resizes the same32 seed42-selected labeled images to192×192.
This deliberate deviation from96px is necessary for a full109px ResNet18 final RF
to fit without padding. It diagnoses equivariance at that resolution, not96px
semantic transfer. Native grids are48×48 and24×24; feature maps are not interpolated
to a common grid. Pixel-edge coordinates start with the first image center at.5.
Analytic source→target homogeneous matrices specify:

- Identity control.
- Translation (+16,+8) pixels.
- Rotation15 degrees about(96,96), positive in the image-coordinate plane.
- Horizontal flip `x' =192-x`.
- Center crop153.6px resized to192: `x'=1.25x-24`, same for y.
- Photometric transform `clip(.65 RGB+.15,0,1)` alone and with translation.

Views use inverse mapping, bilinear `grid_sample`, `align_corners=False`.
Both the original RF and the transformed/inverse-mapped RF corners must fit
entirely inside both visible images. Destination descriptors are restricted to
RF-valid overlap too. No padded descriptor is scored. When no overlap exists,
record it explicitly. Identity can have nonzero error for tied or numerically indistinguishable descriptors (for example uniform backgrounds); no coordinate-based tie breaking is used to inflate scores. RF support is conservative maximum theoretical support,
including residual paths; it is not the empirical effective RF.

Match per-site L2-normalized vectors by maximum cosine similarity. Error is
Euclidean pixel distance in the transformed input-image coordinate system;
PCK thresholds are4/8/16 pixels (2.08/4.17/8.33% of192px width). Report per-image
means and image-average summaries by layer/transform. Save the nearest attainable
target-grid error as a quantization oracle, counts of valid sites, and native stride.
The primary candidate supports differ across architectures because their RFs
differ; compare random controls within architecture and inspect the oracle/counts
before attributing differences to learned quality. This is a synthetic equivariance
test, not semantic correspondence or segmentation.

Raw9×9 RGB patch descriptors are sampled at the identical CNN centers and grid,
using the same conservative valid sites. Both random CNN architectures are evaluated
with the same protocol. Visualizations use the first shared image: yellow lines
are predicted matches; green dots are known target coordinates. Quantitative
metrics take priority over PCA or visual selection.

## Appearance regression

The same750 images have fixed seed42 shuffled image-level60/20/20% train/validation/
test partitions (450/150/150), saved as `appearance_splits.json`. No spatial locations
from a test image can enter probe fitting. Targets are six unnormalized[0,1] RGB
channel means and population standard deviations. Global features predict full-image
stats. Local features predict8×8 RGB-window stats at their receptive-field centers;
window edges snap by Python `round(center)` and are clipped at image boundaries.
For half-integer ResNet18 centers this entails at most half-pixel center offset;
clipped boundary windows have smaller support.

Feature standardization and ridge solves use float64 (caches/targets retain their documented storage precision). This avoids float32 conditioning warnings at small ridge penalties. StandardScaler fits probe TRAIN features only. Ridge alpha=.01/1/100 is selected by
validation MSE; the chosen train-only fit is evaluated once on test images. Local
sites are pooled only after image partitioning; all sites from train images fit the
same per-layer ridge. No refit on validation or test is used. Report average and
per-target MSE and R². Targets retain original units; R² averages channel scores and
can be negative. The nonlearned constant baseline uses the TRAIN target mean.
For local tasks, an explicitly labeled image-global oracle copies each test image's
observed global RGB stats to all its local sites; it uses raw image access, not
representation decoding, and is a separate reference.

Compare jitter-strong and tuned200/42 with identical splits/readouts/layers alongside
prior jitter invariance (.9970 vs.9897 mean cosine). Lower accessible RGB information
and greater transform invariance are related hypotheses, not equivalent definitions.

## Oxford-IIIT Pet frozen dense linear probes

Source: https://www.robots.ox.ac.uk/~vgg/data/pets/ (official image/annotation archives;
torchvision MD5 checksums and official split-file SHA256 recorded in `source.json`).
Official trainval has3680 images; test has3669. Fixed validation is20% within each
breed using seed42, leaving2944 train/736 validation before any overlap exclusion.
Test never selects a probe LR, epoch, layer, readout or SSL configuration.

Images resize directly to96×96 with PIL bilinear, then RGB/255 and challenge mean/std.
Masks resize96×96 with nearest neighbor. Original trimap1=pet,2=background,3=boundary
becomes0/1/2. Boundary is a scored third class, not ignored or merged; report
3-class mean IoU, foreground(pet) IoU, all3 class IoUs, confusion matrix and pixel
accuracy. Class IoU is dataset-aggregate intersection/union, not mean per-image IoU.

For each encoder and selected layer, train exactly one1×1 convolution producing
3 logits, then bilinear upsample to96 (`align_corners=False`) and pixelwise unweighted
cross entropy. No nonlinear decoder, encoder gradient, BN update or learned encoder
normalization. AdamW weight decay1e-4; LR candidates.003/.01,20 epochs each,
batch64, seed42 initialization and seed42+epoch image shuffle. Select the single
LR/epoch checkpoint with highest validation mIoU; ties keep the earlier candidate.
Evaluate that checkpoint once on the official test set. Both final and stride4
intermediate layers and random architectures are included.

Frozen encoder inference is FP32. To bound storage, Pet map caches use FP16, then
convert to FP32 before the linear probe; challenge caches/readouts remain FP32.
This is an explicit evaluation precision approximation shared across all models.
There is no mixed precision SSL change. Probe weights contain only the1×1 layer,
are saved under `external-eval/probes`, and never enter competition exports.

Check exact decoded RGB and96px resized RGB hashes against all SSL challenge inputs;
exclude any exact matches from all external partitions and record their IDs.
This does not establish absence of all crops/near-duplicates or identify an
undocumented original challenge source. No known overlaps are intentionally used.
Fixed first6 retained official test images provide image/ground-truth/predicted-mask
panels; no cherry picking. A dataset-size ablation is deferred to keep this round
bounded; adding another dataset requires supplying maps/masks/split IDs to the same
`train_probe` and `validate` functions, not implementing a new probe pipeline.

## Separation, caching, reporting, cost and launch records

SSL input is solely `data/train.parquet`; statistical training never imports the
downstream dataset loader. External archives, annotations, maps and probe weights
remain in `experiment_outputs/external-eval/`. Submission-capable frozen exports
live only under the pooling/training outputs and contain no downstream probe weights.

Existing final float32 maps are reused after checkpoint/data/map hash checks;
only missing stride4 maps are extracted for those models. Remaining challenge
models get one two-layer extraction pass. Readouts reuse the same final maps.
All jobs keep raw JSON/NPZ metrics and caches; none remove experiment outputs.
Approximate storage: challenge maps4GiB for8 encoders; Pet maps~19GiB plus2GiB
archives/images/annotations; check actual sizes after extraction. Each model keeps
its cache independently to support additional probes without encoder re-extraction.

Initial allocations/estimates (wall estimates, not measured completion claims):
statistical SSL1GPU/4CPU/24GiB, about5–6h from first93s epochs plus milestones
(24h limit); challenge extraction1GPU/4CPU/24GiB, roughly15–60min (4h limit);
embedding/regression CPU4CPU/24GiB, roughly20–90min (4h limit); Pet1GPU/4CPU/24GiB,
roughly2–6h for8 encoders (12h limit). New-encoder follow-ups have1h extraction,
1h CPU and3h Pet limits. No job arrays or child submissions. CPU tasks run after
extraction ends, including on partial extraction failure so unrelated cached models
can still complete. Per-model exceptions are recorded and isolated. Appearance-only correction job199292 reuses all embeddings, maps and semantic metrics, archives initial float32 regression JSON, and repeats only the same fixed regression protocol in float64; it does not change splits or probe hyperparameters.

`REPORT.md`, `performance_matrix.csv/json`, `dense_matrix.json`, per-model raw
metrics and `task_profile.png` update as jobs complete. The consolidated matrix
repeats encoder spatial scores across its readouts only for convenience; those are
not separate experiments. The plot is a native-scale task profile rather than a
radar area: higher-is-better bounded scores keep[0,1], ARI/R² keep possible negatives,
and lower-is-better MSE/pixel errors stay in their raw tables. Pending task axes are NaN gaps, never zero scores. No overall scalar
quality or radar-area score is defined. Random CNNs, raw RGB patches and constant
regression references remain explicit baselines; no best-observed min/max scaling.
The pooled correspondence matrix averages PCK8 equally over translation, rotation,
flip and crop-resize; raw per-transform scores remain available.

See [LAUNCH.md](LAUNCH.md) for job IDs, preflight measurements, deviations and exact
monitoring commands. An interrupted run refuses incompatible plan/source/cache
identity. Preserve the attempt plan and use a fresh output for changed numerical
protocols; do not overwrite completed evaluation endpoints.

## Initial interpretation, before the new SSL endpoint

The completed existing-encoder tasks show tradeoffs rather than a universal winner.
At native readouts, stronger jitter improves five-shot classification and retrieval
relative to tuned200, while reducing global and local RGB R². This is consistent
with reduced linear color accessibility alongside stronger photometric invariance;
it does not prove that color information is absent from nonlinear probes.
The seed43/400 model improves several semantic metrics and synthetic PCK8 but loses
local color accessibility; seed and schedule remain confounded. ResNet18 GAP and
ResNet9 max2 have similar five-fold accuracy yet different few-shot/retrieval and
layer-dependent correspondence profiles. RF/support differences prevent attributing
all spatial differences to learned quality.

For the tuned ResNet9 and ResNet18, frozen statistical readouts improve classification,
retrieval and global RGB regression while leaving dense maps exactly unchanged.
A plausible explanation is better linear access to feature-distribution moments.
RGB mean/std targets have a similar functional form to the statistical readout;
this alignment may favor it on this particular appearance task and is not evidence
that it preserves all appearance information. The trained statistical encoder must
be compared to the frozen statistical readout to investigate learning itself.
All planned endpoints have completed. The statistical-training intervention improves final spatial/color/segmentation scores but lowers several global semantic metrics relative to frozen statistical extraction; it is not broadly superior. See the raw matrix and paired changes for exact effects.

## Global-only cache policy for future evaluations

Use `downstream.run extract-global` followed by `downstream.run cpu-global` when
only classification, few-shot, retrieval, k-NN, clustering and global RGB
accessibility are needed. These stages write `MODEL/global-cache/embeddings.npz`
and its hash/identity manifest, never `mid.npy` or `final.npy`. Final spatial
activations exist for only one inference batch and are pooled immediately.
Global CPU evaluation verifies cache content, checkpoint/data/row identities and
source provenance before interpreting vectors. It never loads spatial tensors or
runs correspondence/local RGB. Results are `semantic-global.json`,
`appearance-global.json`, `global_performance_matrix.{json,csv}` and
`GLOBAL_REPORT.md`, preserving previous dense endpoints and reports.

The older `extract`/`cpu` stages explicitly opt into persistent spatial maps and
dense probes; request them only when those maps are needed. Existing historical
cache files/manifests are not migrated or relabeled. Source changes intentionally
require a fresh output directory when rerunning a completed cache.

The next-round frozen study and external evaluation suite already cache pooled
embeddings only; future SSL checkpoint follow-up jobs use that same policy.
