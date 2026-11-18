# Completed ResNet18 and frozen pooling experiments

Both jobs completed with exit code0. No optional retraining or public uploads were performed.

## Architecture comparison at the matched200-epoch endpoint

| Model | Five-fold | Macro | Official local LOO |
|---|---:|---:|---:|
| ResNet9 tuned seed42, max2 | 73.07% | 69.64% | 72.40% |
| ResNet18 width40 seed42, GAP320 | 72.93% | 69.61% | 73.47% |

ResNet18 changes five-fold accuracy by-0.13pp (one fewer correct image), macro by-0.03pp, and LOO by+1.07pp (eight more correct). It is effectively tied on the outer metric; LOO alone does not establish a better representation or hidden transfer.
The comparison includes depth/width, initialization, readout and native embedding dimension. It is not a pure parameter-count intervention.

| ResNet18 epoch | Five-fold | Macro | LOO |
|---|---:|---:|---:|
| 50 | 63.07% | 59.60% | 64.13% |
| 100 | 70.80% | 67.66% | 69.87% |
| 150 | 72.80% | 69.80% | 73.33% |
| 200 | 72.93% | 69.61% | 73.47% |

## Frozen pooling results grouped by checkpoint

Each cell is **five-fold / local LOO**. Every original max2 readout reproduced saved outer and LOO predictions exactly.

| Checkpoint | Existing max2 | Global mean + max | Global mean + max + std |
|---|---:|---:|---:|
| original-200-seed42 | 70.00% / 68.40% | 70.93% / 71.07% | 71.07% / 70.67% |
| tuned-200-seed42 | 73.07% / 72.40% | 72.93% / 74.00% | 74.13% / 74.67% |
| tuned-400-seed43 | 76.00% / 73.47% | 75.87% / 75.73% | 74.67% / 75.60% |
| jitter-200-seed42 | 72.80% / 72.13% | 74.93% / 74.67% | 75.20% / 74.93% |

Global mean+max and mean+max+std improve LOO on all four frozen checkpoints. Their five-fold effects are checkpoint-dependent: std helps the original200, tuned200 and jitter200 checkpoints but hurts the tuned400 checkpoint. Pure2x2 averaging hurts five-fold accuracy in all four cases; GAP alone is mixed. Global aggregation is not uniformly superior.
The jitter mean+max+std result is75.20% five-fold,71.64% macro and74.93% LOO: gains of+2.40,+2.83 and+2.80pp over the same frozen jitter encoder with max2. It improves15 observed classes, leaves9 unchanged and worsens5; four of five fold changes are positive.
For the tuned400 checkpoint, mean+max preserves most outer accuracy (75.87% versus76.00%) while increasing LOO from73.47% to75.73%. This is the highest LOO in this pooling study, not a measured hidden-set improvement.
Population std is computed per channel over spatial sites; no labeled readout fitting or encoder update occurred. All lower-dimensional embeddings were padded to1024 before the unchanged official probe. Small extra encoder inference was used only for ONNX parity, not a full re-extraction for each readout.
Spatial PCA for a frozen checkpoint is shared across its readouts because its convolutional maps are unchanged. Per-class/fold changes, predictions, nearest neighbors, variance/effective rank and valid exports are in the detailed report.

## Resource measurements

- Backbone parameters:4,365,480. Final trained ONNX:17,464,706 bytes (16.66MiB); the final size differs slightly from the untrained preflight export.
- Training throughput:67.70 images/s across200 epochs; measured epoch wall time:6.09h.
- Peak CUDA allocation:8.32GiB; reservation:8.91GiB on TITAN RTX.
- Primary allocation185780:6h46m46s (6.78 GPU-hours), including startup/evaluation. Preflight185752:5m27s; combined6.87 allocated GPU-hours.
- Frozen study185781:22m47s on4 CPUs (1.52 allocated CPU-hours), no GPU requested.

## Artifacts and next steps

- [Architecture report and curves](../../../experiment_outputs/backbone-pooling/resnet18-tuned-seed42/REPORT.md)
- [Frozen pooling report, PCA and complete readout table](../../../experiment_outputs/backbone-pooling/frozen-readouts/REPORT.md)
- [All pooling metrics](../../../experiment_outputs/backbone-pooling/frozen-readouts/comparison.csv)
- [Prepared optional from-scratch mean+max+std configuration](../../../experiment_outputs/backbone-pooling/frozen-readouts/optional-retrain-avgmaxstd.json)
- [Final resource accounting](../../../experiment_outputs/backbone-pooling/final_accounting.json)

The optional configuration changes only the ResNet9 readout (and enables resource logging) from the tuned original-augmentation recipe; it was prepared because mean matched five-fold improvement across the four checkpoints exceeded the documented0.5pp screening rule. That mean is descriptive: these checkpoints are not four independent training seeds. No additional training was launched.
These are repeatedly reused development scores. The earlier leaderboard history shows poor local/hidden alignment. Treat the frozen readouts as promising transfer-test candidates, not established leaderboard winners.

## Next-round frozen ResNet18 pooling

The completed encoder's original GAP reproduced historical embeddings and both
outer/LOO prediction arrays exactly. Using the same cached spatial maps:

| Readout | Native dimensions | Five-fold | Macro | Official LOO |
|---|---:|---:|---:|---:|
| GAP |320|72.93%|69.61%|73.47%|
| GMP |320|73.47%|70.23%|72.93%|
| Mean+max |640|75.07%|71.78%|74.53%|
| Mean+max+population std |960|75.33%|72.29%|74.80%|

These are frozen-readout changes, not ResNet18 retraining. All4 candidates were
exported with dynamic-batch and cached-feature numerical parity; official-validator
logs are retained with the exports. See the [next-round launch record](../downstream/LAUNCH.md)
and [incremental consolidated report](../../../experiment_outputs/green-area/downstream/REPORT.md).
The200-epoch statistical ResNet9 training and broader downstream comparisons are
running; no improvement in hidden leaderboard performance is claimed.

## Completed statistical-pooling SSL and downstream evaluation

All9 encoders completed all task families; the final matrix contains43 global
readout conditions and18 encoder/layer conditions. The matched200-epoch results:

| Pooling during training / evaluation | Five-fold | Macro | LOO |5-shot| Retrieval mAP | Final PCK8 | Final local RGB R² | Final Pet mIoU |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
|max2 / max2|73.07%|69.64%|72.40%|60.46%|.4548|82.43%|.7995|51.00%|
|max2 / mean+max+std|74.13%|70.64%|74.67%|60.56%|.4565|82.43%|.7995|51.00%|
|mean+max+std / mean+max+std|73.20%|69.46%|74.67%|59.07%|.4375|85.18%|.8761|52.89%|

Training with statistical pooling improves final spatial correspondence, local
color accessibility and final-layer Pet segmentation relative to frozen statistical
extraction, but lowers five-fold, few-shot and retrieval scores while tying LOO.
Its intermediate Pet mIoU is49.96%, so the final-layer advantage is not uniform
across layers. ResNet18 intermediate Pet probes reach52.52%, versus50.34% final.
No intervention consistently improves every task. All findings remain single-seed
comparisons and do not establish hidden leaderboard performance.

Statistical SSL allocation199245 completed in7h12m28s; measured epoch training
wall time6.61h, peak GPU allocation10.12GiB/reservation13.75GiB on TITAN RTX.
All experiment/evaluation allocations, including preflight and recovered attempts,
total9.20 GPU allocation-hours; parent allocation accounting is retained in
`experiment_outputs/green-area/final_accounting.json`. No leaderboard upload occurred.
