# Frozen readouts and complementary encoders

Commands run through `experiments/lejepa/downstream/slurm.sh`, which preserves
repository-local offline caches and limits BLAS threads to the Slurm allocation.
All GPU work requests compatible sm75+ devices. No leaderboard uploads occur.

Jobs: fixture test **203187** (passed 3 tests), extraction **203189** (GPU, 4 CPU,
24 GiB, 2h limit), CPU evaluation **203190** (4 CPU,24 GiB,6h limit), ONNX
selection/export **203191** (4 CPU,16 GiB,1h limit). Supplemental seed43 extraction **203198** (GPU,4CPU,16GiB,30min)
was added after the main extractor had imported its nine-model list. CPU203190
depends on both extraction jobs; export203191 depends on CPU203190. Distinct
--only plan/status files retain provenance of the supplemental extraction. Rejected initial extraction submission lacked the `gpus` partition;
it allocated no job or computation.

Existing verified mid/final maps are reused. Earlier stages are extracted once per
encoder; only pooled embeddings are saved, avoiding redundant spatial-map storage.
Any preexisting cache must match checkpoint/data/row identities and content hashes.
Both tuned seed43/44 checkpoints were verified by provenance audit203177; these
form the same-recipe initialization-diversity pair. Tuned seed42 duplication is a
separate sanity control. All output is under `experiment_outputs/next-round/frozen`.

ResNet9 stage endpoints are conv indices2 (64 channels),7 (128 channels after
first residual),11 (256 after second spatial pool),16 (256 after final residual).
All means concatenate to704 dimensions. Last-two mean+max produces1024; intermediate
mean plus final mean+max produces768; intermediate mean plus final mean+max+std
produces1024. ResNet18 stage widths40/80/160/320 give600 all means,960 last-two
mean+max,920 all means plus final max,960 final mean+max+std. Population std uses
the existing implementation. Lower native dimensions are padded only for the official
classification/interface. Direct concatenation retains independently trained coordinates.

Ten encoders cover six primary trained checkpoints, two random architecture controls,
and tuned200 seed43/44. Every readout uses identical fixed held-out folds, official
feature standardization/ridge selection, few-shot episodes, retrieval/kNN/clustering,
and global appearance train/validation/test partitions. No new spatial probe is run.

Complementarity records agreement/disagreement, all four correctness partitions,
per-class and per-fold results, diagnostic oracle and observed concatenation gain.
The report describes their across-pair association without treating shared-encoder
pairs as independent observations. Duplicate features can change effective ridge
regularization even after feature standardization.

Export selection takes the top two distinct pairs by development outer accuracy,
plus original+statpool to preserve scientific diversity (or the next ranked pair if
already included), and the best single readout for each trained architecture.
Selection is explicitly development based. Every export is self-contained, <20 MiB,
CPU compatible, checked with dynamic batch1/3/4, local cached embedding parity and
all official submission checks. Fresh-process CPU inference reports median latency
at batch1/4 with one thread and peak process RSS; RSS includes runtime/model allocation.
Parameter counts, actual size and SHA256 accompany artifact paths. Exporter process
RSS is also recorded but is not interpreted as isolated inference memory.

Monitor with `squeue -j 203189,203190,203191` and
`sacct -j 203187,203189,203190,203191 --format=JobID,State,Elapsed,AllocTRES,MaxRSS,ExitCode`.
Logs are `experiment_outputs/next-round/frozen/{extract,cpu,export}-JOBID.log`.
Use `python -m experiments.lejepa.next_round.frozen {extract,cpu,export}` inside
Slurm for reproducible reruns; caches/results are verified before reuse.
