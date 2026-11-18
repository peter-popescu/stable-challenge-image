# Authorized storage cleanup — 2026-11-17

The user explicitly authorized pruning reproducible caches and temporary test
checkpoints, retaining encoder weights and useful milestones, and stripping
optimizer state from archival checkpoints that will not be resumed.

## Completed Pet evaluation arrays

Inventory203220 and cleanup203221 identified and removed exactly27 ordinary
`mid.npy`, `final.npy`, and `masks.npy` arrays across nine completed encoders:
23,637,209,472 bytes (22.014GiB). No recursive deletion was performed.
Root/layer metrics, `probe.pt`, histories, examples, original `cache.json`, split
IDs and original array hashes remain. Each cache has a `pruned.json` tombstone.

Every array's original hash was verified before deletion. Preservation fixtures
203226 and203229 checked all nine endpoints, all retained artifact hashes and
array absence. Encoder loading/cache regeneration was forbidden during reuse;
all completed evaluations returned their saved results. `segmentation.run` now
recognizes complete results before any encoder load or cache regeneration.

Audit: `experiment_outputs/external-eval/probes/cache-prune-plan-applied.json`.
The original plan also records the preserved artifact hashes and completion
checks. Regeneration is necessary only for an explicitly requested new spatial
probe; completed metrics and visualization access do not require those arrays.

## Temporary checkpoints and archival milestones

Thirty temporary test checkpoint copies from terminal preflight allocations
were deleted (1.767GiB logical). Passed, failed and timed-out test logs/configs
and error traces remain. Evidence and per-file inventory:
`experiment_outputs/next-round/provenance/temporary-checkpoint-cleanup.json`.

Archival job203227 completed a fixed list of62 historical retained milestones.
It writes distinct `.weights.ckpt` files, removes **only optimizer_states**, and
preserves complete model/projector/SIGReg state, config, epoch and scheduler
metadata. Exact tensor/nonoptimizer metadata equality and strict encoder loading
are checked before the original full checkpoint is removed. All `last.ckpt`
reference files and ordinary final-epoch checkpoint copies remain unchanged.
Useful milestone encoder weights remain available in the new archives.

Original checkpoint hashes in historical provenance are preserved. Per-original
`.archived.json` tombstones, adjacent `archived_checkpoint` records and the central
`provenance/checkpoint-archives.json` map original paths/hashes to new paths/hashes.
These archives are for inference, not optimizer-state resume. Consult the central
manifest for all mappings. Verified savings:4,644,115,862 bytes (4.325GiB).
Combined cache, temporary checkpoint and optimizer cleanup reclaimed about28.1GiB,
raising available home space from~1.9GiB to~30GiB before new experiment writes.

## Active consumers

Cleanup recorded own-user Slurm queue/job details before planning and mutation.
The parent also inspected full `sacct SubmitLine` arguments, since a generic
launcher path alone does not identify its workload. Active export203191 reads
unchanged final `last.ckpt` files and pooled frozen caches. Future training and
follow-up jobs reference new output directories. None consumes deleted Pet arrays
or the62 archived historical milestone paths. No unrelated processes were killed
and no historical final checkpoint identity was rewritten.

## Future global extraction

`downstream.run extract-global` persists only
`MODEL/global-cache/{embeddings.npz,cache.json}`. It pools each batch immediately;
it writes no spatial maps and runs no correspondence/local RGB evaluation.
`cpu-global` validates pooled caches and produces separate global metrics/reports.
Dense map extraction remains explicit for spatial tasks. Existing next-round
multiscale/ensemble and external pipelines already store pooled embeddings only.

Slurm203225 passed three fixtures: native readout parity for real random ResNet9
and ResNet18; no spatial array files; exact frozen/BatchNorm state preservation;
cache reuse and tamper rejection; CPU-global tasks with spatial access forbidden.
See `tests/test_downstream_global.py` and `downstream/PROTOCOL.md`.

## Training recovery

Original preflight203186 failed before writes because its allocated node had
only5.03GiB free in `/ltmp`; the8GiB guard prevented saturation. After the Pet
cleanup made~26GiB available at home, replacement203228 started with guarded
repository storage. The old training holds converted to administrator-held state
when their failed-preflight dependencies were updated; releasing them was denied.
Identical replacement jobs203253/203254/203255 have afterok203228 dependencies
and no manual holds. All frozen/external child dependencies were rewired first;
the obsolete pending203199/203200/203201 jobs were then cancelled. Configurations,
outputs, resources and successful-preflight guards are unchanged. CLEVR still
awaits larger shared storage; it is not
downloaded into the reclaimed space without its60GiB capacity requirement.
