# Shared scratch storage

Use `/nbu/people/ppopescu/stable-challenge-image-next-round` for future bulk
experiment data. The user authorized this shared scratch location and all three
external dataset downloads. Repository paths remain stable through symlinks:

| Repository path | Scratch target |
|---|---|
|`experiment_outputs/next-round/external`|`/nbu/people/ppopescu/stable-challenge-image-next-round/external`|
|`experiment_outputs/next-round/external-new`|`/nbu/people/ppopescu/stable-challenge-image-next-round/external-new`|

`/nbu` resolves to `/cs/local/nbu`. Brown CS's
[filesystem guide](https://cs.brown.edu/about/system/services/files/filesystem/)
describes it as shared, unbacked-up scratch. Keep source code, small manifests and
migration audits in the repository. Persist new external images, annotations,
pooled caches and downstream probe weights under the scratch-backed paths above.
Keep external evaluation artifacts separate from SSL training and competition
exports.

## Verified migration on 2026-11-17

Slurm job **203401** completed successfully (`0:0`) in **8m46s**, using2 CPUs and
8GiB on `echidna.cs.brown.edu`. The compute node verified ownership and an exclusive
write/read/fsync cycle. Available scratch space before copying was
**59,995,128,332,288 bytes**, above the100GiB minimum.

The inactive external tree contained **6,194 files totaling7,585,151,437 bytes**.
Every file's SHA256, size, mode, mtime, owner UID and group GID matched the copied
scratch tree. A second source inventory checked stability before bounded directory
rename/symlink publication. Every home file was rehashed against its scratch copy
immediately before individual unlink; only empty recorded directories were then
removed. A final complete scratch inventory matched the original. Both repository
symlinks were independently checked with `samefile`; the temporary home backup and
scratch staging directory are absent. Existing cache path strings and file hashes
were preserved.

Audits are outside the relocated tree:

- `experiment_outputs/next-round/reporting/external-storage.json`: completed status.
- `experiment_outputs/next-round/reporting/storage-migration-203401/`: initial and
  final-phase Slurm `scontrol`/`sacct SubmitLine` snapshots, full source/scratch file
  inventories, status and6,194 per-file verified-removal records.
- `experiment_outputs/next-round/reporting/storage-migration-203401.log`: execution log.

The source inventory SHA256 is
`fbf2784cfaa7160576df1b481d3665a654f0c200093ca61bbe755d03eee25dac`.
This is the fixed migration snapshot; newly acquired CLEVR files are subsequent
artifacts and are not included in its6,194-file count.

No active external consumers were present in the recorded checks. Active SSL
writers **203253/203254/203255** remain in their original home training directories.
Move those directories only after their writers finish and after the same
verification procedure. The unrelated interactive allocation203388 was inspected
only and remained untouched.

Initial migration203398 was cancelled before source publication after discovering
that path equality needed to resolve the `/nbu` system symlink on both sides.
Attempt203400 was cancelled during inventory to implement safe partial-copy reuse.
Successful203401 reused17 staged files (1,446,864,259 bytes) after exact verification
and copied the remaining6,177 files (6,138,287,178 bytes); its verified staging tree
became the published target. All attempt logs/audits are retained.

## Prior resource decisions

The share was provisioned after initial storage checks found no writable personal
shared location. Authorized home cleanup and cache policy changes are documented
in [CLEANUP.md](CLEANUP.md); launch and training replacement details remain in
[LAUNCH.md](LAUNCH.md). Preflight203186 failed its space check; successful203228
supported replacement training203253/203254/203255. Superseded203199/203200/203201
remain cancelled.

Compute-node outbound probe203213 timed out. Authorized transfer-only fallbacks
must perform no decoding or decompression on the login host. Verification,
extraction, feature computation and downstream probes run through Slurm. Global
feature caches persist pooled embeddings only; full spatial maps require explicit
dense-task need.
