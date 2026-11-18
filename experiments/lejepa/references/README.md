# Teammate backbone reference

`dino_backbones.py` is copied from `backbones.py` on the teammate's
`dino-experiment` branch, commit `e718fe83b85a6ce3e1ea10dfc39785e3baaf6307`:

https://github.com/peter-popescu/stable-challenge-image/blob/e718fe83b85a6ce3e1ea10dfc39785e3baaf6307/backbones.py

Git blob: `3c0c0ee460c9602e66170a72d07e845fd518432b`.
Read through the connected GitHub tool on2026-11-17. The locally cached branch
was older; SSH fetch was unavailable, so the connected read-only service verified
the current file. No authentication configuration was changed.

`test_teammate_architecture_exact` checks identical seeded parameter tensors,
state keys and outputs at96px and48px against `SmallResNet18(width=40)`. The
production adapter adds only `embed_dim`, `spatial()` and a parameter-free readout
interface needed by LeJEPA and diagnostics. GAP equals the teammate's1x1 adaptive
average pooling. No learned320→256 projection is introduced.
