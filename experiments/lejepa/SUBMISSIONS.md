# Submission history and jitter transfer check

Leaderboard scores below were provided by the user on2026-11-16. Identifiers
are recorded as supplied; they are not asserted to match SHA256 of the local
ONNX artifacts. Local scores use the development population, not the hidden set.

| Model | User-provided submission identifier | Five-fold | Local LOO | Leaderboard |
|---|---|---:|---:|---:|
| First baseline,200 epochs | `869f65b3caa341222b2518b3666bccf764d8365e24646192e6d303e59d7c7405` | — | 68.40% | 62.60% |
| Tuned seed42,200 epochs | `93ac0d2c9ef9425b4899b3c9af51d7d378322ee721ce756c61489cd6247fb296` | 73.07% | 72.40% | 61.37% |
| Tuned seed43,400 epochs | `c5a48edfc111f2bfe73a593ece4b52c44c4325cb67ee3554b8f8a53b4bf20fc2` | 76.00% | 73.47% | 62.53% |
| Stronger jitter,seed42,200 epochs | Pending upload | 72.80% | 72.13% | Pending |

The supplied LOO-minus-leaderboard gaps are5.80,11.03,and10.94 percentage points.
These are descriptive gaps between different evaluation populations/protocols,
not estimates of a single correction factor or proof of a particular failure
mechanism. Improved local scores did not improve these three reported hidden
scores. This motivates testing a different invariance policy, not assuming
that lower SSL loss proves better transfer.

## Prepared candidate

Use the predefined200-epoch endpoint from `jitter-strong` in the augmentation
study. It retains2 global96² and2 local48² views, control crop distributions,
and uses jitter B/C/S=.4, H=.1, p=.8. Grayscale remains.05; blur/solarization
remain disabled. LR=.003, lambda=.005, seed42, planned200-epoch cosine.

Submission file:
`experiment_outputs/submissions/jitter-strong-seed42-epoch200/lejepa-resnet9-jitter-strong-seed42-epoch200.onnx`.

SHA256: `6eb03d72f752a4f47b7afa04d45eb1a17b1a495bbc92a7e2f529a2d083ffe809`.
Size:9,768,047 bytes (~9.32MiB), below20MiB. The file is copied byte-for-byte
from the already exported/evaluated encoder; projector is excluded. Checkpoint,
configuration, data/fold and ONNX provenance accompany it. `submission.json`
and validator output record readiness; preparation alone is not an upload.

The150-epoch alternative scored73.20% five-fold and72.40% LOO, versus72.80%
and72.13% at200. These tiny local differences do not predict hidden transfer;
the200-epoch endpoint avoids choosing an intermediate checkpoint after inspecting
the curves. Neither candidate has a measured leaderboard result yet.

Controlled color-jitter cosine increased from.9897(control) to.9970(strong jitter)
at200, while mismatched-image cosine was similar (.7578 versus.7567). This
supports a change in measured color invariance. Lower training invariance loss
is not a cross-policy representation-quality scale: augmentation changes the
prediction problem itself. A hidden evaluation is an experiment, not evidence
of a better model in advance. Further selection using leaderboard scores also
turns that set into model-selection data.

Upload the ONNX through the challenge page after the official local validator
passes. The repository's documented submission policy uses the latest accepted
model, not the best historical score; preserve the older artifacts. No uploads
or changes to the supplied evaluator are performed by this preparation.

Official local submission validation passed all six checks in CPU Slurm job185376
on2026-11-16. The prepared file is ready for upload and has not been uploaded.
