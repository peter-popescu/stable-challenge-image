#!/usr/bin/env bash
set -euo pipefail
: "${SLURM_JOB_ID:?Run bounded endpoint preflight in Slurm}"
exec curl --head --location --fail --connect-timeout 10 --max-time 15 https://dl.fbaipublicfiles.com/clevr/CLEVR_v1.0.zip
