#!/usr/bin/env bash
set -euo pipefail
: "${SLURM_JOB_ID:?Run bounded network check in Slurm}"
for url in https://www.robots.ox.ac.uk/~vgg/data/dtd/download/dtd-r1.0.1.tar.gz https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz https://dl.fbaipublicfiles.com/clevr/CLEVR_v1.0.zip; do
  curl --head --location --fail --connect-timeout 10 --max-time 15 "$url"
done
