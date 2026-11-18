#!/usr/bin/env bash
set -euo pipefail
: "${SLURM_JOB_ID:?Run through Slurm}"
root="$PWD/experiment_outputs/external-eval/data/oxford-iiit-pet"
mkdir -p "$root"
curl --fail --location --retry 2 --connect-timeout 20 --max-time 1800 --output "$root/images.tar.gz" https://thor.robots.ox.ac.uk/~vgg/data/pets/images.tar.gz
curl --fail --location --retry 2 --connect-timeout 20 --max-time 300 --output "$root/annotations.tar.gz" https://thor.robots.ox.ac.uk/~vgg/data/pets/annotations.tar.gz
exec bash experiments/lejepa/downstream/slurm.sh experiments.lejepa.downstream.pet_data
