#!/usr/bin/env bash
# Transfer bytes only. Image processing and archive extraction belong in Slurm.
set -euo pipefail
transfer_limit_seconds=10800
case "${1:?dataset}" in
  dtd) target=experiment_outputs/next-round/external/data/dtd/dtd-r1.0.1.tar.gz; url=https://www.robots.ox.ac.uk/~vgg/data/dtd/download/dtd-r1.0.1.tar.gz ;;
  cifar100) target=experiment_outputs/next-round/external/data/cifar100/cifar-100-python.tar.gz; url=https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz ;;
  clevr) transfer_limit_seconds=43200; target=experiment_outputs/next-round/external/data/clevr/CLEVR_v1.0.zip; url=https://dl.fbaipublicfiles.com/clevr/CLEVR_v1.0.zip ;;
  *) exit 2 ;;
esac
mkdir -p "$(dirname "$target")"
# Conservative minimum free space covers archive, decoded data and pooled caches.
python3 - "$1" "$(dirname "$target")" <<'CHECK'
import shutil,sys
needed={'dtd':3,'cifar100':2,'clevr':60}[sys.argv[1]]*1024**3
free=shutil.disk_usage(sys.argv[2]).free
if free<needed:
    raise SystemExit(f'Insufficient free storage: {free/1024**3:.2f}GiB available, require {needed/1024**3:.0f}GiB for this dataset pipeline; no transfer started')
CHECK
if [ ! -f "$target" ]; then
  curl --fail --location --retry 2 --connect-timeout 20 --max-time "$transfer_limit_seconds" --continue-at - --output "$target.partial" "$url"
  mv "$target.partial" "$target"
fi
