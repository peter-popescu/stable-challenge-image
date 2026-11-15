"""Serve the installed stable-pretraining dashboard on loopback only."""
import argparse
import os
from pathlib import Path


def main():
    for name in ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS']:
        os.environ.setdefault(name, '1')
    root = Path(__file__).resolve().parents[2]
    os.environ.setdefault('SPT_CACHE_DIR', str(root / 'experiment_outputs/cache/stable-pretraining'))
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--directory', type=Path, default=root / 'experiment_outputs')
    p.add_argument('--port', type=int, default=4242)
    a = p.parse_args()
    a.directory.mkdir(parents=True, exist_ok=True)
    from stable_pretraining.web import serve
    serve(a.directory, host='127.0.0.1', port=a.port, poll_interval=5.0)


if __name__ == '__main__':
    main()
