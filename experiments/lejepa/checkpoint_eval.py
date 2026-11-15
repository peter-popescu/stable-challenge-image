"""Export and evaluate one saved checkpoint, optionally compare duration milestones."""
import argparse
import json
import os
from pathlib import Path
import sys


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--checkpoint', required=True, type=Path)
    p.add_argument('--expected-epoch', required=True, type=int, help='Completed epochs, one based')
    p.add_argument('--data', required=True, type=Path)
    p.add_argument('--folds', required=True, type=Path)
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--compare-duration-root', type=Path)
    a = p.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):
        p.error('Run extraction and evaluation within Slurm')
    import torch
    from .export import export
    from . import cross_validate
    from .sweep import digest, atomic_json

    checkpoint = torch.load(a.checkpoint, map_location='cpu', weights_only=False)
    if checkpoint['epoch'] + 1 != a.expected_epoch:
        raise ValueError('Checkpoint epoch does not match requested milestone')
    config = checkpoint['hyper_parameters']['config']
    del checkpoint
    a.output.mkdir(parents=True, exist_ok=True)
    export(a.checkpoint, a.output / 'model.onnx')
    atomic_json(a.output / 'provenance.json', {'checkpoint': str(a.checkpoint.resolve()),
                'checkpoint_sha256': digest(a.checkpoint), 'completed_epochs': a.expected_epoch,
                'config': config, 'onnx_sha256': digest(a.output / 'model.onnx')})
    sys.argv = ['cross_validate', '--model', str(a.output / 'model.onnx'),
                '--data', str(a.data), '--folds', str(a.folds),
                '--output', str(a.output / 'validation')]
    cross_validate.main()
    if a.compare_duration_root:
        root = a.compare_duration_root
        paths = {200:root/'epoch200', 300:root/'epoch300', 400:root}
        paths[a.expected_epoch] = a.output
        rows = []
        for epoch, directory in sorted(paths.items()):
            result = json.loads((directory / 'validation/results.json').read_text())
            if result['data_sha256'] != digest(a.data) or result['fold_manifest_sha256'] != digest(a.folds):
                raise ValueError('Milestone data/folds differ')
            trained = result['results']['trained']
            rows.append({'completed_epochs':epoch, **trained['metrics'], 'folds':trained['folds']})
        atomic_json(root / 'comparison_with350.json', rows)
        lines = ['# Duration milestones, 400-epoch schedule, seed 43', '',
                 '| Epoch | Five-fold accuracy | Macro accuracy | Official local LOO |',
                 '|---|---:|---:|---:|']
        for row in rows:
            lines.append(f'| {row["completed_epochs"]} | {row["accuracy"]:.2%} | {row["macro_accuracy"]:.2%} | {row["official_local_loo_accuracy"]:.2%} |')
        lines += ['', 'Same training run, fixed folds and planned 400-epoch schedule. Checkpoint evaluation does not retrain the encoder.',
                  'These are single-seed development comparisons; choosing an epoch using these scores is model selection, not an independent test.']
        (root / 'MILESTONES.md').write_text('\n'.join(lines) + '\n')
        print('\n'.join(lines), flush=True)


if __name__ == '__main__':
    main()
