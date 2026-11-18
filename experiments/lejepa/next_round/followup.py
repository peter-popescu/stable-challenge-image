"""Evaluate exactly one completed new encoder, never a partial last.ckpt.

Submitted with afterok on its training allocation. Does not submit child jobs.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


def publish_manifest(run, spec):
    """Coordinate identical global/DTD/CIFAR writers for one completed run."""
    import fcntl
    from ..sweep import atomic_json
    manifest = run/'evaluation-model.json'
    with (run/'evaluation-model.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if manifest.exists():
            if json.loads(manifest.read_text()) != [spec]:
                raise ValueError('Manifest changed')
        else:
            atomic_json(manifest, [spec])
    return manifest


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('stage', choices=['extract','cpu','external-extract','external-cpu'])
    parser.add_argument('--run', required=True, choices=['resnet18-tuned-400','resnet9-original-400','resnet9-more-views-200'])
    parser.add_argument('--dataset', choices=['dtd','cifar100','clevr'])
    args = parser.parse_args()
    if not os.environ.get('SLURM_JOB_ID'): parser.error('Run through Slurm')
    import torch
    run = Path('experiment_outputs/next-round/training')/args.run
    expected = json.loads((Path(__file__).parent/(args.run+'.json')).read_text())
    checkpoint = run/'checkpoints/last.ckpt'
    state = torch.load(checkpoint, map_location='cpu', weights_only=False)
    if state['hyper_parameters']['config'] != expected or state['epoch']+1 != expected['epochs']:
        raise ValueError('Follow-on requires the exact completed training endpoint')
    name = args.run+'-seed'+str(expected['seed'])
    spec = dict(name=name, backbone=expected.get('backbone','resnet9'), checkpoint=str(checkpoint))
    manifest = publish_manifest(run, spec)
    if args.stage.startswith('external'):
        if not args.dataset: parser.error('--dataset required for external stages')
        command = ['experiments.lejepa.next_round.external_run', args.stage.split('-')[1],
            '--manifest',str(manifest),'--dataset',args.dataset,
            '--output',str(Path('experiment_outputs/next-round/external-new')/name)]
    else:
        command = ['experiments.lejepa.next_round.frozen',args.stage,'--manifest',str(manifest),
            '--output',str(Path('experiment_outputs/next-round/frozen-new')/name)]
    subprocess.run([sys.executable,'-u','-m',*command],check=True)
    subprocess.run([sys.executable,'-u','-m','experiments.lejepa.next_round.report'],check=True)


if __name__ == '__main__': main()
