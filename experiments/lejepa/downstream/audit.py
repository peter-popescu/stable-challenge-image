"""Official export checks and consolidated report refresh inside Slurm."""
import os
import sys
from pathlib import Path
from ..sweep import run_command,atomic_json,digest
from .common import ROOT


def main():
    if not os.environ.get('SLURM_JOB_ID'):raise RuntimeError('Run in Slurm')
    output=ROOT/'resnet18-pooling/resnet18-tuned-200-seed42/exports'
    statuses={}
    for path in sorted(output.glob('*.onnx')):
        run_command([sys.executable,'test_submission.py',path],output/(path.stem+'-official-validator.log'))
        statuses[path.stem]=dict(official_validator='passed',sha256=digest(path),bytes=path.stat().st_size)
    atomic_json(ROOT/'resnet18-pooling/official-validation.json',statuses)
    from .report import report
    report()


if __name__=='__main__':main()
