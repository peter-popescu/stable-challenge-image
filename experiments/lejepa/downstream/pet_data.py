"""Evaluation-only Oxford-IIIT Pet acquisition through torchvision checksums."""
import os
from pathlib import Path

ROOT=Path('experiment_outputs/external-eval/data')


def main():
    if not os.environ.get('SLURM_JOB_ID'): raise RuntimeError('Run in Slurm')
    from torchvision.datasets import OxfordIIITPet
    from ..sweep import atomic_json, digest
    ROOT.mkdir(parents=True,exist_ok=True)
    dataset=OxfordIIITPet(ROOT,split='trainval',target_types='segmentation',download=True)
    test=OxfordIIITPet(ROOT,split='test',target_types='segmentation')
    annotations=ROOT/'oxford-iiit-pet/annotations'
    atomic_json(ROOT/'source.json',dict(source='https://www.robots.ox.ac.uk/~vgg/data/pets/',
        evaluation_only=True,trainval=len(dataset),test=len(test),
        resources=OxfordIIITPet._RESOURCES,
        split_sha256={name:digest(annotations/name) for name in ['trainval.txt','test.txt']},
        job_id=os.environ['SLURM_JOB_ID']))


if __name__=='__main__': main()
