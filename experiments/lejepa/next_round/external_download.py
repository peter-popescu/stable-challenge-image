"""Explicitly approved evaluation-only acquisition, never called by training."""
import argparse
import hashlib
import os
import shutil
import urllib.request
import zipfile
from pathlib import Path
from .external_data import DATA
from ..sweep import atomic_json,digest


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--dataset',required=True,choices=['dtd','cifar100','clevr'])
    p.add_argument('--approved-large-download',action='store_true',required=True)
    a=p.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):p.error('Dataset acquisition and extraction must run in Slurm')
    DATA.mkdir(parents=True,exist_ok=True)
    needed={'dtd':3,'cifar100':2,'clevr':60}[a.dataset]*1024**3
    free=shutil.disk_usage(DATA).free
    if free<needed:
        raise RuntimeError(f'Insufficient storage before acquisition/extraction: {free/1024**3:.2f}GiB free; require {needed/1024**3:.0f}GiB; no archive writes started')
    if a.dataset=='clevr':
        target=DATA/'clevr';target.mkdir(exist_ok=True)
        archive=target/'CLEVR_v1.0.zip'
        if not archive.exists():
            partial=archive.with_suffix('.zip.partial')
            urllib.request.urlretrieve('https://dl.fbaipublicfiles.com/clevr/CLEVR_v1.0.zip',partial)
            partial.replace(archive)
        official_sha256='5cd61cf1096ed20944df93c9adb31e74d189b8459a94f54ba00090e5c59936d1'
        actual_sha256=digest(archive)
        if archive.stat().st_size!=19021600724 or actual_sha256!=official_sha256:
            raise ValueError('CLEVR archive differs from official TensorFlow Datasets byte count/SHA256')
        with zipfile.ZipFile(archive) as z:
            # Skip unlabeled test partition and questions; no need to expand them.
            names=[n for n in z.namelist() if n.startswith(('CLEVR_v1.0/images/train/','CLEVR_v1.0/images/val/','CLEVR_v1.0/scenes/'))]
            for name in names:
                dest=target/name
                if not dest.resolve().is_relative_to(target.resolve()):raise ValueError('Unsafe archive path')
                info=z.getinfo(name)
                if not dest.exists() or (not info.is_dir() and dest.stat().st_size!=info.file_size):
                    z.extract(name,target)
        atomic_json(target/'source.json',dict(url='https://dl.fbaipublicfiles.com/clevr/CLEVR_v1.0.zip',sha256=actual_sha256,official_sha256=official_sha256,checksum_source='https://raw.githubusercontent.com/tensorflow/datasets/master/tensorflow_datasets/datasets/clevr/checksums.tsv',evaluation_only=True,job_id=os.environ['SLURM_JOB_ID']))
    else:
        from torchvision.datasets import DTD,CIFAR100
        cls=DTD if a.dataset=='dtd' else CIFAR100
        cls(DATA if a.dataset=='dtd' else DATA/a.dataset,download=True)
        atomic_json(DATA/a.dataset/'source.json',dict(dataset=a.dataset,evaluation_only=True,torchvision_checksums=True,job_id=os.environ['SLURM_JOB_ID']))


if __name__=='__main__':main()
