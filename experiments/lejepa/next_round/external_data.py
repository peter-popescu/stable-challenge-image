"""External evaluation adapters; never imported by SSL training."""
import json
from pathlib import Path
import numpy as np
from PIL import Image

DATA = Path('experiment_outputs/next-round/external/data')
THRESHOLDS = np.asarray([0., 8., 8.5, 9., 9.5, 10., 100.])


def clevr_labels(objects):
    # TFDS stores pixel_coords as float32 before VTAB applies thresholds.
    distance = np.asarray([o['pixel_coords'][2] for o in objects], dtype=np.float32).min()
    count = len(objects) - 3
    label = int(np.flatnonzero(THRESHOLDS < distance).max())
    if not 0 <= count < 8 or not 0 <= label < 6:
        raise ValueError('CLEVR annotation outside VTAB classes')
    return count, label


def stratified_validation(labels, per_class=50, seed=42):
    rng = np.random.default_rng(seed)
    train, val = [], []
    for c in np.unique(labels):
        order = rng.permutation(np.flatnonzero(labels == c))
        val.extend(order[:per_class]); train.extend(order[per_class:])
    return np.sort(train), np.sort(val)


class Dataset:
    def __init__(self, name, root=DATA):
        from torchvision.datasets import CIFAR100, DTD
        self.name = name
        self.root = Path(root)
        self.source_indices = None
        if name == 'cifar100':
            self.parts = [CIFAR100(self.root/'cifar100', train=t, download=False) for t in [True,False]]
            self.labels = np.concatenate([np.asarray(d.targets) for d in self.parts])
            self.ids = [f'{s}/{i:05d}' for s,d in zip(['train','test'],self.parts) for i in range(len(d))]
            self.splits = dict(zip(['train','val'],stratified_validation(self.labels[:50000])))
            self.splits['test'] = np.arange(50000,60000)
            self.splits['trainval'] = np.arange(50000)
        elif name == 'dtd':
            # A fixed three-of-ten split subset bounds probe fitting cost.
            self.parts = [DTD(self.root, split=s, partition=1, download=False) for s in ['train','val','test']]
            self.paths = sorted({Path(p) for d in self.parts for p in d._image_files})
            self.ids = [str(p.relative_to(self.root/'dtd')) for p in self.paths]
            classes = sorted({p.parent.name for p in self.paths})
            self.labels = np.asarray([classes.index(p.parent.name) for p in self.paths])
            lookup = {str(p): i for i,p in enumerate(self.paths)}
            self.split_sets = {}
            for partition in [1,2,3]:
                splits = {}
                for split in ['train','val','test']:
                    d = DTD(self.root, split=split, partition=partition, download=False)
                    splits[split] = np.asarray([lookup[str(p)] for p in d._image_files])
                splits['trainval'] = np.concatenate([splits['train'],splits['val']])
                self.split_sets[str(partition)] = splits
            self.splits = self.split_sets['1']
        elif name == 'clevr':
            base = self.root/'clevr/CLEVR_v1.0'
            self.paths, labels, self.ids = [], [], []
            for split in ['train','val']:
                scenes = json.loads((base/f'scenes/CLEVR_{split}_scenes.json').read_text())['scenes']
                scenes = sorted(scenes,key=lambda x:x['image_filename'])
                for scene in scenes:
                    self.paths.append(base/'images'/split/scene['image_filename'])
                    self.ids.append(split+'/'+scene['image_filename'])
                    labels.append(clevr_labels(scene['objects']))
            if len(labels) != 85000: raise ValueError('Require full official 70000/15000 CLEVR partitions')
            self.labels = np.asarray(labels)
            # Official partition roles match VTAB; filename ordering differs from
            # TFDS hashed record order and is explicitly named in the protocol.
            self.splits = dict(train=np.arange(63000),val=np.arange(63000,70000),
                               trainval=np.arange(70000),test=np.arange(70000,85000))
        else: raise ValueError(name)

    def __len__(self): return len(self.ids)

    def image(self,i):
        if self.source_indices is not None: i = int(self.source_indices[i])
        if self.name == 'cifar100':
            return self.parts[0][i][0] if i < 50000 else self.parts[1][i-50000][0]
        with Image.open(self.paths[i]) as im: return im.convert('RGB')

    def exclude(self, excluded):
        keep = np.asarray([i for i in range(len(self)) if i not in set(excluded)])
        remap = np.full(len(self),-1,dtype=np.int64); remap[keep] = np.arange(len(keep))
        self.source_indices = keep
        self.ids = [self.ids[i] for i in keep]; self.labels = self.labels[keep]
        def update(splits):
            return {k:remap[v][remap[v]>=0] for k,v in splits.items()}
        self.splits = update(self.splits)
        if hasattr(self,'split_sets'): self.split_sets = {k:update(v) for k,v in self.split_sets.items()}

    def plan(self):
        from ..sweep import digest
        if self.name == 'cifar100':
            sources = list((self.root/'cifar100/cifar-100-python').glob('*'))
        elif self.name == 'dtd':
            sources = list((self.root/'dtd/dtd/labels').glob('*.txt'))
            archives = list((self.root/'dtd').glob('*.tar.gz'))
            sources += archives
        else:
            sources = list((self.root/'clevr/CLEVR_v1.0/scenes').glob('*.json'))
            archive = self.root/'clevr/CLEVR_v1.0.zip'
            if archive.exists(): sources.append(archive)
        return dict(source_sha256={str(p):digest(p) for p in sources if p.is_file()},dataset=self.name,row_ids=self.ids,labels=self.labels.tolist(),
                    splits={k:v.tolist() for k,v in self.splits.items()},
                    split_sets={k:{s:ids.tolist() for s,ids in v.items()} for k,v in getattr(self,'split_sets',{}).items()},
                    preprocessing='PIL RGB; bilinear resize96x96; float32 RGB/255; challenge mean/std',
                    evaluation_only=True,seed=42)
