"""Fixed lightweight CLEVR count/distance evaluation; never the official VTAB1k.

Reuse validated full-dataset identities and complete full-encoder caches; infer
only1000 training,500 validation and2000 test images for remaining encoders.
"""
import argparse
import json
import os
from pathlib import Path
import time
import numpy as np
from .external_data import DATA, Dataset
from . import external_run
from ..downstream.common import identity
from ..sweep import atomic_json,digest

ROOT=external_run.ROOT/'clevr-light'
PARENT=external_run.ROOT/'clevr'
SIZES=dict(train=1000,val=500,test=2000)
RUNS=('resnet18-tuned-400','resnet9-original-400','resnet9-more-views-200')


def choose_indices(splits, sizes=SIZES, seed=42):
    """No labels enter sampling; one PCG64 stream in train/val/test order."""
    rng=np.random.default_rng(seed);chosen=[];out={};offset=0
    for name in ['train','val','test']:
        bank=np.asarray(splits[name],dtype=np.int64)
        if len(np.unique(bank))!=len(bank) or len(bank)<sizes[name]:raise ValueError('Insufficient/nonunique partition rows')
        take=np.sort(rng.choice(bank,size=sizes[name],replace=False))
        chosen.extend(take.tolist());out[name]=np.arange(offset,offset+len(take),dtype=np.int64);offset+=len(take)
    if len(set(chosen))!=len(chosen):raise ValueError('Parent partitions overlap')
    out['trainval']=np.concatenate([out['train'],out['val']])
    return np.asarray(chosen,dtype=np.int64),out


class Subset:
    name='clevr'
    def __init__(self, base, sizes=SIZES, seed=42):
        self.base=base;self.selected,self.splits=choose_indices(base.splits,sizes,seed)
        self.ids=[base.ids[int(i)] for i in self.selected]
        self.labels=np.asarray(base.labels)[self.selected]
        self.split_sets={'clevr-light':self.splits}
    def __len__(self):return len(self.ids)
    def image(self,i):return self.base.image(int(self.selected[i]))


def checked_parent():
    """Verify annotation/metadata provenance, without any full image rescan."""
    path=PARENT/'dataset.json';overlap=PARENT/'overlap.json'
    plan=json.loads(path.read_text());screen=json.loads(overlap.read_text())
    if plan['overlap_sha256']!=digest(overlap):raise ValueError('Parent overlap audit differs')
    if screen['challenge_sha256']!=digest('data/train.parquet') or screen['data_adapter_sha256']!=digest(Path(external_run.__file__).with_name('external_data.py')):
        raise ValueError('Parent challenge/adapter provenance differs')
    source=json.loads((DATA/'clevr/source.json').read_text())
    if source['sha256']!='5cd61cf1096ed20944df93c9adb31e74d189b8459a94f54ba00090e5c59936d1':
        raise ValueError('Full archive has not passed official checksum validation')
    for filename,sha in plan['source_sha256'].items():
        file=Path(filename)
        if file.suffix=='.json' and digest(file)!=sha:raise ValueError('Parent annotation changed')
        if file.suffix=='.zip' and (sha!=source['sha256'] or file.stat().st_size!=19021600724):
            raise ValueError('Validated archive identity/length differs')
    base=Dataset('clevr',DATA)
    if base.ids!=screen['row_ids']:raise ValueError('Full original row order differs from audited rows')
    base.exclude(screen['excluded'])
    if base.ids!=plan['row_ids'] or base.labels.tolist()!=plan['labels'] or {k:v.tolist() for k,v in base.splits.items()}!=plan['splits']:
        raise ValueError('Parent row IDs, labels or partitions differ')
    return base,plan


def dataset(root):
    base,parent=checked_parent();ds=Subset(base)
    plan=dict(protocol='CLEVR-light fixed1000train/500validation/2000test; NOT official VTAB1k or full VTAB',
              seed=42,sampler='numpy PCG64 choice without replacement;train,val,test sequential; sorted within each role; labels unused',
              numpy_version=np.__version__,sizes=SIZES,parent_dataset_sha256=digest(PARENT/'dataset.json'),
              parent_overlap_sha256=digest(PARENT/'overlap.json'),selected_parent_positions=ds.selected.tolist(),
              row_ids=ds.ids,labels=ds.labels.tolist(),splits={k:v.tolist() for k,v in ds.splits.items()},
              preprocessing=parent['preprocessing'],evaluation_only=True,light_adapter_sha256=digest(__file__))
    path=root/'dataset.json'
    if path.exists() and json.loads(path.read_text())!=plan:raise ValueError('Fixed lightweight dataset plan changed')
    atomic_json(path,plan)
    return ds


def expected_cache(spec,root):
    return dict(identity(spec),dataset_plan_sha256=digest(root/'dataset.json'),readouts=external_run.kinds(spec),
                adapter_sha256=digest(Path(__file__).with_name('frozen_readouts.py')),
                source_sha256=digest(external_run.__file__),data_adapter_sha256=digest(Path(__file__).with_name('external_data.py')))


def rows_by_id(full_ids, subset_ids):
    if len(set(full_ids))!=len(full_ids) or len(set(subset_ids))!=len(subset_ids):raise ValueError('Duplicate row IDs')
    lookup={v:i for i,v in enumerate(full_ids)}
    try:return np.asarray([lookup[v] for v in subset_ids],dtype=np.int64)
    except KeyError as exc:raise ValueError('Subset row absent from completed full cache') from exc


def slice_complete_cache(spec,ds,root):
    source=PARENT/'features'/spec['name'];meta=source/'cache.json'
    if not meta.exists():return False # Partially written arrays are never used.
    target=root/'features'/spec['name'];target.mkdir(parents=True,exist_ok=True)
    if (target/'cache.json').exists():
        external_run.extract(spec,ds,root,'cpu');return True
    saved=json.loads(meta.read_text());wanted=expected_cache(spec,PARENT)
    if saved['plan']!=wanted:raise ValueError('Completed full encoder cache provenance differs')
    names={k+'.npy' for k in external_run.kinds(spec)}
    if set(saved['files'])!=names:raise ValueError('Unexpected full cache file set')
    parent=json.loads((PARENT/'dataset.json').read_text())
    indices=rows_by_id(parent['row_ids'],ds.ids)
    # Verify all full-array hashes before using any slice, on CPU preparation.
    for name,sha in saved['files'].items():
        if digest(source/name)!=sha:raise ValueError('Completed full cache content differs')
    for name in sorted(names):
        x=np.load(source/name,mmap_mode='r')
        if len(x)!=len(parent['row_ids']) or x.dtype!=np.float32:raise ValueError('Full cache shape/dtype differs')
        selected=np.asarray(x[indices])
        if not np.isfinite(selected).all():raise ValueError('Nonfinite selected full embeddings')
        np.save(target/name,selected)
    atomic_json(target/'cache.json',dict(plan=expected_cache(spec,root),
                files={name:digest(target/name) for name in names},seconds=0.,
                reuse=dict(full_cache=str(meta),full_cache_sha256=digest(meta),full_dataset_sha256=digest(PARENT/'dataset.json'),
                           source_array_sha256=saved['files'],method='exact row-ID lookup and direct FP32 slicing; no inference')))
    external_run.extract(spec,ds,root,'cpu')
    return True


def new_spec(run_name):
    import torch
    from .followup import publish_manifest
    run=Path('experiment_outputs/next-round/training')/run_name
    config=json.loads((Path(__file__).parent/(run_name+'.json')).read_text())
    checkpoint=run/'checkpoints/last.ckpt';state=torch.load(checkpoint,map_location='cpu',weights_only=False)
    if state['hyper_parameters']['config']!=config or state['epoch']+1!=config['epochs']:
        raise ValueError('Lightweight follow-up requires exact completed SSL endpoint')
    spec=dict(name=run_name+'-seed'+str(config['seed']),backbone=config.get('backbone','resnet9'),checkpoint=str(checkpoint))
    publish_manifest(run,spec)
    return [spec]


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('stage',choices=['prepare','extract','cpu'])
    p.add_argument('--output',type=Path,default=ROOT);p.add_argument('--run',choices=RUNS)
    a=p.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):p.error('Run all preparation/inference/probing in Slurm')
    output=a.output
    if a.run and a.output==ROOT:output=Path('experiment_outputs/next-round/external-new')/(a.run+'-seed42-clevr-light')
    root=output/'clevr';root.mkdir(parents=True,exist_ok=True);start=time.monotonic()
    try:
        ds=dataset(root);specs=new_spec(a.run) if a.run else external_run.selected_models()
        atomic_json(root/'models.json',specs)
        reused=[]
        if a.stage=='prepare':
            for spec in specs:
                if slice_complete_cache(spec,ds,root):reused.append(spec['name'])
        elif a.stage=='extract':
            for spec in specs:external_run.extract(spec,ds,root,'cuda')
        else:
            for spec in specs:
                if not (root/'features'/spec['name']/'cache.json').exists():raise ValueError('Missing completed lightweight encoder cache')
                external_run.extract(spec,ds,root,'cpu') # Validation-only existing-cache path.
            external_run.evaluate(root,ds,specs)
        atomic_json(root/(a.stage+'-status.json'),dict(status='complete',job_id=os.environ['SLURM_JOB_ID'],seconds=time.monotonic()-start,
                    images=len(ds),train=1000,validation=500,test=2000,reused_completed_full_caches=reused))
        from .external_report import report
        report(output)
        import subprocess,sys
        subprocess.run([sys.executable,'-u','-m','experiments.lejepa.next_round.report'],check=True)
    except Exception as exc:
        atomic_json(root/(a.stage+'-status.json'),dict(status='failed',job_id=os.environ['SLURM_JOB_ID'],error=str(exc)))
        raise


if __name__=='__main__':main()
