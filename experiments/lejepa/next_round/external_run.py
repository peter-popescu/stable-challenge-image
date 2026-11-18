"""Bounded external frozen evaluation; extract once, CPU probe from caches."""
import argparse
import json
import os
import shutil
import time
from pathlib import Path
import numpy as np
from .external_data import DATA, Dataset
from ..downstream.common import models,load,identity,normalize
from ..sweep import atomic_json,digest

ROOT=Path('experiment_outputs/next-round/external')


def selected_models():
    keep={'original-200-seed42','tuned-200-seed42','tuned-400-seed43','jitter-200-seed42','statpool-200-seed42',
          'resnet18-tuned-200-seed42','random-resnet9','random-resnet18'}
    return [s for s in models(True) if s['name'] in keep]


def kinds(spec):
    native='gap' if spec['backbone']=='resnet18' else ('avgmaxstd' if 'statpool' in spec['name'] else 'max2')
    return list(dict.fromkeys([native,'avgmax','avgmaxstd','all_means']))


def extract(spec,ds,root,device,minimum_free_gib=10.):
    import torch
    from .frozen_readouts import stage_maps,readout_arrays
    from PIL import Image
    target=root/'features'/spec['name'];target.mkdir(parents=True,exist_ok=True)
    expected=dict(identity(spec),dataset_plan_sha256=digest(root/'dataset.json'),
                  readouts=kinds(spec),adapter_sha256=digest(Path(__file__).with_name('frozen_readouts.py')),
                  source_sha256=digest(__file__),data_adapter_sha256=digest(Path(__file__).with_name('external_data.py')))
    meta=target/'cache.json'
    if meta.exists():
        old=json.loads(meta.read_text())
        if old['plan']!=expected: raise ValueError('External cache provenance differs')
        for name,sha in old['files'].items():
            if digest(target/name)!=sha: raise ValueError('External cache content differs')
        return
    # Bound the whole pending model write, retaining an8GiB reserve for ongoing training.
    estimate=len(ds)*len(kinds(spec))*1024*4
    required=max(minimum_free_gib*1024**3,estimate+8*1024**3)
    if shutil.disk_usage(target).free<required:
        raise RuntimeError(f'Insufficient frozen cache storage: require {required/1024**3:.2f}GiB before writes')
    model=load(spec,device);before={k:v.cpu().clone() for k,v in model.state_dict().items()}
    arrays={};t=time.monotonic()
    with torch.inference_mode():
        for start in range(0,len(ds),32):
            imgs=[np.asarray(ds.image(i).resize((96,96),Image.Resampling.BILINEAR),dtype=np.float32)/255 for i in range(start,min(start+32,len(ds)))]
            x=torch.from_numpy(np.stack(imgs)).permute(0,3,1,2).to(device)
            readouts=readout_arrays(stage_maps(model,normalize(x),spec['backbone']),spec['backbone'])
            for kind in kinds(spec):
                v=readouts[kind].cpu().numpy()
                if not np.isfinite(v).all(): raise ValueError('Nonfinite frozen embeddings')
                if kind not in arrays: arrays[kind]=np.lib.format.open_memmap(target/(kind+'.npy'),mode='w+',dtype='float32',shape=(len(ds),v.shape[1]))
                arrays[kind][start:start+len(v)]=v
            if start%3200==0:print(spec['name'],start,'/',len(ds),flush=True)
    for k,v in model.state_dict().items():torch.testing.assert_close(v.cpu(),before[k],rtol=0,atol=0)
    for a in arrays.values():a.flush()
    atomic_json(meta,dict(plan=expected,seconds=time.monotonic()-t,
                         files={k+'.npy':digest(target/(k+'.npy')) for k in arrays},
                         peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated() if device=='cuda' else None))


def evaluate(root,ds,specs):
    from .external_probe import probe,fewshot
    results=root/'results';results.mkdir(exist_ok=True)
    rows=[]
    reps=[]
    for spec in specs:
        for kind in kinds(spec):
            path=root/'features'/spec['name']/(kind+'.npy')
            if path.exists():reps.append((spec['name']+'--'+kind,[path]))
    byname={s['name'] for s in specs}
    # Prespecified scientifically distinct pairs; no external test selection.
    for a,b in [('original-200-seed42','tuned-400-seed43'),('jitter-200-seed42','statpool-200-seed42')]:
        paths=[root/'features'/n/'avgmax.npy' for n in [a,b]]
        if {a,b}<=byname and all(p.exists() for p in paths):reps.append((a+'__'+b,paths))
    splits=getattr(ds,'split_sets',{'official':ds.splits})
    tasks=['count','distance'] if ds.name=='clevr' else [ds.name]
    for name,paths in reps:
        arrays=[np.load(p,mmap_mode='r') for p in paths]
        x=arrays[0] if len(arrays)==1 else np.concatenate(arrays,1)
        for ti,task in enumerate(tasks):
            y=ds.labels[:,ti] if ds.name=='clevr' else ds.labels
            for split_id,ss in splits.items():
                stem=f'{name}--{task}--{split_id}';dst=results/(stem+'.json')
                plan=dict(features={str(p):digest(p) for p in paths},dataset_plan_sha256=digest(root/'dataset.json'),
                          probe_sha256=digest(Path(__file__).with_name('external_probe.py')),task=task,split=split_id)
                if dst.exists():
                    r=json.loads(dst.read_text())
                    if r['plan']!=plan:raise ValueError('External probe identity differs')
                else:
                    print('Probe',stem,flush=True);t=time.monotonic()
                    r=dict(plan=plan,representation=name,dimensions=x.shape[1],task=task,split=split_id,
                           **probe(x,y,ss,results/(stem+'.npz')))
                    # Few-shot protocol is bounded to native and stat readouts of
                    # trained encoders plus random controls; no extra SSL variants.
                    if ds.name=='cifar100' and len(paths)==1 and name.endswith(('--max2','--gap','--avgmaxstd')):
                        out=results/(stem+'-fewshot');out.mkdir(exist_ok=True)
                        r['fewshot']=fewshot(x,y,ss,out)
                    r['seconds']=time.monotonic()-t;atomic_json(dst,r)
                rows.append(dict(representation=name,dimensions=r['dimensions'],task=task,split=split_id,
                                 validation_accuracy=r['validation']['accuracy'],test_accuracy=r['test']['accuracy'],
                                 test_macro_accuracy=r['test']['macro_accuracy']))
                atomic_json(root/'matrix.json',rows)


def overlap_screen(ds,root):
    """Exact decoded RGB overlaps only; near-duplicate risk remains."""
    import hashlib
    from training_data import load_training_data
    from PIL import Image
    def image_hashes(im):
        im=im.convert('RGB')
        versions=[im,im.resize((96,96),Image.Resampling.BILINEAR)]
        return [hashlib.sha256(str(v.size).encode()+v.tobytes()).hexdigest() for v in versions]
    path=root/'overlap.json'
    shared=ROOT/ds.name/'overlap.json'
    if not path.exists() and shared.exists() and shared!=path:
        shutil.copyfile(shared,path)
    source=digest('data/train.parquet')
    adapter=digest(Path(__file__).with_name('external_data.py'))
    method='SHA256(decoded RGB size+bytes),original and bilinear96px; exact only;v2'
    if path.exists():
        saved=json.loads(path.read_text())
        if saved['challenge_sha256']!=source or saved['row_ids']!=ds.ids or saved.get('method')!=method or saved.get('data_adapter_sha256')!=adapter:
            raise ValueError('Overlap cache identity changed')
    else:
        challenge=load_training_data('data/train.parquet')
        hashes={v for row in challenge for v in image_hashes(row['image'])}
        excluded=[];h=hashlib.sha256()
        for i in range(len(ds)):
            fingerprints=image_hashes(ds.image(i))
            for fingerprint in fingerprints:h.update(fingerprint.encode())
            if any(fingerprint in hashes for fingerprint in fingerprints):excluded.append(i)
        saved=dict(challenge_sha256=source,row_ids=ds.ids,excluded=excluded,
                   canonical_image_sequence_sha256=h.hexdigest(),method=method,data_adapter_sha256=adapter)
        atomic_json(path,saved)
    ds.exclude(saved['excluded'])


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('stage',choices=['extract','cpu','audit'])
    p.add_argument('--dataset',required=True,choices=['dtd','cifar100','clevr'])
    p.add_argument('--data-root',type=Path,default=DATA)
    p.add_argument('--output',type=Path,default=ROOT)
    p.add_argument('--manifest',type=Path)
    p.add_argument('--minimum-free-gib',type=float,default=10.)
    a=p.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):p.error('Run preprocessing/extraction/probes in Slurm')
    root=a.output/a.dataset;root.mkdir(parents=True,exist_ok=True)
    try:
        ds=Dataset(a.dataset,a.data_root);overlap_screen(ds,root);plan=ds.plan()
        plan['overlap_sha256']=digest(root/'overlap.json')
        path=root/'dataset.json'
        if path.exists() and json.loads(path.read_text())!=plan:raise ValueError('External dataset identity changed')
        atomic_json(path,plan)
        specs=json.loads(a.manifest.read_text()) if a.manifest else selected_models()
        atomic_json(root/'models.json',specs)
        if a.stage=='extract':
            for spec in specs:extract(spec,ds,root,'cuda',a.minimum_free_gib)
        elif a.stage=='cpu':evaluate(root,ds,specs)
        atomic_json(root/(a.stage+'-status.json'),dict(status='complete',job_id=os.environ['SLURM_JOB_ID']))
        from .external_report import report
        report(a.output)
    except Exception as exc:
        atomic_json(root/(a.stage+'-status.json'),dict(status='failed',error=str(exc),job_id=os.environ['SLURM_JOB_ID']))
        raise


if __name__=='__main__':main()
