"""Slurm evaluation. Use extract-global/cpu-global for pooled-only global tasks.

Dense extraction (extract/cpu) is opt-in and persists spatial maps for local probes.
"""
import argparse
import json
import os
from pathlib import Path
import time
import traceback
import numpy as np
from .common import ROOT,models,identity
from ..sweep import atomic_json,digest


def run_one(stage,spec,ds,ids,labels,output):
    import torch
    from .common import cache_features,cache_global_features,validate_global_cache,load,rgb_images,layer_info
    directory=output/spec['name'];directory.mkdir(parents=True,exist_ok=True)
    global_only=stage in ('extract-global','cpu-global')
    cache=directory/('global-cache' if global_only else 'cache')
    if stage=='extract-global':
        cache_global_features(spec,ds,ids,digest('data/train.parquet'),cache)
    elif stage=='extract':
        cache_features(spec,ds,ids,digest('data/train.parquet'),cache)
        from .correspondence import evaluate
        # 32 fixed images shared across all checkpoints; 192px to allow strict
        # RF-valid final ResNet18 descriptors (RF109 exceeds 96px).
        count=min(32,len(ids)); chosen=np.random.default_rng(42).choice(len(ids),count,replace=False)
        model=load(spec,'cuda')
        before={k:v.cpu().clone() for k,v in model.state_dict().items()}
        result=evaluate(model,spec,rgb_images(ds,ids[chosen],size=192),directory/'correspondence')
        for k,v in model.state_dict().items():torch.testing.assert_close(v.cpu(),before[k],rtol=0,atol=0)
        atomic_json(directory/'correspondence.json',result)
    elif stage in ('cpu','appearance','cpu-global'):
        from .semantic import episodes,evaluate
        from .appearance import targets,split_images,probe,metrics
        from ..cross_validate import fixed_plan,probe_embeddings
        splits=episodes(labels)
        episode_manifest=dict(row_ids=ids.tolist(),episodes=splits)
        path=output/'episodes.json'
        if path.exists() and json.loads(path.read_text())!=episode_manifest:raise ValueError('Episode manifest changed')
        atomic_json(path,episode_manifest)
        folds=fixed_plan(Path('experiment_outputs/validation/stratified-5fold-seed42.json'),ids,labels,digest('data/train.parquet'))
        if global_only:validate_global_cache(spec,ids,digest('data/train.parquet'),cache)
        features=np.load(cache/'embeddings.npz');np.testing.assert_array_equal(features['row_ids'],ids)
        semantic={};appearance={};rgb=rgb_images(ds,ids);global_y=targets(rgb);parts=split_images(len(ids))
        atomic_json(output/'appearance_splits.json',dict(row_ids=ids.tolist(),train=parts[0].tolist(),validation=parts[1].tolist(),test=parts[2].tolist()))
        for kind in features.files:
            if kind=='row_ids':continue
            z=features[kind]
            if stage in ('cpu','cpu-global'):
                semantic[kind]=evaluate(z,labels,splits)
                # Reuse completed official classifications when checkpoint+readout
                # were already measured; otherwise use the existing official probe.
                previous=Path('experiment_outputs/backbone-pooling/frozen-readouts')/spec['name']/'results.json'
                if spec['backbone']=='resnet18':previous=ROOT/'resnet18-pooling'/spec['name']/'results.json'
                if previous.exists() and kind in json.loads(previous.read_text())['readouts']:
                    r=json.loads(previous.read_text())
                    if r['checkpoint']['checkpoint_sha256']!=identity(spec)['checkpoint_sha256']:raise ValueError('Historical checkpoint changed')
                    metadata=json.loads((previous.parent.parent/'plan.json').read_text())
                    if metadata['data_sha256']!=digest('data/train.parquet') or metadata['fold_manifest_sha256']!=digest('experiment_outputs/validation/stratified-5fold-seed42.json'):raise ValueError('Historical data/folds changed')
                    semantic[kind]['classification']=r['readouts'][kind]['metrics']
                else:
                    padded=np.pad(z,((0,0),(0,max(0,1024-z.shape[1]))))
                    m,fold_metrics,pred,loo,cm=probe_embeddings(padded,labels,folds)
                    semantic[kind]['classification']=m
                    np.savez(directory/f'{kind}-classification.npz',pred=pred,loo=loo,folds=folds,confusion=cm)
            appearance[kind]=probe(z,global_y,parts)
        for layer,info in ([] if global_only else layer_info(spec).items()):
            x=np.load(cache/(layer+'.npy'),mmap_mode='r').transpose(0,2,3,1)
            y=targets(rgb,info)
            appearance['spatial-'+layer]=probe(x,y,parts)
            # Nonlearned per-image global color: copy it to every local target.
            baseline=np.broadcast_to(global_y[parts[2],None,None,:],y[parts[2]].shape)
            appearance['spatial-'+layer]['image_global_oracle']=metrics(y[parts[2]].reshape(-1,6),baseline.reshape(-1,6))
        if stage in ('cpu','cpu-global'):atomic_json(directory/('semantic-global.json' if global_only else 'semantic.json'),semantic)
        previous=directory/'appearance.json'
        if stage=='appearance' and previous.exists():
            archived=directory/'appearance-float32-attempt199251.json'
            if not archived.exists():previous.rename(archived)
        atomic_json(directory/('appearance-global.json' if global_only else 'appearance.json'),appearance)
    elif stage=='segmentation':
        from .segmentation import run
        run(spec)


def report_global(output):
    """Summarize global endpoints without touching completed dense reports."""
    from ..error_analysis import write_csv
    rows=[]
    for path in sorted(output.glob('*/semantic-global.json')):
        appearance_path=path.parent/'appearance-global.json'
        if not appearance_path.exists():continue
        appearance=json.loads(appearance_path.read_text())
        for kind,result in json.loads(path.read_text()).items():
            c=result['classification'];a=appearance[kind]['test']
            rows.append(dict(model=path.parent.name,readout=kind,dimensions=result['dimensions'],
                accuracy=c['accuracy'],macro_accuracy=c['macro_accuracy'],loo=c['official_local_loo_accuracy'],
                fewshot5=result['fewshot']['5']['accuracy']['mean'],retrieval_map=result['retrieval_map'],
                knn5=result['knn']['5'],clustering_ari=result['clustering_ari'],clustering_nmi=result['clustering_nmi'],
                global_rgb_r2=a['r2'],global_rgb_mse=a['mse']))
    atomic_json(output/'global_performance_matrix.json',rows)
    if rows:write_csv(output/'global_performance_matrix.csv',rows)
    lines=['# Global-only frozen evaluation','','Caches contain pooled embeddings only. No spatial maps, correspondence or local RGB probes were produced by these stages. Existing dense endpoints remain separate. Development metrics do not establish hidden leaderboard gains.','','|Model/readout|Dim|Five-fold|Macro|LOO|5-shot|Retrieval|RGB R²|','|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in rows:
        lines.append(f"|{r['model']}/{r['readout']}|{r['dimensions']}|{r['accuracy']:.2%}|{r['macro_accuracy']:.2%}|{r['loo']:.2%}|{r['fewshot5']:.2%}|{r['retrieval_map']:.4f}|{r['global_rgb_r2']:.4f}|")
    if not rows:lines+=['','Global CPU endpoints pending.']
    (output/'GLOBAL_REPORT.md').write_text('\n'.join(lines)+'\n')


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('stage',choices=['extract-global','cpu-global','extract','cpu','appearance','segmentation'])
    p.add_argument('--manifest',type=Path);p.add_argument('--include-new',action='store_true');p.add_argument('--only')
    p.add_argument('--output',type=Path,default=ROOT/'downstream');a=p.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):p.error('Run in Slurm')
    import torch
    torch.set_num_threads(int(os.environ.get('SLURM_CPUS_PER_TASK','4')))
    selected=json.loads(a.manifest.read_text()) if a.manifest else models(a.include_new)
    if a.only:selected=[s for s in selected if s['name'] in a.only.split(',')]
    a.output.mkdir(parents=True,exist_ok=True)
    # All stages bind to checkpoint, task source and dataset identities.
    plan=dict(models=[identity(s) for s in selected],data_sha256=digest('data/train.parquet'),
              sources={str(f):digest(f) for f in list(Path(__file__).parent.glob('*.py'))+[Path(__file__).parents[1]/f for f in ['model.py','resnet18.py','readouts.py','cross_validate.py']]})
    plan_path=a.output/f'{a.stage}-{a.only or "all"}-plan.json'
    if plan_path.exists() and json.loads(plan_path.read_text())!=plan:raise ValueError('Evaluation plan changed; choose a new output')
    atomic_json(plan_path,plan)
    ds=None;ids=None;labels=None
    if a.stage!='segmentation':
        from training_data import load_training_data
        ds=load_training_data('data/train.parquet');all_y=np.asarray(ds['label']);ids=np.flatnonzero(all_y>=0);labels=all_y[ids]
    statuses={}
    for spec in selected:
        start=time.time();print(a.stage,spec['name'],flush=True)
        if torch.cuda.is_available():torch.cuda.reset_peak_memory_stats()
        try:
            run_one(a.stage,spec,ds,ids,labels,a.output)
            statuses[spec['name']]=dict(status='completed',seconds=time.time()-start,host=os.uname().nodename,cpus=torch.get_num_threads(),
                gpu=torch.cuda.get_device_name() if torch.cuda.is_available() else None,
                peak_gpu_allocated_gib=torch.cuda.max_memory_allocated()/1024**3 if torch.cuda.is_available() else None,
                peak_gpu_reserved_gib=torch.cuda.max_memory_reserved()/1024**3 if torch.cuda.is_available() else None)
        except Exception as error:
            traceback.print_exc();statuses[spec['name']]=dict(status='failed',error=repr(error),seconds=time.time()-start)
        atomic_json(a.output/f'{a.stage}-{a.only or "all"}-status.json',dict(job_id=os.environ['SLURM_JOB_ID'],models=statuses))
        try:
            if a.stage in ('extract-global','cpu-global'):report_global(a.output)
            else:
                from .report import report
                report(a.output)
        except Exception:
            traceback.print_exc()
    if any(v['status']=='failed' for v in statuses.values()):raise SystemExit(1)


if __name__=='__main__':main()
