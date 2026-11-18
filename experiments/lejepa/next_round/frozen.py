"""Slurm-only frozen multiscale, concatenation, complementarity and ONNX study."""
import argparse,json,os,time,resource,subprocess,sys
from pathlib import Path
import numpy as np
from ..sweep import atomic_json,digest
from ..downstream.common import models,load,normalize,rgb_images,identity
from .frozen_readouts import stage_maps,readout_arrays,MultiReadout,EnsembleReadout

ROOT=Path('experiment_outputs/next-round/frozen')

def selected():
    return [s for s in models(True) if s['name']!='crop-200-seed42'] + [dict(name='tuned-200-seed'+str(seed),backbone='resnet9',checkpoint=f'experiment_outputs/seed-repeats/baseline-vs-tuned/winner-seed-{seed}/checkpoints/last.ckpt') for seed in [43,44]]


def pairs():
    a='original-200-seed42';b='tuned-400-seed43';c='jitter-200-seed42';d='statpool-200-seed42'
    return [(a,b),(a,c),(a,d),(c,b),(c,d),('tuned-200-seed43','tuned-200-seed44'),('tuned-200-seed42','tuned-200-seed42')]


def complementarity(y,a,b):
    ca=a==y;cb=b==y
    def scores(mask):
        return dict(examples=int(mask.sum()),agreement=float(np.mean(a[mask]==b[mask])),disagreement=float(np.mean(a[mask]!=b[mask])),both_correct=float(np.mean((ca&cb)[mask])),a_only_correct=float(np.mean((ca&~cb)[mask])),b_only_correct=float(np.mean((~ca&cb)[mask])),only_one_correct=float(np.mean((ca^cb)[mask])),both_wrong=float(np.mean((~ca&~cb)[mask])),oracle_accuracy=float(np.mean((ca|cb)[mask])),a_accuracy=float(ca[mask].mean()),b_accuracy=float(cb[mask].mean()),a_minus_b_accuracy=float(ca[mask].mean()-cb[mask].mean()))
    return dict(scores(np.ones(len(y),bool)),per_class=[dict(class_id=int(c),**scores(y==c)) for c in np.unique(y)],interpretation='Oracle is diagnostic; not realizable ensemble accuracy')


def extract(spec,ds,ids,out):
    import torch
    path=out/spec['name'];path.mkdir(parents=True,exist_ok=True)
    prior=Path('experiment_outputs/green-area/downstream')/spec['name']/'cache'
    plan=dict(identity(spec),row_ids=ids.tolist(),data_sha256=digest('data/train.parquet'),source_sha256=digest(Path(__file__).with_name('frozen_readouts.py')))
    meta=path/'cache.json'
    if meta.exists():
        saved=json.loads(meta.read_text());assert saved['plan']==plan
        assert digest(path/'embeddings.npz')==saved['sha256'];return
    model=load(spec,'cuda');before={k:v.cpu().clone() for k,v in model.state_dict().items()};zs={}
    # Reuse verified cached mid/final maps; only early stages need new extraction.
    cached=None
    if (prior/'cache.json').exists():
        old=json.loads((prior/'cache.json').read_text())
        assert old['plan']['checkpoint_sha256']==plan['checkpoint_sha256']
        assert old['plan']['row_ids']==ids.tolist() and old['plan']['data_sha256']==plan['data_sha256']
        for f in ('mid.npy','final.npy'):assert digest(prior/f)==old['files'][f]
        cached={k:np.load(prior/f,mmap_mode='r') for k,f in [('stage3','mid.npy'),('stage4','final.npy')]}
    start=time.perf_counter()
    with torch.inference_mode():
        for i in range(0,len(ids),16):
            x=normalize(rgb_images(ds,ids[i:i+16]).cuda());maps={}
            if cached:
                if spec['backbone']=='resnet18':
                    x=model.stem(x)
                    for j in range(2):x=model.stages[j](x);maps['stage'+str(j+1)]=x.cpu()
                else:
                    for j,layer in enumerate(model.conv):
                        if j>7:break
                        x=layer(x)
                        if j in (2,7):maps['stage'+str((2,7).index(j)+1)]=x.cpu()
                maps.update({k:torch.from_numpy(np.array(v[i:i+16])) for k,v in cached.items()})
            else:maps={k:v.cpu() for k,v in stage_maps(model,x,spec['backbone']).items()}
            for k,z in readout_arrays(maps,spec['backbone']).items():zs.setdefault(k,[]).append(z.numpy())
    features={k:np.concatenate(v) for k,v in zs.items()}
    if cached:
        historical=np.load(prior/'embeddings.npz')
        np.testing.assert_array_equal(historical['row_ids'],ids)
        for kind in set(features)&set(historical.files):np.testing.assert_allclose(features[kind],historical[kind],rtol=1e-6,atol=1e-6)
    np.savez(path/'embeddings.npz',row_ids=ids,**features)
    for k,v in model.state_dict().items():torch.testing.assert_close(v.cpu(),before[k],rtol=0,atol=0)
    atomic_json(meta,dict(plan=plan,sha256=digest(path/'embeddings.npz'),reused_mid_final=bool(cached),seconds=time.perf_counter()-start,peak_gpu_bytes=torch.cuda.max_memory_allocated()))


def evaluate_one(name,z,y,ids,folds,ds,out):
    from ..cross_validate import probe_embeddings
    from ..downstream.semantic import evaluate,episodes
    from ..downstream.appearance import probe,targets,split_images
    p=out/'conditions'/name;p.mkdir(parents=True,exist_ok=True)
    plan=dict(feature_sha256=__import__('hashlib').sha256(z.tobytes()).hexdigest(),row_ids=ids.tolist(),folds=folds.tolist(),labels=y.tolist(),sources={str(f):digest(f) for f in [Path(__file__),Path('evaluate.py'),Path('experiments/lejepa/downstream/semantic.py'),Path('experiments/lejepa/downstream/appearance.py'),Path('experiments/lejepa/cross_validate.py')]})
    if (p/'results.json').exists():
        assert json.loads((p/'plan.json').read_text())==plan;return json.loads((p/'results.json').read_text())
    atomic_json(p/'plan.json',plan)
    m,f,pred,loo,cm=probe_embeddings(np.pad(z,((0,0),(0,1024-z.shape[1]))),y,folds)
    sem=evaluate(z,y,episodes(y));sem['classification']=m
    app=probe(z,targets(rgb_images(ds,ids)),split_images(len(ids)))
    result=dict(name=name,dimensions=z.shape[1],semantic=sem,appearance=app,fold_metrics=f)
    np.savez(p/'predictions.npz',pred=pred,loo=loo,confusion=cm,folds=folds,row_ids=ids,labels=y)
    atomic_json(p/'results.json',result);print(name,m['accuracy'],m['official_local_loo_accuracy'],flush=True)
    return result


def report(out):
    rows=[]
    for p in sorted((out/'conditions').glob('*/results.json')):
        r=json.loads(p.read_text());s=r['semantic'];c=s['classification']
        rows.append(dict(name=r['name'],dimensions=r['dimensions'],accuracy=c['accuracy'],macro_accuracy=c['macro_accuracy'],loo=c['official_local_loo_accuracy'],fewshot5=s['fewshot']['5']['accuracy']['mean'],retrieval=s['retrieval_map'],knn5=s['knn']['5'],ari=s['clustering_ari'],appearance_r2=r['appearance']['test']['r2']))
    atomic_json(out/'matrix.json',rows)
    from ..error_analysis import write_csv
    if rows:write_csv(out/'matrix.csv',rows)
    lines=['# Frozen multiscale and multi-encoder study','','Development scores; no hidden leaderboard gains inferred. Same fixed outer folds and official standardized ridge selection. No block weighting or learned dimensionality reduction.','','|Condition|Dim|Five-fold|Macro|LOO|5-shot|Retrieval|RGB R²|','|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in rows:lines.append(f"|{r['name']}|{r['dimensions']}|{r['accuracy']:.2%}|{r['macro_accuracy']:.2%}|{r['loo']:.2%}|{r['fewshot5']:.2%}|{r['retrieval']:.4f}|{r['appearance_r2']:.4f}|")
    lines+=['','ResNet9 stages: stem64 after conv index2; residual128 after index7; intermediate256 after index11; final256 after index16. All-stage means=704; last-two mean+max=1024; intermediate mean+final mean+max=768; intermediate mean+final statistics=1024. ResNet18 uses four residual stages40/80/160/320.','Duplicating tuned200 uses exactly the same embedding twice. Standardization and ridge selection remain unchanged, so its effects can include effective regularization. Complementarity uses identical held-out folds; oracle accuracy is diagnostic only.','Existing segmentation, local color and transformation-specific correspondence remain in experiment_outputs/green-area/downstream and are shared across frozen readouts. No spatial probe was repeated.']
    cp=out/'complementarity.json'
    if cp.exists():
        pairs_data=json.loads(cp.read_text())
        lines+=['','|Pair|Disagreement|Both correct|A only|B only|Both wrong|Oracle|Concat gain pp|','|---|---:|---:|---:|---:|---:|---:|---:|']
        for q in pairs_data:lines.append(f"|{q['a']} + {q['b']}|{q['disagreement']:.2%}|{q['both_correct']:.2%}|{q['a_only_correct']:.2%}|{q['b_only_correct']:.2%}|{q['both_wrong']:.2%}|{q['oracle_accuracy']:.2%}|{100*q['gain_over_best_component']:+.2f}|")
        unique=[q for q in pairs_data if q['a']!=q['b']]
        if len(unique)>2 and np.std([q['disagreement'] for q in unique])>0 and np.std([q['gain_over_best_component'] for q in unique])>0:
            correlation=float(np.corrcoef([q['disagreement'] for q in unique],[q['gain_over_best_component'] for q in unique])[0,1])
            lines+=['',f'Descriptive Pearson disagreement/concat-gain correlation across {len(unique)} distinct pairs: {correlation:.3f}. Pairs share encoders and are not independent replicates; this is not a predictive or causal estimate. Per-class and per-fold partitions are retained in complementarity.json.']
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n')


def cpu(specs,ds,ids,y,out):
    from ..cross_validate import fixed_plan
    folds=fixed_plan(Path('experiment_outputs/validation/stratified-5fold-seed42.json'),ids,y,digest('data/train.parquet'))
    allz={};results={}
    for spec in specs:
        cache=np.load(out/spec['name']/'embeddings.npz');np.testing.assert_array_equal(cache['row_ids'],ids)
        prior=Path('experiment_outputs/green-area/downstream')/spec['name']/'cache'/'embeddings.npz'
        if prior.exists():
            historical=np.load(prior);np.testing.assert_array_equal(historical['row_ids'],ids)
            for kind in set(cache.files)&set(historical.files)-{'row_ids'}:np.testing.assert_allclose(cache[kind],historical[kind],rtol=1e-6,atol=1e-6)
        for kind in cache.files:
            if kind=='row_ids':continue
            name=spec['name']+'__'+kind;allz[name]=cache[kind]
            results[name]=evaluate_one(name,cache[kind],y,ids,folds,ds,out);report(out)
    comp=[]
    for a,b in pairs():
        ka=a+'__avgmax';kb=b+'__avgmax'
        if ka not in allz or kb not in allz:continue
        name=a+'++'+b
        pa=np.load(out/'conditions'/ka/'predictions.npz')['pred'];pb=np.load(out/'conditions'/kb/'predictions.npz')['pred']
        r=complementarity(y,pa,pb);r.update(a=a,b=b,per_fold=[dict(fold=int(f),**complementarity(y[folds==f],pa[folds==f],pb[folds==f])) for f in np.unique(folds)])
        z=np.concatenate([allz[ka],allz[kb]],1)
        e=evaluate_one(name,z,y,ids,folds,ds,out)
        r.update(ensemble_accuracy=e['semantic']['classification']['accuracy'],gain_over_best_component=e['semantic']['classification']['accuracy']-max(results[ka]['semantic']['classification']['accuracy'],results[kb]['semantic']['classification']['accuracy']))
        comp.append(r);atomic_json(out/'complementarity.json',comp);report(out)


def export_candidates(specs,ds,ids,out):
    import torch,onnxruntime as ort
    from ..export import export_encoder
    lookup={s['name']:s for s in specs};matrix=json.loads((out/'matrix.json').read_text())
    ranked=sorted([r for r in matrix if '++' in r['name'] and len(set(r['name'].split('++')))==2],key=lambda r:r['accuracy'],reverse=True)
    distinct=next((r for r in ranked if r['name']=='original-200-seed42++statpool-200-seed42'),None)
    ensembles=ranked[:2]+([distinct] if distinct is not None and distinct not in ranked[:2] else ranked[2:3])
    singles=[]
    for backbone in ['resnet9','resnet18']:
        names={s['name'] for s in specs if s['backbone']==backbone and s.get('checkpoint')}
        singles.append(max([r for r in matrix if '__' in r['name'] and r['name'].split('__')[0] in names],key=lambda r:r['accuracy']))
    exported=[];directory=out/'exports';directory.mkdir(exist_ok=True)
    x=normalize(rgb_images(ds,ids[:4])).numpy()
    for r in ensembles+singles:
        name=r['name'];path=directory/(name+'.onnx')
        if '++' in name:
            a,b=name.split('++');model=EnsembleReadout(load(lookup[a]),load(lookup[b]));z=np.concatenate([np.load(out/a/'embeddings.npz')['avgmax'][:4],np.load(out/b/'embeddings.npz')['avgmax'][:4]],1)
        else:
            a,k=name.split('__');model=MultiReadout(load(lookup[a]),lookup[a]['backbone'],k);z=np.load(out/a/'embeddings.npz')[k][:4]
        export_encoder(model,path)
        options=ort.SessionOptions();options.intra_op_num_threads=1;options.inter_op_num_threads=1
        session=ort.InferenceSession(str(path),options,providers=['CPUExecutionProvider']);actual=session.run(None,{'image':x})[0]
        assert actual.dtype==np.float32 and np.isfinite(actual).all()
        np.testing.assert_allclose(actual,np.pad(z,((0,0),(0,1024-z.shape[1]))),rtol=1e-3,atol=1e-4)
        for batch in (1,4):
            for _ in range(3):session.run(None,{'image':x[:batch]})
            times=[]
            for _ in range(20):
                start=time.perf_counter();session.run(None,{'image':x[:batch]});times.append(time.perf_counter()-start)
            r['latency_ms_batch'+str(batch)]=1000*float(np.median(times))
        with (directory/(name+'-official.log')).open('w') as f:subprocess.run([sys.executable,'-m','experiments.lejepa.next_round.frozen_validate',str(path)],stdout=f,stderr=subprocess.STDOUT,check=True)
        r['fresh_process_cpu']=json.loads(subprocess.check_output([sys.executable,'-m','experiments.lejepa.next_round.frozen_benchmark',str(path)],text=True))
        r.update(path=str(path),sha256=digest(path),size_bytes=path.stat().st_size,parameters=sum(p.numel() for p in model.parameters()),peak_process_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,memory_scope='Cumulative exporter/evaluation process peak; fresh-process inference RSS recorded separately',official_checks=True,cached_parity=True)
        exported.append(r);atomic_json(directory/'manifest.json',exported)


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('stage',choices=['extract','cpu','export']);p.add_argument('--output',type=Path,default=ROOT);p.add_argument('--manifest',type=Path);p.add_argument('--only');a=p.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):p.error('Slurm required')
    import torch
    torch.set_num_threads(int(os.environ.get('SLURM_CPUS_PER_TASK','4')))
    from training_data import load_training_data
    ds=load_training_data('data/train.parquet');labels=np.asarray(ds['label']);ids=np.flatnonzero(labels>=0);y=labels[ids]
    specs=json.loads(a.manifest.read_text()) if a.manifest else selected()
    if a.only:specs=[s for s in specs if s['name'] in a.only.split(',')]
    suffix='-'+a.only if a.only else ''
    a.output.mkdir(parents=True,exist_ok=True)
    atomic_json(a.output/(a.stage+suffix+'-plan.json'),dict(job_id=os.environ['SLURM_JOB_ID'],models=[identity(s) for s in specs],sources={str(f):digest(f) for f in Path(__file__).parent.glob('frozen*.py')}))
    if a.stage=='extract':
        for spec in specs:print('extract',spec['name'],flush=True);extract(spec,ds,ids,a.output)
    elif a.stage=='cpu':cpu(specs,ds,ids,y,a.output)
    else:export_candidates(specs,ds,ids,a.output)
    atomic_json(a.output/(a.stage+suffix+'-status.json'),dict(status='completed',job_id=os.environ['SLURM_JOB_ID']))
if __name__=='__main__':main()
