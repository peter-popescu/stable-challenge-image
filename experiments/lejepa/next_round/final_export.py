"""Challenge-only final selection, cached parity, official checks and CPU audit."""
import json,os,subprocess,sys,time
from pathlib import Path
from ..sweep import atomic_json,digest

BASE=Path('experiment_outputs/next-round')
OUT=BASE/'final-frozen'
METRICS=('accuracy','macro_accuracy','loo','fewshot5','retrieval','knn5','ari','appearance_r2')


def collect():
    rows=[]
    for path in [BASE/'frozen/matrix.json',*sorted((BASE/'frozen-new').glob('*/matrix.json'))]:
        for row in json.loads(path.read_text()):rows.append(dict(row,source_matrix=str(path)))
    return rows


def best(rows,key='accuracy'):
    secondary='loo' if key=='accuracy' else 'accuracy'
    return max(rows,key=lambda r:(r[key],r[secondary],r['name']))


def compare(a,b,label):
    return dict(comparison=label,before=a['name'],after=b['name'],before_metrics={k:a[k] for k in METRICS},
                after_metrics={k:b[k] for k in METRICS},delta={k:b[k]-a[k] for k in METRICS},
                delta_accuracy_pp=100*(b['accuracy']-a['accuracy']),delta_loo_pp=100*(b['loo']-a['loo']))


def summarize(rows):
    lookup={r['name']:r for r in rows};matched=[];multiscale=[];ensemble=[]
    for label,a,b in [('original recipe, seed42; planned200 vs400 cosine','original-200-seed42','resnet9-original-400-seed42'),
                      ('tuned ResNet18, seed42; planned200 vs400 cosine','resnet18-tuned-200-seed42','resnet18-tuned-400-seed42'),
                      ('tuned ResNet9, seed42; two vs six local views, broader recipe','tuned-200-seed42','resnet9-more-views-200-seed42')]:
        for old in rows:
            if old['name'].startswith(a+'__'):
                name=b+'__'+old['name'].split('__',1)[1]
                if name in lookup:matched.append(compare(old,lookup[name],label))
    models=sorted({r['name'].split('__',1)[0] for r in rows if '++' not in r['name']})
    for name in models:
        stats=lookup.get(name+'__avgmaxstd')
        if not stats:continue
        for r in rows:
            if r['name'].startswith(name+'__') and r['name'].split('__',1)[1] in ('all_means','last_two_avgmax','all_means_final_max','mid_mean_final_avgmax','mid_mean_final_avgmaxstd'):
                multiscale.append(compare(stats,r,'multiscale versus final mean+max+std'))
    r18=[r for r in rows if r['name'].startswith(('resnet18-tuned-200-seed42__','resnet18-tuned-400-seed42__'))]
    strongest18=best(r18)
    for row in rows:
        if '++' not in row['name']:continue
        a,b=row['name'].split('++');components=[];strong=[]
        for component in (a,b):
            components.append(compare(lookup[component+'__avgmax'],row,'concatenation versus component512'))
            strongest=best([r for r in rows if r['name'].startswith(component+'__')])
            strong.append(compare(strongest,row,'concatenation versus component best challenge readout'))
        ensemble.append(dict(name=row['name'],components512=components,component_best=strong,
                             versus_strongest_r18=compare(strongest18,row,'ensemble versus strongest near-budget ResNet18'),
                             duplicate_control=a==b))
    summary=dict(selection_basis='Challenge fixed outer five-fold and official local LOO only. No external endpoint enters selection.',
                 matched_readout_comparisons=matched,multiscale_vs_final_statistics=multiscale,ensemble_comparisons=ensemble,
                 strongest_r18=strongest18,protocol_risks=['200 and400 endpoints have separately planned cosine horizons; duration and LR histories change together.',
                    'More views changes global/local invariance weighting, SIGReg sample count, BatchNorm exposure and compute together.',
                    'Historical tuned400/43 versus tuned200/42 remains seed-confounded; matched readouts within a checkpoint do not remove that confound.',
                    'Repeated challenge readout selection uses development data; no hidden score or general transfer gain is implied.',
                    'Oracle complementarity is diagnostic; probes and feature standardization are unchanged.'])
    atomic_json(OUT/'matrix.json',rows);atomic_json(OUT/'summary.json',summary)
    from ..error_analysis import write_csv
    write_csv(OUT/'matrix.csv',rows)
    lines=['# Final frozen challenge profile','','Metrics remain in their native units; no task-aggregate score. Candidate selection uses challenge development five-fold/LOO only. No leaderboard submissions.','','|Model/readout|Dim|Five-fold|Macro|LOO|5-shot|Retrieval|RGB R²|','|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in rows:lines.append(f"|{r['name']}|{r['dimensions']}|{r['accuracy']:.2%}|{r['macro_accuracy']:.2%}|{r['loo']:.2%}|{r['fewshot5']:.2%}|{r['retrieval']:.4f}|{r['appearance_r2']:.4f}|")
    lines+=['','Matched native changes (full matched-readout table in summary.json):','','|Comparison|Native readout|Five-fold change pp|LOO change pp|Retrieval change|','|---|---|---:|---:|---:|']
    for r in matched:
        if r['before'].endswith('__max2') or r['before'].startswith('resnet18') and r['before'].endswith('__gap'):
            lines.append(f"|{r['comparison']}|{r['before'].split('__')[1]}|{r['delta_accuracy_pp']:+.2f}|{r['delta_loo_pp']:+.2f}|{r['delta']['retrieval']:+.4f}|")
    lines+=['','The same seed and recipe endpoints compare completed planned200 versus400 schedules; the epoch200 milestone of a400 schedule is a different learning-rate history. All local scores remain development results. Multiscale and ensemble task profiles, component controls and near-budget ResNet18 comparisons are in summary.json; per-class/fold complementarity stays in frozen/complementarity.json.']
    (OUT/'REPORT.md').write_text('\n'.join(lines)+'\n')
    return summary


def choose(rows):
    new=[r for r in rows if '/frozen-new/' in r['source_matrix'] and '++' not in r['name']]
    primary=best(new)
    # A prespecified final-layer statistical alternative distinguishes readout
    # mechanisms; choose it by challenge LOO then five-fold, never external tasks.
    stats=[r for r in new if r['name'].endswith('__avgmaxstd') and r['name']!=primary['name']]
    selected=[primary]+([best(stats,'loo')] if stats else [])
    atomic_json(OUT/'selection.json',dict(rule='Primary=max new five-fold then LOO. Secondary=best distinct final mean+max+std by LOO then five-fold; cap2; no external metrics.',selected=selected,
        best_new_family=[best([r for r in new if r['name'].startswith(n+'__')]) for n in sorted({r['name'].split('__')[0] for r in new})]))
    return selected


def export_new(row,ds):
    import numpy as np
    import torch,onnxruntime as ort,onnx
    from ..downstream.common import load,normalize,rgb_images
    from ..export import export_encoder
    from .frozen_readouts import MultiReadout
    name,kind=row['name'].split('__');root=Path(row['source_matrix']).parent
    spec=next(s['spec'] for s in json.loads((root/'extract-plan.json').read_text())['models'] if s['spec']['name']==name)
    cache=root/name;meta=json.loads((cache/'cache.json').read_text())
    if digest(spec['checkpoint'])!=meta['plan']['checkpoint_sha256'] or digest(cache/'embeddings.npz')!=meta['sha256']:raise ValueError('Selected cache/checkpoint identity changed')
    if digest('data/train.parquet')!=meta['plan']['data_sha256']:raise ValueError('Challenge data changed')
    z=np.load(cache/'embeddings.npz');ids=z['row_ids'];np.testing.assert_array_equal(ids,np.asarray(meta['plan']['row_ids']))
    if meta['plan']['source_sha256']!=digest(Path(__file__).with_name('frozen_readouts.py')):raise ValueError('Pooled adapter changed')
    encoder=load(spec);before={k:v.clone() for k,v in encoder.state_dict().items()}
    model=MultiReadout(encoder,spec['backbone'],kind);model.eval().requires_grad_(False)
    directory=OUT/'exports';directory.mkdir(exist_ok=True);path=directory/(row['name']+'.onnx')
    export_encoder(model,path)
    # Legacy ONNX padding can leave the known feature width symbolic. Pin only
    # that metadata dimension after dynamic numerical parity establishes1024.
    annotated=onnx.load(str(path));width=annotated.graph.output[0].type.tensor_type.shape.dim[1]
    width.ClearField('dim_param');width.dim_value=1024;onnx.save(annotated,str(path))
    if path.stat().st_size>=20*1024**2:raise ValueError('ONNX exceeds strict20MiB limit')
    options=ort.SessionOptions();options.intra_op_num_threads=1;options.inter_op_num_threads=1
    session=ort.InferenceSession(str(path),options,providers=['CPUExecutionProvider'])
    for start,batch in ((0,1),(0,3),(0,4),(4,4),(8,4),(12,4)):
        chosen=ids[start:start+batch];x=normalize(rgb_images(ds,chosen)).numpy()
        actual=session.run(None,{'image':x})[0]
        expected=np.pad(z[kind][start:start+batch],((0,0),(0,1024-row['dimensions'])))
        if actual.dtype!=np.float32 or not np.isfinite(actual).all():raise ValueError('Invalid ONNX output')
        np.testing.assert_allclose(actual,expected,rtol=1e-3,atol=1e-4)
    graph=onnx.load(str(path));onnx.checker.check_model(graph)
    if any(t.data_location==onnx.TensorProto.EXTERNAL for t in graph.graph.initializer):raise ValueError('ONNX uses external tensor data')
    inputs=session.get_inputs();outputs=session.get_outputs()
    if len(inputs)!=1 or inputs[0].name!='image' or inputs[0].type!='tensor(float)' or inputs[0].shape[1:]!=[3,96,96] or isinstance(inputs[0].shape[0],int):raise ValueError('Invalid/dynamic input signature')
    if len(outputs)!=1 or outputs[0].type!='tensor(float)' or outputs[0].shape[1:]!=[1024] or isinstance(outputs[0].shape[0],int):raise ValueError('Invalid/dynamic output signature')
    for k,v in encoder.state_dict().items():torch.testing.assert_close(v,before[k],rtol=0,atol=0)
    official=directory/(row['name']+'-official.log')
    with official.open('w') as f:subprocess.run([sys.executable,'-m','experiments.lejepa.next_round.frozen_validate',str(path)],stdout=f,stderr=subprocess.STDOUT,check=True)
    return dict(row,path=str(path),sha256=digest(path),size_bytes=path.stat().st_size,parameters=sum(p.numel() for p in model.parameters()),
                checkpoint=spec['checkpoint'],checkpoint_sha256=meta['plan']['checkpoint_sha256'],cache=str(cache/'embeddings.npz'),cache_sha256=meta['sha256'],
                cached_parity=True,parity_challenge_images=16,dynamic_batches=[1,3,4],official_checks=True,official_log=str(official),
                selection='Challenge development five-fold/LOO only; no external task selection',self_contained=True)


def main():
    if not os.environ.get('SLURM_JOB_ID'):raise RuntimeError('Slurm required')
    import torch
    torch.set_num_threads(int(os.environ.get('SLURM_CPUS_PER_TASK','4')))
    OUT.mkdir(parents=True,exist_ok=True);rows=collect();summarize(rows);candidates=choose(rows)
    atomic_json(OUT/'status.json',dict(job_id=os.environ['SLURM_JOB_ID'],phase='exporting',sources={str(p):digest(p) for p in [Path(__file__),Path(__file__).with_name('final_cpu_benchmark.py')]}))
    from training_data import load_training_data
    ds=load_training_data('data/train.parquet')
    fresh=[export_new(r,ds) for r in candidates];old=json.loads((BASE/'frozen/exports/manifest.json').read_text());combined=[]
    for r in old+fresh:
        if digest(r['path'])!=r['sha256'] or Path(r['path']).stat().st_size!=r['size_bytes']:raise ValueError('Candidate artifact changed before benchmark')
        historical={k:r[k] for k in ('fresh_process_cpu','peak_process_rss_kib','memory_scope','latency_ms_batch1','latency_ms_batch4') if k in r}
        clean={k:v for k,v in r.items() if k not in historical}
        if historical:clean['historical_resource_attempt']=dict(historical,note='ru_maxrss may inherit exporter high-water; retained for audit, superseded by /proc counters')
        clean['fresh_process_cpu']=json.loads(subprocess.check_output([sys.executable,'-m','experiments.lejepa.next_round.final_cpu_benchmark',r['path']],text=True))
        clean['latency_ms_batch1']=clean['fresh_process_cpu']['batches']['1']['median_batch_ms']
        clean['latency_ms_batch4']=clean['fresh_process_cpu']['batches']['4']['median_batch_ms']
        combined.append(clean);atomic_json(OUT/'exports/manifest.json',combined)
    atomic_json(OUT/'status.json',dict(job_id=os.environ['SLURM_JOB_ID'],phase='complete',new_exports=len(fresh),combined_candidates=len(combined)))
    print('Completed final challenge-only export selection and corrected CPU audit',flush=True)
if __name__=='__main__':
    try:main()
    except Exception as error:
        if OUT.exists():atomic_json(OUT/('failure-'+os.environ.get('SLURM_JOB_ID','unknown')+'.json'),dict(job_id=os.environ.get('SLURM_JOB_ID'),error=repr(error),phase='failed'))
        raise
