"""Strict read-only aggregation of all completed external frozen evaluations.

No encoder inference, probe fitting, feature mutation, or test-based selection.
"""
import csv
import hashlib
import json
import os
from pathlib import Path
import time
import numpy as np
from ..error_analysis import classification_summary
from ..sweep import atomic_json,digest

ROOT=Path('experiment_outputs/next-round')
OLD=['original-200-seed42','tuned-200-seed42','tuned-400-seed43','jitter-200-seed42','resnet18-tuned-200-seed42','random-resnet9','random-resnet18','statpool-200-seed42']
NEW=['resnet18-tuned-400-seed42','resnet9-original-400-seed42','resnet9-more-views-200-seed42']
PAIRS=[('original-200-seed42','tuned-400-seed43'),('jitter-200-seed42','statpool-200-seed42')]
HASHES={}
INPUTS={}
COUNTS={'prediction_files':0}


def sha(path):
    path=Path(path);key=str(path)
    if key not in HASHES:HASHES[key]=digest(path)
    return HASHES[key]


def read(path):
    path=Path(path);INPUTS[str(path)]=sha(path)
    return json.loads(path.read_text())


def kinds(spec):
    native='gap' if spec['backbone']=='resnet18' else ('avgmaxstd' if 'statpool' in spec['name'] else 'max2')
    return list(dict.fromkeys([native,'avgmax','avgmaxstd','all_means']))


def dim(spec,kind):
    c=320 if spec['backbone']=='resnet18' else 256
    return dict(gap=c,max2=1024,avgmax=2*c,avgmaxstd=3*c,all_means=600 if c==320 else 704)[kind]


def require(ok,message):
    if not ok:raise ValueError(message)


def check_probe(r,y,path,refit):
    require(r['refit_trainval']==refit,'Unexpected final refit policy: '+str(path))
    cc=r['candidates'];require(len(cc)==13,'Ridge grid length differs')
    np.testing.assert_allclose([c['relative_lambda'] for c in cc],np.logspace(-4,2,13),rtol=0,atol=0)
    chosen=max(range(len(cc)),key=lambda i:(cc[i]['accuracy'],-i))
    require(r['validation']==cc[chosen],'Validation-only selected lambda/metrics differ')
    with np.load(path) as saved:
        pred=saved['predictions'];metrics,cm=classification_summary(y,pred)
        require(np.array_equal(cm,saved['confusion']),'Saved confusion differs: '+str(path))
    require(metrics==r['test'],'Saved prediction test metrics differ: '+str(path))
    require(len(pred)==len(y),'Test row count differs')
    COUNTS['prediction_files']+=1


def stats(values):
    values=np.asarray(values,dtype=float)
    return dict(mean=float(values.mean()),sd=float(values.std(ddof=1)) if len(values)>1 else 0.,n=len(values))


def csvfile(path,rows):
    if not rows:return
    keys=list(rows[0])
    with Path(path).open('w',newline='') as stream:
        out=csv.DictWriter(stream,fieldnames=keys);out.writeheader()
        for row in rows:out.writerow({k:json.dumps(v) if isinstance(v,(list,dict)) else v for k,v in row.items()})


def summarize(directory,dataset,expected_models,reference,all_specs):
    data=read(directory/'dataset.json')
    if reference is not None:require(data==reference,'Models did not use identical dataset/labels/row IDs/splits: '+str(directory))
    labels=np.asarray(data['labels']);n=len(data['row_ids'])
    require(len(set(data['row_ids']))==n and len(labels)==n,'Nonunique/misaligned dataset rows')
    if dataset=='clevr':
        require(n==3500 and data['sizes']==dict(train=1000,val=500,test=2000),'Wrong lightweight CLEVR budget')
        require(data['light_adapter_sha256']==sha(Path(__file__).with_name('clevr_light.py')),'Light adapter source changed')
        require(data['seed']==42 and 'NOT official' in data['protocol'],'Light protocol/version marker missing')
        require({k:len(data['splits'][k]) for k in ['train','val','test','trainval']}==dict(train=1000,val=500,test=2000,trainval=1500),'Wrong light partitions/refit count')
        tasks=['count','distance'];splitsets={'clevr-light':data['splits']};protocol='clevr-light-1000train-500val-2000test; finalrefit1500; NOT officialVTAB1k'
    elif dataset=='dtd':
        require(n==5640 and set(data['split_sets'])=={'1','2','3'},'DTD official split coverage differs')
        splitsets=data['split_sets'];tasks=['dtd'];protocol='DTD official splits1/2/3; train1880,val1880,test1880; finalrefit3760'
    else:
        require(n==60000 and {k:len(data['splits'][k]) for k in ['train','val','test','trainval']}==dict(train=45000,val=5000,test=10000,trainval=50000),'CIFAR official/validation sizes differ')
        splitsets={'official':data['splits']};tasks=['cifar100'];protocol='CIFAR100 official50ktrain/10ktest; validation50perclass seed42; finalrefit50000'
    for ss in splitsets.values():
        train,val,test=[set(ss[k]) for k in ['train','val','test']]
        require(not(train&val or train&test or val&test),'External partition leakage')
        require(set(ss['trainval'])==train|val,'Final trainval partition differs')
        require(train|val|test==set(range(n)),'Partition row coverage differs')
    for stage in ['extract','cpu']:
        require(read(directory/(stage+'-status.json'))['status']=='complete','Incomplete external stage: '+str(directory))
    specs=read(directory/'models.json');require({s['name'] for s in specs}==set(expected_models),'Encoder set differs')
    registry={s['name']:s for s in specs};cache={};expected={}
    for spec in specs:
        name=spec['name'];all_specs.setdefault(name,spec)
        require(all_specs[name]==spec,'Encoder specification changed between datasets')
        meta=read(directory/'features'/name/'cache.json');p=meta['plan']
        require(p['spec']==spec and p['dataset_plan_sha256']==sha(directory/'dataset.json'),'Cache model/dataset binding differs')
        require(p['source_sha256']==sha(Path(__file__).with_name('external_run.py')) and p['data_adapter_sha256']==sha(Path(__file__).with_name('external_data.py')) and p['adapter_sha256']==sha(Path(__file__).with_name('frozen_readouts.py')),'Feature source provenance differs')
        require(p['checkpoint_sha256']==(sha(spec['checkpoint']) if spec.get('checkpoint') else None),'Checkpoint content identity differs')
        require(p['readouts']==kinds(spec) and set(meta['files'])=={k+'.npy' for k in kinds(spec)},'Feature readout coverage differs')
        for kind in kinds(spec):
            file=directory/'features'/name/(kind+'.npy');require(file.is_file(),'Feature artifact missing')
            # Header only: report does not rehash/read gigabytes of frozen arrays.
            x=np.load(file,mmap_mode='r');require(x.shape==(n,dim(spec,kind)) and x.dtype==np.float32,'Feature shape/dtype differs');del x
            cache[str(file)]=meta['files'][kind+'.npy'];expected[name+'--'+kind]=dim(spec,kind)
    if set(expected_models)==set(OLD):
        for a,b in PAIRS:expected[a+'__'+b]=1024
    records={};raw=[];few=[]
    files=sorted((directory/'results').glob('*.json'))
    require(len(files)==len(expected)*len(tasks)*len(splitsets),'Unexpected result count: '+str(directory))
    for path in files:
        r=read(path);rep=r['representation'];task=r['task'];split=r['split']
        require(rep in expected and task in tasks and split in splitsets and r['dimensions']==expected[rep],'Unexpected result identity')
        key=(rep,task,split);require(key not in records,'Duplicate external endpoint');records[key]=r
        plan=r['plan'];require(plan['dataset_plan_sha256']==sha(directory/'dataset.json') and plan['probe_sha256']==sha(Path(__file__).with_name('external_probe.py')) and plan['task']==task and plan['split']==split,'Probe source/data/split provenance differs')
        components=rep.split('__') if '__' in rep else [rep.split('--')[0]]
        kind='avgmax' if '__' in rep else rep.split('--')[1]
        expectedfeatures={str(directory/'features'/name/(kind+'.npy')):cache[str(directory/'features'/name/(kind+'.npy'))] for name in components}
        require(plan['features']==expectedfeatures,'Probe feature hash bindings differ')
        y=labels[:,0 if task=='count' else 1] if dataset=='clevr' else labels
        test_y=y[np.asarray(splitsets[split]['test'])]
        check_probe(r,test_y,path.with_suffix('.npz'),True)
        raw.append(dict(representation=rep,task=task,split=split,protocol=protocol,validation_accuracy=r['validation']['accuracy'],validation_macro_accuracy=r['validation']['macro_accuracy'],test_accuracy=r['test']['accuracy'],test_macro_accuracy=r['test']['macro_accuracy'],relative_lambda=r['validation']['relative_lambda'],test_images=len(test_y),result_path=str(path),result_sha256=sha(path)))
        shouldfew=dataset=='cifar100' and '__' not in rep and rep.endswith(('--max2','--gap','--avgmaxstd'))
        require(bool(r.get('fewshot'))==shouldfew,'Few-shot representation coverage differs')
        if shouldfew:
            ff=r['fewshot'];require(len(ff)==9 and {(f['shots'],f['seed']) for f in ff}=={(s,t) for s in [1,5,20] for t in [42,43,44]},'Few-shot episode coverage differs')
            for f in ff:
                probe=path.parent/(path.stem+'-fewshot')/f"fewshot-{f['shots']}-{f['seed']}.npz"
                check_probe(f,test_y,probe,False)
                few.append(dict(representation=rep,shots=f['shots'],seed=f['seed'],validation_accuracy=f['validation']['accuracy'],validation_macro_accuracy=f['validation']['macro_accuracy'],test_accuracy=f['test']['accuracy'],test_macro_accuracy=f['test']['macro_accuracy'],relative_lambda=f['validation']['relative_lambda'],result_path=str(path)))
    require(set(records)=={(rep,task,split) for rep in expected for task in tasks for split in splitsets},'Missing expected endpoint')
    matrix=read(directory/'matrix.json')
    require(len(matrix)==len(raw),'Saved matrix count differs')
    bykey={(r['representation'],r['task'],r['split']):r for r in raw}
    for r in matrix:
        rr=bykey[(r['representation'],r['task'],r['split'])]
        require(all(r[k]==rr[k] for k in ['validation_accuracy','test_accuracy','test_macro_accuracy']),'Saved matrix metrics differ')
    groups=[]
    for rep in expected:
        model,readout=(rep,'concat-meanmax') if '__' in rep else rep.split('--')
        for task in tasks:
            rr=[r for r in raw if r['representation']==rep and r['task']==task]
            summary={k:stats([r[k] for r in rr]) for k in ['validation_accuracy','validation_macro_accuracy','test_accuracy','test_macro_accuracy']}
            groups.append(dict(model=model,readout=readout,representation=rep,task=task,protocol=protocol,dimensions=expected[rep],
                  validation_accuracy=summary['validation_accuracy']['mean'],validation_macro_accuracy=summary['validation_macro_accuracy']['mean'],
                  test_accuracy=summary['test_accuracy']['mean'],test_macro_accuracy=summary['test_macro_accuracy']['mean'],split_stats=summary,
                  splits=[r['split'] for r in rr],fewshot=False,test_images_per_split=[r['test_images'] for r in rr],source_results=[r['result_path'] for r in rr]))
    return data,raw,groups,few


def paired(rows):
    comparisons=[('original duration','original-200-seed42','resnet9-original-400-seed42','max2'),
                 ('ResNet18 duration','resnet18-tuned-200-seed42','resnet18-tuned-400-seed42','gap'),
                 ('six-local-view recipe','tuned-200-seed42','resnet9-more-views-200-seed42','max2')]
    lookup={(r['model'],r['readout'],r['task']):r for r in rows};out=[]
    for label,a,b,native in comparisons:
        for kind in [native,'avgmaxstd','all_means']:
            for task in ['dtd','cifar100','count','distance']:
                x,y=lookup[a,kind,task],lookup[b,kind,task]
                out.append(dict(comparison=label,control=a,experiment=b,readout=kind,task=task,
                                control_validation=x['validation_accuracy'],experiment_validation=y['validation_accuracy'],
                                control_test=x['test_accuracy'],experiment_test=y['test_accuracy'],
                                validation_delta_pp=100*(y['validation_accuracy']-x['validation_accuracy']),
                                test_delta_pp=100*(y['test_accuracy']-x['test_accuracy']),
                                macro_delta_pp=100*(y['test_macro_accuracy']-x['test_macro_accuracy'])))
    return out


def ensembles(rows):
    lookup={(r['model'],r['readout'],r['task']):r for r in rows};out=[]
    for a,b in PAIRS:
        for task in ['dtd','cifar100','count','distance']:
            en=lookup[a+'__'+b,'concat-meanmax',task]
            aa,bb=lookup[a,'avgmax',task],lookup[b,'avgmax',task]
            out.append(dict(ensemble=a+'__'+b,task=task,validation_accuracy=en['validation_accuracy'],test_accuracy=en['test_accuracy'],
                            component_a_test=aa['test_accuracy'],component_b_test=bb['test_accuracy'],
                            delta_a_pp=100*(en['test_accuracy']-aa['test_accuracy']),delta_b_pp=100*(en['test_accuracy']-bb['test_accuracy']),
                            resnet18_stat_test=lookup['resnet18-tuned-200-seed42','avgmaxstd',task]['test_accuracy']))
    return out


def fmt(x):return f'{100*x:.2f}'


def report(rows,few,pairedrows,ensemble_rows,audit,output):
    lines=['# Final external frozen-representation evaluation','',
        'All11 encoders completed DTD,CIFAR100 and lightweight CLEVR count/distance. The prespecified matrix is retained; test scores did not select new encoders/readouts, experiments or leaderboard artifacts.',
        'Validation and test remain distinct. DTD numbers average official splits1/2/3; their sample SD describes overlapping dataset splits, not independent encoder seeds. CIFAR uses the official50k/10k split with5000-image validation, then refits50k. CLEVR uses fixed1000 train/500 validation/2000 test images and final1500-label refit; **not official VTAB1k or the full VTAB protocol**. All11 encoders share exactly the same3500 row IDs, labels and partition roles.',
        '',f"Verification passed: {audit['raw_endpoints']}raw endpoints,{audit['aggregate_endpoints']}task/representation rows,{audit['fewshot_episodes']}CIFAR few-shot episodes. Saved predictions reproduce every micro/macro/confusion metric. Source/checkpoint/feature-manifest bindings and fixed dataset plans match. Bulk feature arrays were not rehashed in this reporting-only job; their saved SHA bindings are checked against result plans. No feature extraction or probe fitting ran.",
        '', '## Findings','',
        'Longer original-hyperparameter training has mixed transfer: native DTD/count/distance rise while CIFAR changes−0.37pp. Longer ResNet18 GAP improves counting slightly but decreases DTD,CIFAR and distance; multiscale extraction changes that profile. Six local views yield small native changes and lower statistical-readout scores on all four external tasks, so this run supplies no broad transfer gain for the extra compute.',
        '',
        'Multiscale means expose useful texture information: relative to final-layer statistical pooling, ResNet18 DTD improves+2.68pp at200 epochs and+3.74pp at400; CIFAR improves+0.94/+2.71pp. Other encoders and structured tasks retain tradeoffs, including the statistical-pooling-trained model.',
        '',
        'The prespecified jitter+statistical ensemble reaches47.36%DTD and46.20%CIFAR, above its matched mean+max components on both tasks. Its35.80%count score is below the statistical component36.05%; distance ties the jitter component51.60%. Complementarity therefore helps independent semantic/texture tasks without uniformly improving structured information. Random ResNet18 all-stage means already reach48.90%on distance, so task scores must be read alongside random controls.',
        '', '## Matched planned recipe comparisons','',
        'All values below are test accuracy changes in percentage points; validation changes and macro changes are retained in the JSON/CSV. Different planned200/400 cosine schedules are part of the duration recipes. The historical tuned400/seed43 model is not treated as a matched duration control. Six local views change invariance weighting,SIGReg samples,BatchNorm exposure and compute as documented in TRAINING.md; this is a broader recipe comparison. All results remain single trained replicates.',
        '', '| Comparison | Readout | DTD Δpp | CIFAR Δpp | Count Δpp | Distance Δpp |','|---|---|---:|---:|---:|---:|']
    combos=[]
    for p in pairedrows:
        key=p['comparison'],p['readout']
        if key not in combos:combos.append(key)
    for label,kind in combos:
        values=[next(p['test_delta_pp'] for p in pairedrows if p['comparison']==label and p['readout']==kind and p['task']==task) for task in ['dtd','cifar100','count','distance']]
        lines.append('|'+label+'|'+kind+'|'+'|'.join(f'{v:+.2f}' for v in values)+'|')
    lines += ['', '## Prespecified ensembles and matched512-dimensional components','',
        'Two independent encoder coordinate systems are concatenated; no coordinate averaging/block weighting is used. Each component contributes512 mean+max features. Comparisons below retain the fixed pairs, rather than choosing them using external tests.',
        '', '| Pair | Task | Component A | Component B | Concat | Δvs A(pp) | Δvs B(pp) | R18 statistical |','|---|---|---:|---:|---:|---:|---:|---:|']
    for r in ensemble_rows:lines.append(f"|{r['ensemble']}|{r['task']}|{fmt(r['component_a_test'])}|{fmt(r['component_b_test'])}|{fmt(r['test_accuracy'])}|{r['delta_a_pp']:+.2f}|{r['delta_b_pp']:+.2f}|{fmt(r['resnet18_stat_test'])}|")
    lines += ['', '## Complete model-by-task matrix','',
        'Each task cell is **validation / test / test macro (%)**. DTD test mean±split SD is shown separately. Task metrics remain separate; no heterogeneous overall score is computed.',
        '', '| Model / readout | DTD val/test/macro | DTD test SD(pp) | CIFAR val/test/macro | Count val/test/macro | Distance val/test/macro |','|---|---|---:|---|---|---|']
    conditions=[]
    for r in rows:
        if r['representation'] not in conditions:conditions.append(r['representation'])
    lookup={(r['representation'],r['task']):r for r in rows}
    for rep in conditions:
        cells=[]
        for task in ['dtd','cifar100','count','distance']:
            r=lookup[rep,task];cells.append('/'.join(fmt(r[k]) for k in ['validation_accuracy','test_accuracy','test_macro_accuracy']))
        sd=100*lookup[rep,'dtd']['split_stats']['test_accuracy']['sd']
        lines.append(f'|{rep}|{cells[0]}|{sd:.2f}|{cells[1]}|{cells[2]}|{cells[3]}|')
    lines += ['', '## CIFAR100 label budgets','',
        'Fixed1/5/20 support images per class, seeds42/43/44; training support and validation are disjoint. Lambda uses validation; no validation labels enter the few-shot refit. Test mean±sample SD and macro are across these three support draws, not independently trained encoders.',
        '', '| Representation | Shots/class | Val mean(%) | Test mean±SD(%) | Test macro(%) |','|---|---:|---:|---:|---:|']
    for r in few:lines.append(f"|{r['representation']}|{r['shots']}|{fmt(r['validation_accuracy'])}|{fmt(r['test_accuracy'])}±{100*r['split_stats']['test_accuracy']['sd']:.2f}|{fmt(r['test_macro_accuracy'])}|")
    lines += ['', '## Artifacts and interpretation limits','',
        '- `final_external_matrix.json/csv`:180 aggregate full-probe task rows with separate validation/test/macro and split statistics.',
        '- `final_external_raw.json/csv`:270 saved split endpoints and source result paths/hashes.',
        '- `final_external_fewshot.json/csv`:63 aggregate budget rows; `final_external_fewshot_raw.json`:189 episode endpoints.',
        '- `final_external_paired.json/csv`:36 matched native/statistical/all-stage-mean task comparisons.',
        '- `final_external_ensembles.json/csv`:8 matched pair/task comparisons.',
        '- `final_external_verification.json`:dataset/checkpoint/source/result audit and input hashes.',
        '', 'Random ResNet9 and ResNet18 are retained throughout the complete matrix as controls. Structured counting/distance can depend on appearance and position cues without semantic reasoning. CLEVR task scores do not imply reasoning-network performance. External endpoints complement the existing segmentation, transformation-specific correspondence and color-accessibility profiles; they do not replace those metrics or establish hidden leaderboard gains. If these validation results guide future SSL/readout design, call them development data. No external test result in this round triggers new training, new readout selection or a submission. Full CLEVR artifacts remain preserved but the cancelled full evaluation is excluded from these completed lightweight results.']
    body='\n'.join(lines)+'\n'
    prose={
        'All11':'All 11','DTD,CIFAR100':'DTD, CIFAR-100','official splits1/2/3':'official splits 1/2/3',
        'official50k/10k split with5000-image validation, then refits50k':'official 50,000/10,000 split with 5,000-image validation, then refits on all 50,000 training images',
        'fixed1000 train/500 validation/2000 test images and final1500-label refit':'fixed 1,000 training, 500 validation and 2,000 test images, with a final refit on 1,500 labels',
        'same3500':'same 3,500','270raw':'270 raw','180task':'180 task','189CIFAR':'189 CIFAR',
        'changes−0.37pp':'changes by −0.37 pp','DTD,CIFAR and distance':'DTD, CIFAR and distance',
        'improves+2.68pp at200 epochs and+3.74pp at400':'improves by +2.68 pp at 200 epochs and +3.74 pp at 400',
        'improves+0.94/+2.71pp':'improves by +0.94/+2.71 pp','statistical-pooling-trained model':'model trained with statistical pooling',
        'reaches47.36%DTD and46.20%CIFAR':'reaches 47.36% on DTD and 46.20% on CIFAR-100',
        'Its35.80%count':'Its 35.80% count','component36.05%':'component at 36.05%','component51.60%':'component at 51.60%',
        'reach48.90%on':'reach 48.90% on','planned200/400':'planned 200/400','tuned400/seed43':'tuned 400/seed 43',
        'weighting,SIGReg samples,BatchNorm':'weighting, SIGReg samples, BatchNorm',
        '## Prespecified ensembles and matched512-dimensional components':'## Prespecified ensembles and their components',
        'contributes512':'contributes 512','Fixed1/5/20':'Fixed 1/5/20','seeds42/43/44':'seeds 42/43/44',
        'rows; `final_external_fewshot_raw.json`:189':'rows; `final_external_fewshot_raw.json`: 189',
        'json/csv`:180':'json/csv`: 180','json/csv`:270':'json/csv`: 270','json/csv`:63':'json/csv`: 63',
        'json/csv`:36':'json/csv`: 36','json/csv`:8':'json/csv`: 8'
    }
    for before,after in prose.items():body=body.replace(before,after)
    (output/'FINAL_EXTERNAL_REPORT.md').write_text(body)


def main():
    if not os.environ.get('SLURM_JOB_ID'):raise RuntimeError('Run lightweight reporting in Slurm')
    start=time.monotonic();output=ROOT/'external';all_specs={};references={};raw=[];rows=[];fewraw=[];validated=[]
    directories=[(output/'dtd','dtd',OLD),(output/'cifar100','cifar100',OLD),(output/'clevr-light/clevr','clevr',OLD)]
    for model in NEW:
        directories += [(ROOT/'external-new'/model/'dtd','dtd',[model]),(ROOT/'external-new'/model/'cifar100','cifar100',[model]),(ROOT/'external-new'/(model+'-clevr-light')/'clevr','clevr',[model])]
    for directory,dataset,models in directories:
        print('Validate',directory,flush=True)
        data,rr,gg,ff=summarize(directory,dataset,models,references.get(dataset),all_specs)
        references.setdefault(dataset,data);raw+=rr;rows+=gg;fewraw+=ff;validated.append(str(directory))
    require(set(all_specs)==set(OLD+NEW) and len(raw)==270 and len(rows)==180 and len(fewraw)==189,'Final expected coverage differs')
    few=[]
    for rep in sorted({r['representation'] for r in fewraw}):
        model,kind=rep.split('--')
        for shots in [1,5,20]:
            rr=[r for r in fewraw if r['representation']==rep and r['shots']==shots]
            require(len(rr)==3,'Few-shot replicate count differs');ss={k:stats([r[k] for r in rr]) for k in ['validation_accuracy','validation_macro_accuracy','test_accuracy','test_macro_accuracy']}
            few.append(dict(model=model,readout=kind,representation=rep,task='cifar100-fewshot',protocol=f'CIFAR100 {shots}shots/class; seeds42/43/44; validation5000; no trainval refit',shots=shots,
                            validation_accuracy=ss['validation_accuracy']['mean'],validation_macro_accuracy=ss['validation_macro_accuracy']['mean'],test_accuracy=ss['test_accuracy']['mean'],test_macro_accuracy=ss['test_macro_accuracy']['mean'],split_stats=ss,fewshot=True,seeds=[42,43,44]))
    require(len(few)==63,'Few-shot summary count differs')
    pairs=paired(rows);ens=ensembles(rows)
    audit=dict(status='passed',job_id=os.environ['SLURM_JOB_ID'],encoders=len(all_specs),raw_endpoints=len(raw),aggregate_endpoints=len(rows),fewshot_episodes=len(fewraw),fewshot_summaries=len(few),
               directories=validated,saved_prediction_files_verified=COUNTS['prediction_files'],source_sha256={name:sha(Path(__file__).with_name(name)) for name in ['external_run.py','external_data.py','external_probe.py','frozen_readouts.py','clevr_light.py','final_external_report.py']},clevr_row_ids_sha256=hashlib.sha256(json.dumps(references['clevr']['row_ids']).encode()).hexdigest(),
               clevr_rows_identical_all11=True,clevr_sizes=references['clevr']['sizes'],source_checks='Exact dataset plans and model/source/cache hash bindings; prediction/confusion reconstruction; bulk feature arrays only header-checked, not rehashed',
               checkpoint_sha256={m:sha(s['checkpoint']) if s.get('checkpoint') else None for m,s in all_specs.items()},input_sha256=INPUTS,
               seconds=time.monotonic()-start,full_clevr='Cancelled full-dataset results excluded; full/partial artifacts unchanged')
    for name,value in [('matrix',rows),('raw',raw),('fewshot',few),('fewshot_raw',fewraw),('paired',pairs),('ensembles',ens),('combined',rows+few),('verification',audit)]:
        atomic_json(output/('final_external_'+name+'.json'),value)
        if isinstance(value,list) and value and all(list(r)==list(value[0]) for r in value):csvfile(output/('final_external_'+name+'.csv'),value)
    report(rows,few,pairs,ens,audit,output)
    print(json.dumps({k:audit[k] for k in ['status','encoders','raw_endpoints','aggregate_endpoints','fewshot_episodes','fewshot_summaries','seconds']}),flush=True)


if __name__=='__main__':main()
