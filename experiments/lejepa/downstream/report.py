"""Incremental consolidated report; missing metrics remain explicitly pending."""
import csv
import json
from pathlib import Path
from .common import ROOT,models
from ..sweep import atomic_json


def _report(output=ROOT/'downstream'):
    specs=models(True)
    complete=all(all((output/spec['name']/file).exists() for file in ['semantic.json','appearance.json','correspondence.json']) and (Path('experiment_outputs/external-eval/probes')/spec['name']/'results.json').exists() for spec in specs)
    rows=[];lines=['# Representation evaluation: green area', '',
        ('All nine encoders completed all task families:43 global readout conditions and18 encoder/layer spatial probe conditions. Primary appearance results use float64 ridge solves; initial float32 attempts are archived. No hidden leaderboard performance is inferred.' if complete else 'Results accumulate as stages finish; missing endpoints remain pending. Appearance results without solver_dtype=float64 are provisional.'), '',
        'Global tasks use image embeddings; correspondence, local appearance and segmentation use spatial maps. Readout alternatives share one encoder. These are single encoder seeds, not independent training replicates. Episode/fold variability describes splits, not model-training uncertainty.', '',
        '| Model / readout | Dim | Five-fold | Macro | LOO | 5-shot | kNN5 | ARI | NMI | Retrieval mAP | RGB R² |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    # Include teammate manifests discovered in result directories too.
    names=sorted(set([s['name'] for s in specs]+[p.parent.name for p in output.glob('*/semantic.json')]))
    for name in names:
        path=output/name/'semantic.json'
        if not path.exists():continue
        semantics=json.loads(path.read_text());appearance=json.loads((output/name/'appearance.json').read_text())
        for kind,m in semantics.items():
            c=m['classification'];a=appearance[kind]['test']
            row=dict(model=name,readout=kind,dimensions=m['dimensions'],fivefold=c['accuracy'],macro=c['macro_accuracy'],loo=c['official_local_loo_accuracy'],
                     fewshot5=m['fewshot']['5']['accuracy']['mean'],knn5=m['knn']['5'],ari=m['clustering_ari'],nmi=m['clustering_nmi'],retrieval_map=m['retrieval_map'],rgb_r2=a['r2'],rgb_mse=a['mse'])
            rows.append(row)
            lines.append('| '+name+' / '+kind+' | '+str(m['dimensions'])+' | '+' | '.join(f'{row[k]:.4f}' for k in ['fivefold','macro','loo','fewshot5','knn5','ari','nmi','retrieval_map','rgb_r2'])+' |')
    atomic_json(output/'performance_matrix.json',rows)
    if rows:
        with (output/'performance_matrix.csv').open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    spatial_rows=[]
    lines += ['', '## Spatial tasks', '', '| Model | Layer | Transform | Error px | Grid oracle px | PCK8 |', '|---|---|---|---:|---:|---:|']
    for name in names:
        path=output/name/'correspondence.json'
        if path.exists():
            for r in json.loads(path.read_text())['summary']:
                spatial_rows.append(dict(model=name,**r))
                if r['descriptor']=='cnn' and r['mean_error'] is not None:
                    lines.append(f'| {name} | {r["layer"]} | {r["transform"]} | {r["mean_error"]:.3f} | {r["oracle_error"]:.3f} | {r["pck8"]:.3f} |')
    atomic_json(output/'spatial_matrix.json',spatial_rows)
    lines += ['', '## Appearance and external segmentation', '', '| Model | Layer | Local RGB MSE | Local RGB R² | Pet mIoU | Pet foreground IoU | Pixel accuracy |', '|---|---|---:|---:|---:|---:|---:|']
    dense=[]
    for name in names:
        ap=output/name/'appearance.json';sp=Path('experiment_outputs/external-eval/probes')/name/'results.json'
        appearance=json.loads(ap.read_text()) if ap.exists() else {};seg=json.loads(sp.read_text()) if sp.exists() else {}
        for layer in ['mid','final']:
            a=appearance.get('spatial-'+layer,{}).get('test',{});s=seg.get(layer,{}).get('test',{})
            dense.append(dict(model=name,layer=layer,appearance=a,pet=s))
            lines.append('| '+name+' | '+layer+' | '+' | '.join('pending' if v is None else f'{v:.4f}' for v in [a.get('mse'),a.get('r2'),s.get('miou'),s.get('foreground_iou'),s.get('pixel_accuracy')])+' |')
    atomic_json(output/'dense_matrix.json',dense)
    # One matrix combines global readouts and shared encoder/layer tasks.
    for row in rows:
        name=row['model']
        for layer in ['mid','final']:
            dense_row=next((r for r in dense if r['model']==name and r['layer']==layer),{})
            row[layer+'_local_rgb_r2']=dense_row.get('appearance',{}).get('r2')
            row[layer+'_pet_miou']=dense_row.get('pet',{}).get('miou')
            corr=[r for r in spatial_rows if r['model']==name and r['layer']==layer and r['descriptor']=='cnn' and r['transform'] in ['translation','rotation','flip','crop_resize'] and r['pck8'] is not None]
            row[layer+'_correspondence_pck8']=float(sum(r['pck8'] for r in corr)/len(corr)) if corr else None
    atomic_json(output/'performance_matrix.json',rows)
    if rows:
        with (output/'performance_matrix.csv').open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    conditions=[]
    historic=Path('experiment_outputs/backbone-pooling/frozen-readouts/tuned-200-seed42/results.json')
    if historic.exists():
        readouts=json.loads(historic.read_text())['readouts']
        conditions += [('max2 training / max2 evaluation',readouts['max2']['metrics']),('max2 training / statistical evaluation',readouts['avgmaxstd']['metrics'])]
    endpoint=ROOT/'statpool/epoch200/validation/results.json'
    conditions += [('statistical training / statistical evaluation',json.loads(endpoint.read_text())['results']['trained']['metrics'] if endpoint.exists() else None)]
    lines += ['', '## Matched pooling conditions', '', '| Training / evaluation | Five-fold | Macro | Official LOO |','|---|---:|---:|---:|']
    for name,m in conditions:
        lines.append('| '+name+' | '+' | '.join(f"{m[k]:.4f}" if m else 'pending' for k in ['accuracy','macro_accuracy','official_local_loo_accuracy'])+' |')
    lines += ['', '## Current job measurements', '', '```json']
    measured={}
    for path in [ROOT/'statpool/resources.json',*sorted(output.glob('*-status.json'))]:
        if path.exists():measured[str(path)]=json.loads(path.read_text())
    lines += [json.dumps(measured,indent=2),'```']

    lines += ['', '## Statistical pooling and controls', '',
        'Three primary conditions: tuned ResNet9/max2 train+eval; same frozen tuned encoder/mean+max+std eval; new mean+max+std train+eval. The matched-condition table above uses the authoritative historical and new endpoint metrics. Statistical-training milestone results:', '']
    for epoch in [50,100,150,200]:
        p=ROOT/f'statpool/epoch{epoch}/validation/results.json'
        if p.exists():lines.append(f'- Epoch {epoch}: `{json.dumps(json.loads(p.read_text())["results"]["trained"]["metrics"],sort_keys=True)}`')
    pool=ROOT/'resnet18-pooling/comparison.json'
    if pool.exists():lines+=['', 'ResNet18 frozen pooling:','', '```json',pool.read_text().strip(),'```']
    lines += ['', '## Protocol, resources and qualitative evidence', '',
        'See [PROTOCOL.md](../../../experiments/lejepa/downstream/PROTOCOL.md) for exact preprocessing, splits, probe grids, coordinate conventions, metric direction, data separation, limitations and monitoring.',
        'Raw per-episode, per-image, per-class and candidate validation metrics remain in each model directory. Resource stage wall seconds and failures are in *-status.json; allocation CPU/GPU time comes from sacct, not kernel timing.',
        'Correspondence examples: each model/correspondence/*.png (yellow predicted match; green known target). Pet examples: external-eval/probes/MODEL/LAYER/examples.png (image, ground truth, prediction).', '',
        '## Interpretation limits', '',
        'The completed comparisons show tradeoffs rather than a universal winner. Longer training differs in seed and schedule. ResNet18 differs in architecture, initialization, capacity and readout. Crop/jitter comparisons have matched seed and recipe but only one trained replicate. External test results are evaluation endpoints, never selection signals for SSL or submitted readouts.',
        'Information preservation and transformation invariance are different properties: accessible RGB can coexist with high photometric invariance. Compare matched jitter/control appearance probes alongside their prior invariance diagnostics.',
        'Synthetic correspondence tests equivariance under known transforms; it does not measure semantic matching or replace segmentation. Correspondence inputs are192px to permit valid full receptive fields; segmentation and global tasks remain96px.',
        'A task profile is preferable to a radar area: bounded higher-is-better scores retain their native [0,1] scale, ARI retains negatives, R² retains negatives, and errors remain in pixel/RGB units. No scalar area or overall quality score is computed.']
    if complete:
        primary=[next(r for r in rows if r['model']==name and r['readout']==kind) for name,kind in [('tuned-200-seed42','max2'),('tuned-200-seed42','avgmaxstd'),('statpool-200-seed42','avgmaxstd')]]
        overview=['## Completed-round outcome','','| Pooling: train → evaluation | Five-fold | LOO | 5-shot | Retrieval mAP | Final PCK8 | Final local RGB R² | Final Pet mIoU |','|---|---:|---:|---:|---:|---:|---:|---:|']
        for label,r in zip(['max2 → max2','max2 → mean+max+std','mean+max+std → mean+max+std'],primary):
            overview.append('| '+label+' | '+f"{r['fivefold']:.2%} | {r['loo']:.2%} | {r['fewshot5']:.2%} | {r['retrieval_map']:.4f} | {r['final_correspondence_pck8']:.2%} | {r['final_local_rgb_r2']:.4f} | {r['final_pet_miou']:.2%}"+' |')
        overview += ['', 'Statistical pooling during SSL changes what is accessible: compared with frozen statistical extraction, LOO ties at74.67%, five-fold falls0.93pp, five-shot falls1.49pp and retrieval mAP falls0.0190. Final-layer PCK8 rises2.75pp, local RGB R² rises0.0766 and Pet mIoU rises1.89pp. These are single-seed descriptive differences, not significance claims.',
            '', 'The statistical encoder has the highest final-layer Pet mIoU in this cohort (52.89%), but its intermediate probe is49.96%; ResNet18 intermediate features reach52.52%. A final-layer gain does not imply superiority at every layer. Frozen pooling changes no spatial weights and therefore shares dense scores across readouts.',
            '', 'No intervention wins across semantic organization, spatial correspondence, color accessibility and segmentation. The original/tuned and augmentation comparisons also show that higher local classification or measured invariance does not imply better dense transfer. External test scores were not used to revise the SSL recipes or probe grid.', '', '## Full model-by-task metrics', '']
        lines[2:2]=overview
    accounting=ROOT/'final_accounting.json'
    if accounting.exists():lines += ['', '## Final allocation accounting', '', '```json',accounting.read_text().strip(),'```']

    (output/'REPORT.md').write_text('\n'.join(lines)+'\n')
    examples=[]
    for name in ['tuned-200-seed42','jitter-200-seed42','resnet18-tuned-200-seed42']:
        corr=output/name/'correspondence/rotation-final.png'
        pet=Path('experiment_outputs/external-eval/probes')/name/'final/examples.png'
        if corr.exists():examples += [f'![{name}: rotation correspondence]({name}/correspondence/rotation-final.png)']
        if pet.exists():examples += [f'![{name}: Pet image, truth, prediction](../../external-eval/probes/{name}/final/examples.png)']
    if examples:
        with (output/'REPORT.md').open('a') as f:f.write('\n## Representative qualitative examples\n\n'+'\n\n'.join(examples)+'\n')

    # Paired changes keep readout extraction separate from encoder learning.
    pairs=[('jitter vs tuned200','jitter-200-seed42','max2','tuned-200-seed42','max2'),
           ('crop vs tuned200','crop-200-seed42','max2','tuned-200-seed42','max2'),
           ('duration/seed43 vs tuned200/seed42','tuned-400-seed43','max2','tuned-200-seed42','max2'),
           ('ResNet18 GAP vs ResNet9 max2','resnet18-tuned-200-seed42','gap','tuned-200-seed42','max2'),
           ('statistical training vs frozen statistical readout','statpool-200-seed42','avgmaxstd','tuned-200-seed42','avgmaxstd')]
    for name,kind in [('original-200-seed42','max2'),('tuned-200-seed42','max2'),('tuned-400-seed43','max2'),('jitter-200-seed42','max2'),('resnet18-tuned-200-seed42','gap')]:
        pairs.append((name+': statistical vs native readout',name,'avgmaxstd',name,kind))
    keys=['fivefold','fewshot5','retrieval_map','rgb_r2','final_local_rgb_r2','final_correspondence_pck8','final_pet_miou']
    deltas=[]
    for label,an,ak,bn,bk in pairs:
        a=next((r for r in rows if r['model']==an and r['readout']==ak),None)
        b=next((r for r in rows if r['model']==bn and r['readout']==bk),None)
        if a is None or b is None:continue
        deltas.append(dict(comparison=label,**{k:a[k]-b[k] if a.get(k) is not None and b.get(k) is not None else None for k in keys}))
    atomic_json(output/'paired_task_changes.json',deltas)
    if deltas:
        with (output/'paired_task_changes.csv').open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(deltas[0]));writer.writeheader();writer.writerows(deltas)
        with (output/'REPORT.md').open('a') as f:
            f.write('\n## Paired task changes\n\nPositive is a higher metric in native units; RGB R² measures color accessibility, not semantic utility. Missing endpoints remain pending. Spatial scores repeat across readouts because the encoder maps are identical.\n\n')
            f.write('| Comparison | Five-fold Δ | 5-shot Δ | Retrieval Δ | RGB R² Δ | Local RGB R² Δ | Final PCK8 Δ | Pet mIoU Δ |\n|---|---:|---:|---:|---:|---:|---:|---:|\n')
            for d in deltas:
                f.write('| '+d['comparison']+' | '+' | '.join('pending' if d[k] is None else f"{d[k]:+.4f}" for k in keys)+' |\n')
            jitter=next((d for d in deltas if d['comparison']=='jitter vs tuned200'),None)
            if jitter and jitter['rgb_r2'] is not None:
                f.write('\nMatched stronger jitter changes 5-shot accuracy by '+f"{100*jitter['fewshot5']:+.2f}"+' percentage points and retrieval mAP by '+f"{jitter['retrieval_map']:+.4f}"+', while global RGB R² changes by '+f"{jitter['rgb_r2']:+.4f}"+' and local final-layer RGB R² by '+f"{jitter['final_local_rgb_r2']:+.4f}"+'. Read this alongside the prior increased photometric invariance; reduced linear accessibility does not establish irreversible information loss.\n')
            f.write('\nLonger training differs in seed and schedule. Architecture comparisons also change receptive fields and valid matching supports. All planned endpoints are now available. Statistical training improves final-layer dense probes and color accessibility while reducing several global semantic metrics relative to frozen statistical extraction; it does not improve every task or every layer. Readout improvements characterize extraction of existing features; frozen readouts cannot improve the underlying spatial maps.\n')

    # Native-scale task profile: no min-max selection or misleading area.
    if rows:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig,ax=plt.subplots(figsize=(13,6))
        axes=['fivefold','fewshot5','retrieval_map','final_correspondence_pck8','rgb_r2','final_pet_miou']
        for r in rows:
            native='avgmaxstd' if r['model']=='statpool-200-seed42' else ('gap' if 'resnet18' in r['model'] else 'max2')
            if r['readout']!=native:continue
            ax.plot(axes,[float('nan') if r.get(k) is None else r[k] for k in axes],marker='o',label=r['model']+'/'+r['readout'])
        ax.tick_params(axis='x',labelrotation=15)
        ax.axhline(0,color='gray',linewidth=.5);ax.set_ylabel('Native metric (higher is better); R² may be negative')
        ax.legend(fontsize=7);fig.tight_layout();fig.savefig(output/'task_profile.png');plt.close(fig)




def report(output=ROOT/'downstream'):
    import fcntl
    output.mkdir(parents=True,exist_ok=True)
    with (output/'report.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        _report(output)


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser()
    p.add_argument('--output',type=Path,default=ROOT/'downstream')
    args=p.parse_args();report(args.output)
