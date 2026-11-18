"""Frozen spatial-readout study; extract maps once, reuse the official probe."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys

from .sweep import digest, atomic_json, run_command


def cases(backbone="resnet9"):
    if backbone == "resnet18":
        root = Path("experiment_outputs/backbone-pooling/resnet18-tuned-seed42")
        return [dict(name="resnet18-tuned-200-seed42", checkpoint=str(root/"checkpoints/last.ckpt"), reference=str(root/"epoch200/validation"))]
    base = Path('experiment_outputs')
    tuned = base/'sweeps/lejepa-lr-sigreg-seed42/continued-200/lr-0.003_lambda-0.005'
    original = base/'lejepa-resnet9-seed42-retry'
    duration = base/'duration/tuned-seed43-400'
    jitter = base/'augmentation-study/seed42-200-diversity-retry/jitter-strong'
    return [dict(name='original-200-seed42', checkpoint=str(original/'checkpoints/last.ckpt'),
                 reference=str(original/'cross_validation')),
            dict(name='tuned-200-seed42', checkpoint=str(tuned/'checkpoints/last.ckpt'), reference=str(tuned/'validation')),
            dict(name='tuned-400-seed43', checkpoint=str(duration/'checkpoints/last.ckpt'), reference=str(duration/'validation')),
            dict(name='jitter-200-seed42', checkpoint=str(jitter/'checkpoints/last.ckpt'), reference=str(jitter/'epoch200/validation'))]


def source_plan(data, folds, backbone="resnet9"):
    sources = list(Path(__file__).parent.rglob('*.py')) + [Path(n) for n in ['evaluate.py','training_data.py','export_onnx.py','uv.lock']]
    return dict(data_sha256=digest(data), fold_manifest_sha256=digest(folds),
                sources={str(p):digest(p) for p in sources},
                cases=[dict(c,checkpoint_sha256=digest(c['checkpoint']),reference_sha256=digest(Path(c['reference'])/'results.json')) for c in cases(backbone)])


def aggregate_report(output, plan):
    from .error_analysis import write_csv
    rows = []
    lines = ['# Frozen pooling ablation', '',
             'No convolutional weights are changed. Scores are development results, not hidden transfer estimates.', '',
             '| Checkpoint | Readout | Dimensions | Five-fold | Macro | Local LOO | Change vs native |',
             '|---|---|---:|---:|---:|---:|---:|']
    for case in plan['cases']:
        path = output/case['name']/'results.json'
        if not path.exists():
            continue
        result = json.loads(path.read_text())
        baseline = result['readouts'][result.get('baseline', 'max2')]['metrics']
        for kind, r in result['readouts'].items():
            m = r['metrics']
            row = dict(checkpoint=case['name'], readout=kind, dimensions=r['dimensions'],
                       accuracy=m['accuracy'], macro_accuracy=m['macro_accuracy'],
                       official_local_loo_accuracy=m['official_local_loo_accuracy'],
                       accuracy_delta=m['accuracy']-baseline['accuracy'],
                       macro_delta=m['macro_accuracy']-baseline['macro_accuracy'],
                       loo_delta=m['official_local_loo_accuracy']-baseline['official_local_loo_accuracy'])
            rows.append(row)
            lines.append(f'| {case["name"]} | {kind} | {r["dimensions"]} | {m["accuracy"]:.2%} | {m["macro_accuracy"]:.2%} | {m["official_local_loo_accuracy"]:.2%} | {100*row["accuracy_delta"]:+.2f} pp |')
        lines += ['', f'![Spatial features: {case["name"]}]({case["name"]}/spatial/spatial.png)', '']
    write_csv(output/'comparison.csv',rows)
    atomic_json(output/'comparison.json',rows)
    lines += ['', 'Every reported checkpoint must first reproduce the original native outer and official LOO predictions exactly. Features are checked against saved ONNX embeddings numerically.',
              'Per-class/per-fold paired changes, nearest neighbors, variance and effective rank are saved per checkpoint. PCA maps are identical across readouts for a fixed frozen encoder; the shared map is shown once.',
              'GAP/GMP concatenate channels only; std is population standard deviation over spatial sites. All embeddings are padded to1024 before the unchanged official probe. Exported candidates are checked against the cached representation.',
              'Checkpoint400 uses seed43 and a different schedule; compare readouts within each checkpoint before assessing consistency. Local selection does not establish leaderboard improvement. No public uploads or optional retraining are launched.']
    (output/'REPORT.md').write_text('\n'.join(lines)+'\n')


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--backbone', choices=['resnet9','resnet18'],default='resnet9')
    p.add_argument('--data', type=Path, default=Path('data/train.parquet'))
    p.add_argument('--folds', type=Path, default=Path('experiment_outputs/validation/stratified-5fold-seed42.json'))
    p.add_argument('--output', type=Path, default=Path('experiment_outputs/backbone-pooling/frozen-readouts'))
    a = p.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):
        p.error('Run pooling extraction/probes in Slurm')
    import numpy as np
    import torch
    import onnxruntime as ort
    from training_data import load_training_data
    from evaluate import MEAN,STD
    from .model import load_encoder
    from .readouts import READOUTS, SpatialReadout, FrozenReadout, output_dim
    from .export import export_encoder
    from .cross_validate import fixed_plan, probe_embeddings
    from .error_analysis import write_csv, neighbors
    torch.set_num_threads(int(os.environ.get('SLURM_CPUS_PER_TASK','4')))
    a.output.mkdir(parents=True,exist_ok=True)
    plan = source_plan(a.data,a.folds,a.backbone)
    plan_path = a.output/'plan.json'
    if plan_path.exists() and json.loads(plan_path.read_text()) != plan:
        raise ValueError('Frozen pooling plan/source changed; use a new output root')
    atomic_json(plan_path,plan)
    source = a.output/'source'
    source.mkdir(exist_ok=True)
    shutil.copytree(Path(__file__).parent,source/'lejepa',dirs_exist_ok=True,ignore=shutil.ignore_patterns('__pycache__'))
    shutil.copy2(a.folds,a.output/'folds.json')
    ds = load_training_data(str(a.data))
    labels_all = np.asarray(ds['label'])
    ids = np.flatnonzero(labels_all != -1)
    labels = labels_all[ids]
    folds = fixed_plan(a.folds,ids,labels,plan['data_sha256'])
    viz_ids = np.random.default_rng(42).choice(len(ds),12,replace=False).tolist()
    cache_ids = sorted(set(ids.tolist()+viz_ids))
    for case in plan['cases']:
        directory = a.output/case['name']
        directory.mkdir(exist_ok=True)
        atomic_json(a.output/'progress.json',dict(checkpoint=case['name'],phase='extracting'))
        reference = json.loads((Path(case['reference'])/'results.json').read_text())
        if reference['data_sha256'] != plan['data_sha256'] or reference['fold_manifest_sha256'] != plan['fold_manifest_sha256']:
            raise ValueError('Reference data/folds mismatch')
        encoder = load_encoder(case['checkpoint']).eval().requires_grad_(False)
        channels = encoder.spatial_channels
        baseline_kind = 'gap' if a.backbone == 'resnet18' else 'max2'
        kinds = ('gap','gmp','avgmax','avgmaxstd') if a.backbone == 'resnet18' else READOUTS
        state_before = {k:v.clone() for k,v in encoder.state_dict().items()}
        cache = directory/'cache'
        cache.mkdir(exist_ok=True)
        expected_cache = dict(checkpoint_sha256=case['checkpoint_sha256'],data_sha256=plan['data_sha256'],row_ids=cache_ids)
        if (cache/'cache.json').exists():
            metadata = json.loads((cache/'cache.json').read_text())
            if any(metadata[k] != v for k,v in expected_cache.items()) or metadata['maps_sha256'] != digest(cache/'maps.npy'):
                raise ValueError('Spatial cache provenance/content changed')
        else:
            maps = np.lib.format.open_memmap(cache/'maps.npy',mode='w+',dtype='float32',shape=(len(cache_ids),channels,12,12))
            with torch.inference_mode():
                for start in range(0,len(cache_ids),16):
                    selected = cache_ids[start:start+16]
                    raw = np.stack([np.asarray(ds[row]['image'].convert('RGB'),dtype=np.float32) for row in selected])/255.0
                    x = ((raw-MEAN)/STD).astype(np.float32).transpose(0,3,1,2)
                    maps[start:start+len(selected)] = encoder.spatial(torch.from_numpy(x)).numpy()
            maps.flush()
            del maps
            atomic_json(cache/'cache.json',dict(expected_cache,maps_sha256=digest(cache/'maps.npy')))
        maps = np.load(cache/'maps.npy',mmap_mode='r')
        lookup = {row:i for i,row in enumerate(cache_ids)}
        labeled_maps = torch.from_numpy(np.stack([maps[lookup[int(row)]] for row in ids]))
        features, readouts, prediction_rows, fold_rows, class_rows = {}, {}, [], [], []
        with torch.inference_mode():
            for kind in kinds:
                z = SpatialReadout(kind)(labeled_maps)
                features[kind] = torch.nn.functional.pad(z,(0,1024-z.shape[1])).numpy()
        # Fail closed before interpreting alternatives if the native baseline differs.
        stored = np.load(Path(case['reference'])/'embeddings.npz')
        np.testing.assert_array_equal(stored['row_ids'],ids)
        np.testing.assert_allclose(features[baseline_kind],stored['trained'],rtol=1e-3,atol=1e-4)
        baseline_predictions = list(csv.DictReader((Path(case['reference'])/'predictions.csv').open()))
        baseline = None
        for kind in kinds:
            atomic_json(a.output/'progress.json',dict(checkpoint=case['name'],phase='probing',readout=kind))
            metrics, fold_metrics, pred, loo, confusion = probe_embeddings(features[kind],labels,folds)
            if kind == baseline_kind:
                np.testing.assert_array_equal(pred,[int(r['trained_outer_prediction']) for r in baseline_predictions])
                np.testing.assert_array_equal(loo,[int(r['trained_official_loo_prediction']) for r in baseline_predictions])
                for key in ['accuracy','macro_accuracy','official_local_loo_accuracy']:
                    if metrics[key] != reference['results']['trained']['metrics'][key]:
                        raise ValueError('Original readout does not reproduce historical scores')
                baseline = dict(metrics=metrics,folds=fold_metrics,predictions=pred.copy())
            singular = np.linalg.svd(features[kind]-features[kind].mean(0),compute_uv=False)
            probabilities = singular/max(singular.sum(),1e-12)
            neighbor_positions,_ = neighbors(features[kind],k=3)
            readouts[kind] = dict(dimensions=output_dim(channels,kind),metrics=metrics,folds=fold_metrics,
                                 diagnostics=dict(mean_variance=float(features[kind].var(0).mean()),
                                                  effective_rank=float(np.exp(-(probabilities*np.log(probabilities+1e-12)).sum()))))
            for i,row_id in enumerate(ids):
                prediction_rows.append(dict(readout=kind,row_id=int(row_id),label=int(labels[i]),fold=int(folds[i]),
                                            outer_prediction=int(pred[i]),official_loo_prediction=int(loo[i]),
                                            fixed_vs_max2=int(pred[i]==labels[i] and baseline['predictions'][i]!=labels[i]),
                                            regressed_vs_max2=int(pred[i]!=labels[i] and baseline['predictions'][i]==labels[i]),
                                            neighbors=json.dumps(ids[neighbor_positions[i]].tolist())))
            for fold,control in zip(fold_metrics,baseline['folds']):
                fold_rows.append(dict(readout=kind,**fold,delta=fold['accuracy']-control['accuracy']))
            for row,control in zip(metrics['per_class'],baseline['metrics']['per_class']):
                class_rows.append(dict(readout=kind,**row,delta=row['accuracy']-control['accuracy']))
            np.save(directory/f'{kind}_confusion.npy',confusion)
            print(case['name'],kind,metrics['accuracy'],metrics['official_local_loo_accuracy'],flush=True)
        np.savez_compressed(directory/'embeddings.npz',row_ids=ids,labels=labels,folds=folds,**features)
        write_csv(directory/'predictions.csv',prediction_rows)
        write_csv(directory/'fold_changes.csv',fold_rows)
        write_csv(directory/'class_changes.csv',class_rows)
        result = dict(checkpoint=case,readouts=readouts,baseline_reproduced=True,baseline=baseline_kind,
                      data_sha256=plan['data_sha256'],fold_manifest_sha256=plan['fold_manifest_sha256'])
        atomic_json(directory/'results.json',result)
        # Export max2 plus locally improved alternatives; local selection is explicit.
        candidates = [baseline_kind]+[kind for kind in kinds if kind != baseline_kind and
                               (readouts[kind]['metrics']['accuracy']>baseline['metrics']['accuracy'] or
                                readouts[kind]['metrics']['official_local_loo_accuracy']>baseline['metrics']['official_local_loo_accuracy'])]
        export_dir = directory/'exports'
        export_dir.mkdir(exist_ok=True)
        for kind in candidates:
            model = export_dir/f'{kind}.onnx'
            export_encoder(FrozenReadout(encoder,kind),model)
            options = ort.SessionOptions(); options.intra_op_num_threads=1
            session = ort.InferenceSession(str(model),options,providers=['CPUExecutionProvider'])
            selected = ids[:8]
            raw = np.stack([np.asarray(ds[int(row)]['image'].convert('RGB'),dtype=np.float32) for row in selected])/255.0
            x = ((raw-MEAN)/STD).astype(np.float32).transpose(0,3,1,2)
            np.testing.assert_allclose(session.run(None,{'image':x})[0],features[kind][:8],rtol=1e-3,atol=1e-4)
            atomic_json(export_dir/f'{kind}.json',dict(onnx_sha256=digest(model),size_bytes=model.stat().st_size,
                        cached_representation_parity=True,checkpoint_sha256=case['checkpoint_sha256'],readout=kind))
        for key,value in encoder.state_dict().items():
            torch.testing.assert_close(value,state_before[key],rtol=0,atol=0)
        run_command([sys.executable,'-u','-m','experiments.lejepa.diagnostics',case['checkpoint'],
                     '--data',a.data,'--output',directory/'spatial','--shared','--count','12','--seed','42',
                     '--spatial-cache',cache],directory/'diagnostics.log')
        aggregate_report(a.output,plan)
    atomic_json(a.output/'progress.json',dict(phase='completed',checkpoints=len(plan['cases'])))
    # Prepare a from-scratch follow-up only for a substantial mean matched gain.
    if a.backbone == 'resnet18':
        return
    rows = json.loads((a.output/'comparison.json').read_text())
    average = {kind:float(np.mean([r['accuracy_delta'] for r in rows if r['readout']==kind])) for kind in READOUTS if kind!='max2'}
    best = max(average,key=average.get)
    if average[best]>=.005:
        c = json.loads(Path('experiments/lejepa/augmentations/control.json').read_text())
        c.update(backbone='resnet9',readout=best,resource_metrics=True)
        atomic_json(a.output/f'optional-retrain-{best}.json',c)
    print('Frozen pooling study complete:',a.output/'REPORT.md',flush=True)


if __name__=='__main__':
    main()
