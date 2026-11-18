"""One preflighted ResNet18 training run with standard milestone evaluations."""
import argparse
import csv
import json
import os
from pathlib import Path
import shutil
import sys
import time

from .sweep import atomic_json,digest,run_command,compatible_result


def report(output, reference):
    from .error_analysis import write_csv
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    rows, class_changes, fold_changes = [],[],[]
    baseline = reference['results']['trained']
    for epoch in [50,100,150,200]:
        path = output/f'epoch{epoch}/validation/results.json'
        if not path.exists():
            continue
        result = json.loads(path.read_text())
        for key in ['data_sha256','fold_manifest_sha256']:
            if result[key] != reference[key]:
                raise ValueError('Architecture result differs in data/folds')
        r = result['results']['trained'];m=r['metrics']
        rows.append(dict(epoch=epoch,**{k:m[k] for k in ['accuracy','macro_accuracy','official_local_loo_accuracy']}))
        if epoch==200:
            for current,control in zip(r['folds'],baseline['folds']):
                fold_changes.append(dict(**current,accuracy_delta=current['accuracy']-control['accuracy']))
            for current,control in zip(m['per_class'],baseline['metrics']['per_class']):
                if current['class_id']!=control['class_id']:
                    raise ValueError('Class ordering mismatch')
                class_changes.append(dict(**current,accuracy_delta=current['accuracy']-control['accuracy']))
    atomic_json(output/'comparison.json',dict(milestones=rows,resnet9_control=baseline['metrics']))
    write_csv(output/'class_changes.csv',class_changes);write_csv(output/'fold_changes.csv',fold_changes)
    preflight = json.loads((output/'preflight.json').read_text())
    lines=['# Milestone SSL training: '+preflight['config'].get('backbone','resnet9')+' / '+preflight['config'].get('readout','max2'), '',
           'Configuration: `'+json.dumps(preflight['config'])+'`. Zero padding only at official probe/export interfaces.',
           f'Backbone parameters: {preflight["parameters"]:,}. Preflight ONNX: {preflight["onnx_bytes"]:,} bytes. Measured spatial shapes: {preflight['spatial_shapes']}.', '',
           '| Epoch | Five-fold | Macro | Official local LOO |', '|---|---:|---:|---:|']
    for row in rows:
        lines.append(f'| {row["epoch"]} | {row["accuracy"]:.2%} | {row["macro_accuracy"]:.2%} | {row["official_local_loo_accuracy"]:.2%} |')
    b=baseline['metrics']
    lines += ['', f'ResNet9 seed42 tuned200 control: {b["accuracy"]:.2%} five-fold, {b["macro_accuracy"]:.2%} macro, {b["official_local_loo_accuracy"]:.2%} LOO.',
              'Endpoint matched comparison includes backbone depth/width, GAP readout and native embedding dimension. Frozen pooling results help separate readout effects; this is not a depth-only ablation. Initialization follows the teammate, not the ResNet9 constructor.',
              'resources.json records epoch throughput and peak GPU allocation/reservation; phase_times.json records subprocess wall time. Allocation elapsed GPU time must also include preflight/evaluation overhead. Source/data/fold/config hashes are frozen in plan.json.',
              'All local scores reuse the labeled development population; leaderboard transfer is not inferred. No optional training or public upload is performed.']
    if (output/'resources.json').exists():
        resources=json.loads((output/'resources.json').read_text())
        lines += ['',f'Resources: `{json.dumps({k:v for k,v in resources.items() if k!="epochs"})}`']
    fig,axes=plt.subplots(1,3,figsize=(14,4))
    for axis,key in zip(axes,['accuracy','macro_accuracy','official_local_loo_accuracy']):
        axis.plot([r['epoch'] for r in rows],[100*r[key] for r in rows],marker='o',label=preflight['config'].get('backbone','resnet9'))
        axis.axhline(100*b[key],color='gray',linestyle='--',label='ResNet9 endpoint')
        axis.set(xlabel='Completed epochs',ylabel='Percent',title=key)
    axes[0].legend();fig.tight_layout();fig.savefig(output/'learning_curves.png');plt.close(fig)
    lines += ['', '![Probe curves](learning_curves.png)']
    for epoch in [50,100,150,200]:
        if (output/f'epoch{epoch}/spatial/spatial.png').exists():
            lines += ['',f'![Spatial PCA epoch{epoch}](epoch{epoch}/spatial/spatial.png)']
    metrics=output/'metrics/metrics.csv'
    if metrics.exists():
        records=[r for r in csv.DictReader(metrics.open()) if r.get('loss_epoch')]
        fig,axes=plt.subplots(1,3,figsize=(14,4))
        for axis,key in zip(axes,['loss_epoch','invariance_epoch','sigreg_epoch']):
            axis.plot([int(float(r['epoch']))+1 for r in records],[float(r[key]) for r in records])
            axis.set(xlabel='Completed epochs',title=key)
        fig.tight_layout();fig.savefig(output/'training_curves.png');plt.close(fig)
        lines += ['', '![SSL losses](training_curves.png)']
    (output/'REPORT.md').write_text('\n'.join(lines)+'\n')


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--config',type=Path,default=Path('experiments/lejepa/architecture/resnet18-tuned.json'))
    p.add_argument('--data',type=Path,default=Path('data/train.parquet'))
    p.add_argument('--folds',type=Path,default=Path('experiment_outputs/validation/stratified-5fold-seed42.json'))
    p.add_argument('--output',type=Path,default=Path('experiment_outputs/backbone-pooling/resnet18-tuned-seed42'))
    p.add_argument('--preflight',type=Path,default=Path('experiment_outputs/backbone-pooling/preflight/preflight.json'))
    a=p.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):
        p.error('Run in one GPU Slurm allocation after preflight')
    import torch
    preflight=json.loads(a.preflight.read_text())
    if preflight['status']!='passed' or preflight['onnx_bytes']>=20*1024**2:
        raise ValueError('Backbone preflight failed or exceeds challenge budget')
    for path,expected in preflight['source_hashes'].items():
        if digest(path)!=expected:
            raise ValueError(f'Backbone source changed after preflight: {path}')
    config_path=a.config
    config=json.loads(config_path.read_text())
    if config!=preflight['config']:
        raise ValueError('Training configuration differs from exact-batch preflight')
    reference_path=Path('experiment_outputs/sweeps/lejepa-lr-sigreg-seed42/continued-200/lr-0.003_lambda-0.005/validation/results.json')
    reference=json.loads(reference_path.read_text())
    if reference['data_sha256']!=digest(a.data) or reference['fold_manifest_sha256']!=digest(a.folds):
        raise ValueError('Architecture and reference data/folds differ')
    a.output.mkdir(parents=True,exist_ok=True)
    sources=list(Path(__file__).parent.rglob('*.py'))+[Path(n) for n in ['evaluate.py','training_data.py','export_onnx.py','pyproject.toml','uv.lock']]
    plan=dict(config=config,data_sha256=digest(a.data),fold_manifest_sha256=digest(a.folds),
              preflight_sha256=digest(a.preflight),reference_sha256=digest(reference_path),
              source_hashes={str(path):digest(path) for path in sources})
    plan_path=a.output/'plan.json'
    if plan_path.exists() and json.loads(plan_path.read_text())!=plan:
        raise ValueError('Architecture run plan changed; use a new output directory')
    atomic_json(plan_path,plan)
    source=a.output/'source';source.mkdir(exist_ok=True)
    shutil.copytree(Path(__file__).parent,source/'lejepa',dirs_exist_ok=True,ignore=shutil.ignore_patterns('__pycache__'))
    for path in sources:
        if path.parent==Path('.'):
            shutil.copy2(path,source/path.name)
    shutil.copy2(a.preflight,a.output/'preflight.json');shutil.copy2(a.folds,a.output/'folds.json')
    atomic_json(a.output/'config.json',config)
    times=json.loads((a.output/'phase_times.json').read_text()) if (a.output/'phase_times.json').exists() else []
    def execute(command,log,phase,epoch):
        atomic_json(a.output/'progress.json',dict(phase=phase,target_epoch=epoch,job_id=os.environ['SLURM_JOB_ID']))
        start=time.time();run_command(command,log)
        times.append(dict(phase=phase,epoch=epoch,seconds=time.time()-start,job_id=os.environ['SLURM_JOB_ID']))
        atomic_json(a.output/'phase_times.json',times)
    def completed(path):
        if not path.exists():return 0
        state=torch.load(path,map_location='cpu',weights_only=False)
        if state['hyper_parameters']['config']!=config:raise ValueError('Checkpoint configuration changed')
        return state['epoch']+1
    for epoch in [50,100,150,200]:
        last=a.output/'checkpoints/last.ckpt'
        if completed(last)<epoch:
            command=[sys.executable,'-u','-m','experiments.lejepa.main','--config',a.output/'config.json',
                     '--data',a.data,'--output',a.output,'--accelerator','gpu','--retain-every','50','--stop-after-epochs',str(epoch)]
            if last.exists():command+=['--resume',last]
            execute(command,a.output/'training.log','training',epoch)
        matches=list((a.output/'checkpoints').glob(f'retained-epoch-{epoch-1:03d}-step-*.ckpt'))
        if len(matches)!=1 or completed(matches[0])!=epoch:
            raise ValueError('Missing or incorrect milestone checkpoint')
        checkpoint=matches[0];destination=a.output/f'epoch{epoch}';destination.mkdir(exist_ok=True)
        provenance_path=destination/'provenance.json'
        provenance=json.loads(provenance_path.read_text()) if provenance_path.exists() else {}
        model=destination/'model.onnx'
        if not (model.exists() and provenance.get('checkpoint_sha256')==digest(checkpoint)
                and provenance.get('onnx_sha256')==digest(model)
                and compatible_result(destination/'validation/results.json',digest(model),plan['fold_manifest_sha256'],plan['data_sha256'])):
            execute([sys.executable,'-u','-m','experiments.lejepa.checkpoint_eval','--checkpoint',checkpoint,
                     '--expected-epoch',str(epoch),'--data',a.data,'--folds',a.folds,'--output',destination],destination/'evaluation.log','evaluating',epoch)
        if not (destination/'spatial/diagnostics.json').exists():
            execute([sys.executable,'-u','-m','experiments.lejepa.diagnostics',checkpoint,'--data',a.data,
                     '--output',destination/'spatial','--shared','--count','12','--seed','42'],destination/'diagnostics.log','diagnostics',epoch)
        report(a.output,reference)
    atomic_json(a.output/'progress.json',dict(phase='completed',epoch=200,job_id=os.environ['SLURM_JOB_ID']))


if __name__=='__main__':main()
