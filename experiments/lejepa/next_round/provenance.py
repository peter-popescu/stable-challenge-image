"""Audit saved checkpoint configs without trusting directory labels (Slurm only)."""
import inspect
import json
import os
from pathlib import Path


def main():
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('Checkpoint audit runs in Slurm')
    import torch
    from ..downstream.common import models
    from ..augmentation_policy import augmentation_metadata
    from ..sweep import atomic_json, digest
    from ..model import make_encoder, make_method
    output=Path('experiment_outputs/next-round/provenance')
    output.mkdir(parents=True,exist_ok=True)
    specs=[s for s in models(True) if s.get('checkpoint')]
    for recipe in ['baseline','winner']:
        for seed in [43,44]:
            path=Path(f'experiment_outputs/seed-repeats/baseline-vs-tuned/{recipe}-seed-{seed}/checkpoints/last.ckpt')
            if path.exists(): specs.append(dict(name=f'{recipe}-200-seed{seed}',checkpoint=str(path)))
    rows=[]
    for spec in specs:
        path=Path(spec['checkpoint'])
        ckpt=torch.load(path,map_location='cpu',weights_only=False)
        config=ckpt['hyper_parameters']['config']
        root=path.parent.parent
        evidence=[]
        for candidate in [root/'config.json',root/'resolved.json',root/'resume.json']:
            if candidate.exists():
                saved=json.loads(candidate.read_text())
                saved=saved.get('config',saved)
                if saved != config: raise ValueError(f'Config conflict: {candidate}')
                evidence.append(dict(path=str(candidate),sha256=digest(candidate)))
        encoder=make_encoder(config)
        row=dict(name=spec['name'],checkpoint=str(path),checkpoint_sha256=digest(path),
                 architecture=config.get('backbone','resnet9'),seed=config['seed'],
                 config=config,augmentation=augmentation_metadata(config),
                 pooling=config.get('readout','max2'),training_horizon=config['epochs'],
                 checkpoint_epoch=ckpt['epoch']+1,global_step=ckpt['global_step'],
                 parameters=sum(p.numel() for p in encoder.parameters()),
                 optimizer_lrs=[g['lr'] for g in ckpt['optimizer_states'][0]['param_groups']],
                 scheduler_state=ckpt.get('lr_schedulers'),configuration_evidence=evidence,
                 metrics_paths=[str(p) for p in root.rglob('metrics.csv') if 'source' not in p.parts])
        rows.append(row)
        print(row['name'],row['seed'],row['checkpoint_epoch'],flush=True)
    atomic_json(output/'checkpoints.json',dict(job_id=os.environ['SLURM_JOB_ID'],checkpoints=rows))
    lines=['# Checkpoint provenance audit','',
           'Authoritative training seeds come from saved checkpoint `hyper_parameters.config`; saved run configs are cross-checked exactly. Epoch counts are completed epochs.', '',
           '| Checkpoint | Backbone | Seed | LR / SIGReg | Pooling | Horizon / epoch | SHA256 | Artifact |',
           '|---|---|---:|---|---|---|---|---|']
    for r in rows:
        c=r['config']
        lines.append(f"| {r['name']} | {r['architecture']} | {r['seed']} | {c['lr']} / {c['lamb']} | {r['pooling']} | {r['training_horizon']} / {r['checkpoint_epoch']} | `{r['checkpoint_sha256']}` | `{r['checkpoint']}` |")
    lines += ['', 'Full hyperparameters, augmentation policy, optimizer endpoint LR, scheduler state, configuration evidence and metric paths are in `checkpoints.json`. The tuned400 model is seed43. Comparing it with tuned200 seed42 confounds seed, duration and LR history. Seed43 and seed44 200-epoch baseline/tuned checkpoints provide initialization-diversity controls.']
    (output/'MANIFEST.md').write_text('\n'.join(lines)+'\n')
    method=make_method(rows[0]['config'])
    module=inspect.getmodule(type(method))
    (output/'pinned-objective.py.txt').write_text(inspect.getsource(module))
    atomic_json(output/'objective.json',dict(module=str(module.__file__),sha256=digest(module.__file__),class_name=type(method).__name__))


if __name__=='__main__': main()
