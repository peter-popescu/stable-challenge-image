"""Read-only final checkpoint, LR-history, milestone and allocation audit in Slurm."""
import csv
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import re
import subprocess

from ..augmentation_policy import augmentation_metadata
from ..sweep import atomic_json, digest


RUNS=[('resnet18-tuned-400',203253,'ResNet18 tuned200 GAP'),
      ('resnet9-original-400',203254,'Original ResNet9 200 max2'),
      ('resnet9-more-views-200',203255,'Tuned ResNet9 four-view200 max2')]
METRICS=['accuracy','macro_accuracy','official_local_loo_accuracy']


def audited_history(root,config,size):
    steps=size//config['batch_size']
    expected_count=steps*config['epochs']
    rates={};losses={};epoch_losses={};max_error=0.;max_loss_error=0.
    for row in csv.DictReader((root/'metrics/metrics.csv').open()):
        step=int(row['step'])
        if row.get('lr-AdamW'):
            if step in rates:raise ValueError('Repeated LR step')
            epoch=step//steps;warmup=config['warmup_epochs']
            scale=((epoch+1)/max(1,warmup) if epoch<warmup else
                   .5*(1+math.cos(math.pi*min((epoch-warmup)/max(1,config['epochs']-warmup),1))))
            expected=config['lr']*scale;actual=float(row['lr-AdamW'])
            if not math.isfinite(actual) or not math.isclose(actual,expected,rel_tol=1e-12,abs_tol=1e-15):
                raise ValueError(f'Actual LR differs from planned horizon at step{step}')
            rates[step]=actual;max_error=max(max_error,abs(actual-expected))
        if row.get('loss_step'):
            if step in losses:raise ValueError('Repeated loss step')
            values={k:float(row[k+'_step']) for k in ['loss','invariance','sigreg']}
            if not all(math.isfinite(v) for v in values.values()):raise ValueError('Nonfinite SSL loss')
            expected=values['invariance']+config['lamb']*values['sigreg']
            if not math.isclose(values['loss'],expected,rel_tol=2e-6,abs_tol=2e-6):raise ValueError('Loss scaling mismatch')
            max_loss_error=max(max_loss_error,abs(values['loss']-expected));losses[step]=values
        if row.get('loss_epoch'):
            epoch=int(row['epoch'])+1
            values={k:float(row[k+'_epoch']) for k in ['loss','invariance','sigreg']}
            if not all(math.isfinite(v) for v in values.values()):raise ValueError('Nonfinite epoch loss')
            epoch_losses[epoch]=values
    if sorted(rates)!=list(range(expected_count)) or sorted(losses)!=list(range(expected_count)):
        raise ValueError('Incomplete actual optimizer LR/loss history')
    if sorted(epoch_losses)!=list(range(1,config['epochs']+1)):raise ValueError('Incomplete epoch loss history')
    prior=json.loads((root/'lr_history.json').read_text())
    if prior['planned_horizon']!=config['epochs'] or prior['steps']!=[dict(step=k,lr=v) for k,v in sorted(rates.items())]:
        raise ValueError('Saved LR report differs from raw metrics')
    return dict(status='passed',schedule_horizon=config['epochs'],steps_per_epoch=steps,
                optimizer_steps=expected_count,finite_loss_steps=len(losses),finite_epoch_losses=len(epoch_losses),
                maximum_lr_absolute_error=max_error,maximum_loss_formula_absolute_error=max_loss_error,
                first_step_lr=rates[0],last_applied_optimizer_lr=rates[expected_count-1],
                epoch_start_lrs={str(e):rates[e*steps] for e in range(0,config['epochs'],50)},
                final_epoch_losses=epoch_losses[config['epochs']],metrics_sha256=digest(root/'metrics/metrics.csv'))


def main():
    if not os.environ.get('SLURM_JOB_ID'):raise RuntimeError('Checkpoint loading/audit requires Slurm')
    import torch
    from ..model import load_encoder
    torch.set_num_threads(int(os.environ.get('SLURM_CPUS_PER_TASK','2')))
    output=Path('experiment_outputs/next-round/provenance')
    original=output/'checkpoints.json'
    old=json.loads(original.read_text())
    rows=[]
    for row in old['checkpoints']:
        path=Path(row['checkpoint'])
        if digest(path)!=row['checkpoint_sha256']:raise ValueError('Historical final checkpoint identity changed')
        checkpoint=torch.load(path,map_location='cpu',weights_only=False)
        if checkpoint['hyper_parameters']['config']!=row['config'] or checkpoint['epoch']+1!=row['checkpoint_epoch']:
            raise ValueError('Historical checkpoint metadata changed')
        rows.append(dict(row,final_reaudit=dict(job_id=os.environ['SLURM_JOB_ID'],status='unchanged_identity_and_config')))
        del checkpoint
    resources=[];comparisons=[]
    for name,job,control_name in RUNS:
        root=Path('experiment_outputs/next-round/training')/name
        path=root/'checkpoints/last.ckpt'
        c=json.loads((root/'config.json').read_text())
        checkpoint=torch.load(path,map_location='cpu',weights_only=False)
        if checkpoint['hyper_parameters']['config']!=c or checkpoint['epoch']+1!=c['epochs']:
            raise ValueError('Final checkpoint config or epoch mismatch')
        evidence=[]
        for f in [root/'resolved.json',root/'resume.json',Path('experiments/lejepa/next_round')/(name+'.json')]:
            value=json.loads(f.read_text());saved=value.get('config',value)
            if saved!=c:raise ValueError('Config differs from saved run evidence')
            evidence.append(dict(path=str(f),sha256=digest(f)))
        resolved=json.loads((root/'resolved.json').read_text())
        log=root/'training.log';text=log.read_text();seed=re.search(r'Seed set to (\d+)',text)
        if not seed or int(seed.group(1))!=c['seed']:raise ValueError('Initial seed log differs')
        encoder=load_encoder(path)
        for key,tensor in checkpoint['state_dict'].items():
            if tensor.is_floating_point() and not torch.isfinite(tensor).all():raise ValueError('Nonfinite saved SSL tensor')
        milestone_rows=[]
        data_hash=resolved['data_sha256'];fold_hash=digest(root/'folds.json')
        for epoch in range(50,c['epochs']+1,50):
            matches=list((root/'checkpoints').glob(f'retained-epoch-{epoch-1:03d}-step-*.ckpt'))
            if len(matches)!=1:raise ValueError('Missing or repeated retained checkpoint')
            milestone=matches[0];state=torch.load(milestone,map_location='cpu',weights_only=False)
            if state['epoch']+1!=epoch or state['hyper_parameters']['config']!=c:raise ValueError('Milestone provenance mismatch')
            if state['global_step']!=(resolved['dataset_size']//c['batch_size'])*epoch:raise ValueError('Wrong milestone global step')
            evaluation=root/f'epoch{epoch}'
            provenance=json.loads((evaluation/'provenance.json').read_text())
            if provenance['checkpoint_sha256']!=digest(milestone) or provenance['config']!=c:raise ValueError('Milestone evaluation identity differs')
            metrics=json.loads((evaluation/'validation/results.json').read_text())
            if metrics['data_sha256']!=data_hash or metrics['fold_manifest_sha256']!=fold_hash:raise ValueError('Evaluation data/folds differ')
            milestone_rows.append(dict(epoch=epoch,checkpoint=str(milestone),checkpoint_sha256=provenance['checkpoint_sha256'],
                                       metrics={k:metrics['results']['trained']['metrics'][k] for k in METRICS}))
            del state
        history=audited_history(root,c,resolved['dataset_size'])
        row=dict(name=name+'-seed'+str(c['seed']),checkpoint=str(path),checkpoint_sha256=digest(path),
                 architecture=c.get('backbone','resnet9'),seed=c['seed'],config=c,
                 augmentation=augmentation_metadata(c),pooling=c.get('readout','max2'),training_horizon=c['epochs'],
                 checkpoint_epoch=checkpoint['epoch']+1,global_step=checkpoint['global_step'],
                 parameters=sum(p.numel() for p in encoder.parameters()),
                 optimizer_lrs=[g['lr'] for g in checkpoint['optimizer_states'][0]['param_groups']],
                 scheduler_state=checkpoint.get('lr_schedulers'),configuration_evidence=evidence,
                 initialization_log=dict(path=str(log),line=text[:seed.start()].count('\n')+1,seed=int(seed.group(1))),
                 milestones=milestone_rows,history_audit=history,training_job_id=job,
                 metrics_paths=[str(root/'metrics/metrics.csv')])
        if checkpoint['global_step']!=history['optimizer_steps']:raise ValueError('Final step count differs')
        rows.append(row)
        resource=json.loads((root/'resources.json').read_text())
        if sorted(map(int,resource['epochs']))!=list(range(1,c['epochs']+1)):raise ValueError('Incomplete resource epochs')
        wall=sum(v['seconds'] for v in resource['epochs'].values())
        if not math.isclose(wall,resource['training_wall_seconds'],rel_tol=1e-12):raise ValueError('Resource wall sum differs')
        images=sum(v['images'] for v in resource['epochs'].values())
        resource_summary=dict(name=name,job_id=job,gpu=resource['gpu'],epochs=c['epochs'],
                              training_wall_hours=wall/3600,images=images,images_per_second=images/wall,
                              view_images=images*(c['global_views']+c['local_views']),
                              input_pixels=images*(c['global_views']*c.get('global_size',96)**2+c['local_views']*c['local_size']**2),
                              peak_allocated_gib=resource['peak_allocated_gib'],peak_reserved_gib=resource['peak_reserved_gib'],
                              history_audit=history,retained_milestone_count=len(milestone_rows))
        resources.append(resource_summary)
        comparison=json.loads((root/'comparison.json').read_text())
        baseline=comparison['matched_control'];final=milestone_rows[-1]['metrics']
        comparisons.append(dict(name=name,control_name=control_name,control={k:baseline[k] for k in METRICS},
                                final=final,delta={k:final[k]-baseline[k] for k in METRICS},
                                epoch200_of_this_schedule=next(r['metrics'] for r in milestone_rows if r['epoch']==200),
                                milestones=milestone_rows))
        print(name,'strict checkpoint/LR/loss/resource/milestone audit passed',flush=True)
        del checkpoint,encoder
    manifest=dict(schema=2,original_manifest=str(original),original_manifest_sha256=digest(original),
                  audit_job_id=os.environ['SLURM_JOB_ID'],created_utc=datetime.now(timezone.utc).isoformat(),checkpoints=rows)
    atomic_json(output/'final-checkpoints.json',manifest)
    ids=[203177,203186,203199,203200,203201,203227,203228,203253,203254,203255]
    command=['sacct','-j',','.join(map(str,ids)),'--format=JobID,State,ElapsedRaw,AllocCPUS,AllocTRES,ExitCode','-P']
    accounting=[r for r in csv.DictReader(subprocess.check_output(command,text=True).splitlines(),delimiter='|') if '.' not in r['JobID']]
    def totals(entries):
        return dict(cpu_allocated_hours=sum(int(r['ElapsedRaw'])*int(r['AllocCPUS'] or 0) for r in entries)/3600,
                    gpu_allocated_hours=sum(int(r['ElapsedRaw'])*int((re.search(r'(?:^|,)gres/gpu=(\d+)',r['AllocTRES']) or [None,'0'])[1]) for r in entries)/3600)
    training_ids={str(job) for _,job,_ in RUNS}
    training=[r for r in accounting if r['JobID'] in training_ids]
    overhead=[r for r in accounting if r['JobID'] not in training_ids]
    if any(r['State']!='COMPLETED' or r['ExitCode']!='0:0' for r in training):raise ValueError('Training allocation not successfully complete')
    allocation_by_id={int(r['JobID']):r for r in training}
    for r in resources:r['gpu_allocation_hours']=int(allocation_by_id[r['job_id']]['ElapsedRaw'])/3600
    result=dict(status='passed',scope='Three training jobs plus their known preflight/provenance/archival/cancelled attempts; downstream allocations accounted separately',
                audit_job_id=os.environ['SLURM_JOB_ID'],training=resources,comparisons=comparisons,
                allocations=accounting,training_allocation_totals=totals(training),overhead_allocation_totals=totals(overhead),
                combined_allocation_totals=totals(accounting),audit_allocation_note='Current final-audit allocation is separate; collect its terminal sacct after this process exits',
                pixel_work_proxy_correction='Earlier launch notes incorrectly used5/3; correct per-step backbone input-pixel ratio is(2*96²+6*48²)/(2*96²+2*48²)=7/5=1.4. Training configurations unchanged.')
    atomic_json(output/'final-training-audit.json',result)
    lines=['# Final checkpoint and training audit','',f"Auditjob{os.environ['SLURM_JOB_ID']}: all14 final checkpoint identities/configurations verified; historical11 unchanged. Full evidence: [final-checkpoints.json](final-checkpoints.json).",'',
           '| Recipe | Native matched200 control five-fold / macro / LOO | New endpoint five-fold / macro / LOO | Δ five-fold / macro / LOO (pp) |',
           '|---|---|---|---|']
    fmt=lambda m:' / '.join(f'{100*m[k]:.2f}%' for k in METRICS)
    for r in comparisons:lines.append(f"| {r['name']} | {fmt(r['control'])} | {fmt(r['final'])} | "+' / '.join(f"{100*r['delta'][k]:+.2f}" for k in METRICS)+' |')
    lines+=['','The400-epoch runs preserve initialization seeds and recipe settings but plan a400-epoch cosine horizon; their epoch200 checkpoint is not a completed200-epoch cosine run. These are single-seed schedule-horizon comparisons, not isolated duration-only causal estimates. Native readouts are GAP320 for ResNet18 and max2/1024 for ResNet9. No hidden leaderboard gain is inferred.','',
            '| Recipe | Retained milestones | Measured train hours | Allocated GPU hours | Images/s | Peak allocation/reservation GiB |',
            '|---|---:|---:|---:|---:|---:|']
    for r in resources:lines.append(f"| {r['name']} | {r['retained_milestone_count']} | {r['training_wall_hours']:.3f} | {r['gpu_allocation_hours']:.3f} | {r['images_per_second']:.2f} | {r['peak_allocated_gib']:.3f} / {r['peak_reserved_gib']:.3f} |")
    lines+=['','All training used NVIDIA RTX A6000. Actual LR histories cover23200/23200/11600 consecutive optimizer steps, all400/400/200 epochs and finite logged loss components; the pinned total-loss formula was independently checked. All8/8/4 retained milestones load with exact configuration/epoch/step/evaluation provenance.', '',
            result['pixel_work_proxy_correction'],'','Full per-step history checks, epoch200-of400 metrics, checkpoint hashes, allocation overhead and raw sacct evidence: [final-training-audit.json](final-training-audit.json). Original audit history and checkpoint weights were not modified.']
    (output/'FINAL-TRAINING-AUDIT.md').write_text('\n'.join(lines)+'\n')


if __name__=='__main__':main()
