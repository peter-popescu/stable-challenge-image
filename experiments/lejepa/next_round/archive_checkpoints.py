"""Archive authorized completed milestones without altering final checkpoint identities."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path

from ..sweep import atomic_json, digest
from .storage_guard import require_free_space


def equal_metadata(left,right,torch):
    if isinstance(left,torch.Tensor):
        if not isinstance(right,torch.Tensor) or left.dtype!=right.dtype or left.shape!=right.shape or not torch.equal(left,right):
            raise ValueError('Checkpoint tensor changed during archival')
    elif isinstance(left,dict):
        if not isinstance(right,dict) or left.keys()!=right.keys():raise ValueError('Checkpoint mapping changed')
        for key,value in left.items():equal_metadata(value,right[key],torch)
    elif isinstance(left,(list,tuple)):
        if type(left)!=type(right) or len(left)!=len(right):raise ValueError('Checkpoint sequence changed')
        for a,b in zip(left,right):equal_metadata(a,b,torch)
    elif left!=right:
        raise ValueError('Checkpoint metadata changed during archival')


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--plan',type=Path,required=True)
    parser.add_argument('--apply',action='store_true')
    args=parser.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):parser.error('Archive loading and verification require Slurm')
    import torch
    from ..model import load_encoder
    torch.set_num_threads(int(os.environ.get('SLURM_CPUS_PER_TASK','2')))
    output=args.plan.parent/'checkpoint-archives.json'
    plan=json.loads(args.plan.read_text())
    result=json.loads(output.read_text()) if output.exists() else dict(plan=str(args.plan),job_id=os.environ['SLURM_JOB_ID'],entries=[])
    done={entry['original_path']:entry for entry in result['entries']}
    for text in plan['paths']:
        path=Path(text)
        if path.name=='last.ckpt' or not path.name.startswith('retained-epoch-') or '.weights.' in path.name:
            raise ValueError('Plan contains a final reference or nonmilestone checkpoint')
        if any('test' in part or 'preflight' in part or part=='next-round' for part in path.parts):
            raise ValueError('Plan contains a temporary or new-round checkpoint')
        archive=path.with_name(path.stem+'.weights.ckpt')
        tombstone=Path(str(path)+'.archived.json')
        if not path.exists() and text in done and done[text]['status']=='archived':
            if digest(archive)!=done[text]['archive_sha256']:raise ValueError('Existing archive identity changed')
            continue
        require_free_space(archive,512*1024**2)
        original_sha=digest(path)
        checkpoint=torch.load(path,map_location='cpu',weights_only=False)
        if 'optimizer_states' not in checkpoint:raise ValueError('Milestone has no optimizer state to remove')
        compact={k:v for k,v in checkpoint.items() if k!='optimizer_states'}
        annotation=dict(inference_only=True,original_path=text,original_sha256=original_sha,
                        removed_keys=['optimizer_states'],scheduler_state_preserved=True)
        if 'archival' in compact:raise ValueError('Checkpoint already has an archival annotation')
        compact['archival']=annotation
        entry=dict(original_path=text,original_sha256=original_sha,original_bytes=path.stat().st_size,
                   archive_path=str(archive),removed_keys=['optimizer_states'],
                   checkpoint_epoch=checkpoint['epoch']+1,config=checkpoint['hyper_parameters']['config'],
                   status='inventoried')
        if not args.apply:
            print(json.dumps(entry),flush=True)
            continue
        if archive.exists():raise ValueError('Archive already exists without a completed mapping')
        temporary=Path(str(archive)+'.tmp')
        torch.save(compact,temporary)
        restored=torch.load(temporary,map_location='cpu',weights_only=False)
        equal_metadata({k:v for k,v in checkpoint.items() if k!='optimizer_states'},
                       {k:v for k,v in restored.items() if k!='archival'},torch)
        equal_metadata(annotation,restored['archival'],torch)
        # Strict model loading additionally checks the public frozen encoder adapter.
        before=load_encoder(path)
        after=load_encoder(temporary)
        equal_metadata(before.state_dict(),after.state_dict(),torch)
        equal_metadata(before.experiment_config,after.experiment_config,torch)
        temporary.replace(archive)
        entry.update(archive_sha256=digest(archive),archive_bytes=archive.stat().st_size,
                     encoder_exactly_verified=True,nonoptimizer_metadata_exactly_verified=True,
                     status='archive_verified',job_id=os.environ['SLURM_JOB_ID'])
        entry['saved_bytes']=entry['original_bytes']-entry['archive_bytes']
        # Preserve the historical checkpoint/hash fields; add a separate mapping.
        updated=[]
        for provenance in path.parent.parent.rglob('provenance.json'):
            if 'source' in provenance.parts:continue
            saved=json.loads(provenance.read_text())
            source=saved.get('checkpoint')
            if source and Path(source).resolve()==path.resolve():
                if saved.get('checkpoint_sha256')!=original_sha:raise ValueError('Historical provenance hash differs')
                saved['archived_checkpoint']={k:entry[k] for k in ['original_path','original_sha256','archive_path','archive_sha256']}
                atomic_json(provenance,saved);updated.append(str(provenance))
        entry['updated_adjacent_provenance']=updated
        atomic_json(tombstone,entry)
        result['entries'].append(entry);atomic_json(output,result)
        path.unlink()
        entry['status']='archived';atomic_json(tombstone,entry)
        result.update(saved_bytes=sum(e['saved_bytes'] for e in result['entries']),
                      completed_entries=len(result['entries']),updated_utc=datetime.now(timezone.utc).isoformat())
        atomic_json(output,result)
        print(text,'->',archive,'saved',entry['saved_bytes'],flush=True)
        del checkpoint,compact,restored,before,after
    result['status']='completed' if args.apply else 'inventory_only'
    atomic_json(output,result)


if __name__=='__main__':main()
