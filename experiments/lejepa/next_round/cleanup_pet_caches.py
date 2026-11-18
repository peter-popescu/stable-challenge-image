"""Authorized pruning of completed, reproducible Pet cache arrays only.

Preserve checkpoints/probes, metrics, histories, figures, split IDs and original
cache hashes. No recursive deletion. Refuse incomplete evaluations or active
segmentation consumers. Run inventory, hashing and deletion in Slurm.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time
from ..sweep import atomic_json,digest

ROOT=Path('experiment_outputs/external-eval/probes')
NAMES=('mid.npy','final.npy','masks.npy')


def active_jobs():
    queue=subprocess.check_output(['squeue','--noheader','--user',os.environ['USER'],'--format=%i|%j|%T'],text=True)
    rows=[]
    for line in queue.splitlines():
        jid,name,state=line.split('|',2)
        if jid==os.environ.get('SLURM_JOB_ID'):continue
        details=subprocess.check_output(['scontrol','show','job',jid,'--oneliner'],text=True)
        submit=subprocess.check_output(['sacct','--jobs',jid,'--noheader','--parsable2','--format=JobID,SubmitLine%4096'],text=True)
        # Both explicit Pet paths and the reusable segmentation entrypoint are
        # consumers. Pending jobs are checked too, since they can start mid-prune.
        if any(s in (details+' '+submit).lower() for s in ['segmentation','external-eval/probes','downstream.pet']):
            raise RuntimeError(f'Potential active/pending Pet cache consumer {jid}: {name}')
        rows.append(dict(job_id=jid,name=name,state=state,scontrol=details.strip(),submit_line=submit.strip()))
    return rows


def completed(directory):
    cache=directory/'cache';meta=cache/'cache.json'
    if cache.is_symlink() or not meta.is_file():raise ValueError(f'No ordinary cache metadata: {directory}')
    saved=json.loads(meta.read_text())
    root=json.loads((directory/'results.json').read_text())
    artifacts=[]
    for layer in ['mid','final']:
        for name in ['results.json','probe.pt','history.json','examples.png']:
            path=directory/layer/name
            if not path.is_file() or path.stat().st_size==0:raise ValueError(f'Incomplete preserved artifact {path}')
            artifacts.append(path)
        result=json.loads((directory/layer/'results.json').read_text())
        if root.get(layer)!=result or not result.get('test') or not result.get('selection'):
            raise ValueError(f'Incomplete/inconsistent root result {directory}/{layer}')
    artifacts += [directory/'results.json',meta]
    if not saved.get('splits') or set(saved['files'])!=set(NAMES):raise ValueError('Unexpected cache schema')
    return saved,{str(p):digest(p) for p in artifacts}


def inventory():
    rows=[]
    # scandir bounds the inventory to one model/cache level.
    for entry in sorted(os.scandir(ROOT),key=lambda e:e.name):
        if not entry.is_dir(follow_symlinks=False):continue
        directory=Path(entry.path)
        if not (directory/'cache').exists():continue
        saved,artifacts=completed(directory)
        for item in os.scandir(directory/'cache'):
            if item.name not in NAMES:continue
            if not item.is_file(follow_symlinks=False):raise ValueError('Array path is not ordinary file')
            st=item.stat(follow_symlinks=False)
            rows.append(dict(path=str(Path(item.path)),bytes=st.st_size,mtime_ns=st.st_mtime_ns,
                             inode=st.st_ino,original_sha256=saved['files'][item.name],
                             preserved_artifacts_sha256=artifacts))
    return rows


def main():
    p=argparse.ArgumentParser(__doc__);mode=p.add_mutually_exclusive_group(required=True)
    mode.add_argument('--plan',action='store_true');mode.add_argument('--apply',type=Path)
    p.add_argument('--output',type=Path,default=ROOT/'cache-prune-plan.json');a=p.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):p.error('Inventory/hash/prune must run in Slurm')
    jobs=active_jobs()
    if a.plan:
        rows=inventory()
        plan=dict(schema=1,created_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
                  job_id=os.environ['SLURM_JOB_ID'],scope=str(ROOT),allowed_names=list(NAMES),
                  active_jobs=jobs,files=rows,total_bytes=sum(r['bytes'] for r in rows),
                  authorization='User explicitly approved reproducible completed Pet cache array cleanup; retain all other artifacts')
        atomic_json(a.output,plan);print(json.dumps(dict(plan=str(a.output),files=len(rows),bytes=plan['total_bytes'])),flush=True)
        return
    plan=json.loads(a.apply.read_text())
    if plan['scope']!=str(ROOT) or plan['allowed_names']!=list(NAMES):raise ValueError('Unexpected prune scope')
    current=inventory()
    if current!=plan['files']:raise ValueError('Prune inventory changed since plan')
    # Verify every byte hash before any deletion; unchanged metadata binds the
    # original run. All expensive reads run in this CPU allocation.
    for row in current:
        path=Path(row['path'])
        if path.name not in NAMES or path.parent.name!='cache' or path.parent.parent.parent!=ROOT:
            raise ValueError('Path escaped authorized array scope')
        if digest(path)!=row['original_sha256']:raise ValueError(f'Cache content differs: {path}')
    active_jobs()
    audit=a.apply.with_name(a.apply.stem+'-applied.json')
    result=dict(plan_sha256=digest(a.apply),job_id=os.environ['SLURM_JOB_ID'],active_jobs=jobs,
                bytes_reclaimed=0,deleted=[],preservation='all original cache.json, splits, root/layer metrics, probe weights, histories, figures retained')
    atomic_json(audit,result)
    grouped={}
    for row in current:
        path=Path(row['path']);st=path.stat()
        if st.st_size!=row['bytes'] or st.st_mtime_ns!=row['mtime_ns'] or st.st_ino!=row['inode']:
            raise ValueError('Array changed after hash verification')
        path.unlink()
        result['deleted'].append(row);result['bytes_reclaimed']+=row['bytes']
        atomic_json(audit,result)
        grouped.setdefault(path.parent,[]).append(row)
    for cache,rows in grouped.items():
        atomic_json(cache/'pruned.json',dict(evaluation_complete=True,job_id=os.environ['SLURM_JOB_ID'],
                    audit=str(audit),deleted_arrays=[dict(path=r['path'],bytes=r['bytes'],sha256=r['original_sha256']) for r in rows],
                    original_metadata_preserved=True,regenerate_only_for_explicit_new_probe_request=True))
        completed(cache.parent)
    result['status']='complete';atomic_json(audit,result)
    print(json.dumps(dict(audit=str(audit),files=len(current),bytes_reclaimed=result['bytes_reclaimed'])),flush=True)


if __name__=='__main__':main()
