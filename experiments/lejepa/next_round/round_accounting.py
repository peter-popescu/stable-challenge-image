"""Metadata-only parent-allocation accounting for this round; safe to refresh."""
from collections import defaultdict
from datetime import datetime, timezone
import getpass
import json
import os
from pathlib import Path
import re
import subprocess


ROOT=Path('experiment_outputs/next-round')
IDENTIFIER=re.compile(r'(?<![A-Za-z0-9])(?:203|204|206)\d{3}(?![A-Za-z0-9])')
KNOWN={203177,203186,203199,203200,203201,203220,203221,203226,203227,
       203228,203232,203253,203254,203255,203398,203400,203401,203433,
       206582,206588,206596,206606}
TRAINING={203253,203254,203255}
EXCLUDED={203388:'Unrelated interactive zsh allocation; explicitly outside this experiment round'}
TERMINAL={'COMPLETED','FAILED','CANCELLED','TIMEOUT','OUT_OF_MEMORY','NODE_FAIL','BOOT_FAIL','DEADLINE','REVOKED','PREEMPTED'}


def discover():
    evidence=defaultdict(set)
    for job in KNOWN:evidence[job].add('explicit task-owned job inventory')
    def scan(path):
        text=path.read_text()
        for match in IDENTIFIER.finditer(text):
            value=int(match.group())
            if value>=203177:evidence[value].add(str(path))
    for path in Path('experiments/lejepa/next_round').glob('*.md'):scan(path)
    for directory,dirs,files in os.walk(ROOT,followlinks=True):
        dirs[:]=[d for d in dirs if d not in {'source','data','cache','conditions','__pycache__','tests'} and not d.startswith('preflight-work-')]
        for filename in files:
            path=Path(directory)/filename
            if filename.endswith('.log'):
                match=re.search(r'-(20[346]\d{3})\.log$',filename)
                if match:evidence[int(match.group(1))].add(str(path))
            elif filename.endswith('.json') and (filename.endswith(('jobs.json','status.json','plan.json')) or filename in {'launch.json','final-audit-allocation.json'}):
                if filename.startswith('jobs-'):continue # scheduler snapshots also contain unrelated allocations
                scan(path)
    for path in Path('experiment_outputs/external-eval/probes').glob('cache-prune*.log'):
        match=re.search(r'-(\d{6})\.log$',path.name)
        if match:evidence[int(match.group(1))].add(str(path))
    return evidence


def category(job,command,state,gpus):
    if job in TRAINING:return 'training','training'
    if job==203433:return 'recovery','feature_extraction'
    if job in {203199,203200,203201,203232,203398,203400,206588} or state in TERMINAL-{'COMPLETED'}:
        return 'recovery','preflight' if job==203186 else 'task_recovery'
    lower=command.lower()
    if any(token in lower for token in ['cache-prune','prune_pet_cache','archive_checkpoints','cleanup','storage_migration','network','storage_probe']):
        return 'recovery','storage_or_startup'
    if job in {203186,203209,203228} or any(token in lower for token in ['pytest','fixture','_test','preflight','test_followup','test_round','test_frozen','test_external','test_downstream','nyu_test','network_probe']):
        return 'preflight','preflight'
    if gpus or any(token in lower for token in [' extract','external-extract']):return 'feature_extraction','feature_extraction'
    if any(token in lower for token in ['external_download',' clevr_light prepare','clevr_light prepare','external_run prepare','clevr-prepare','cifar100-prepare','dtd-prepare']):
        return 'data_preparation_storage','data_preparation_storage'
    if any(token in lower for token in [' cpu','external-cpu','followup global','frozen cpu','downstream.run','appearance','segmentation']):
        return 'downstream_cpu','downstream_cpu'
    return 'report','report_or_audit'


def total(rows):
    return dict(jobs=len(rows),gpu_allocation_hours=sum(r['gpu_allocation_hours_so_far'] for r in rows),
                cpu_allocation_hours=sum(r['cpu_allocation_hours_so_far'] for r in rows))


def main():
    evidence=discover()
    command=['sacct','--user='+getpass.getuser(),'--allocations','-j',','.join(map(str,sorted(evidence))), '--json']
    result=json.loads(subprocess.check_output(command,text=True))
    if result.get('errors'):raise RuntimeError(result['errors'])
    allocations=[];excluded=[dict(job_id=k,reason=v) for k,v in EXCLUDED.items()];found=set()
    for source in result['jobs']:
        job=int(source['job_id']);found.add(job)
        if job in EXCLUDED:continue
        submit=source.get('submit_line','')
        if not any(token in submit for token in ['next_round','next-round','external-eval/probes/cache-prune']) and job not in KNOWN:
            excluded.append(dict(job_id=job,reason='No task module/output-path match in SubmitLine and not in explicit task-owned inventory'))
            continue
        allocated=source['tres']['allocated']
        cpus=sum(r['count'] for r in allocated if r['type']=='cpu')
        gpus=sum(r['count'] for r in allocated if r['type']=='gres' and r['name']=='gpu')
        elapsed=int(source['time']['elapsed']);state=source['state']['current'][0]
        primary,intended=category(job,submit,state,gpus)
        allocations.append(dict(job_id=job,name=source['name'],state=state,exit_code=source['exit_code'],
                                elapsed_seconds=elapsed,allocated_cpus=cpus,allocated_gpus=gpus,
                                gpu_allocation_hours_so_far=elapsed*gpus/3600,
                                cpu_allocation_hours_so_far=elapsed*cpus/3600,
                                category=primary,intended_category=intended,
                                terminal=state in TERMINAL,
                                submit_line=submit,time=source['time'],partition=source['partition'],
                                discovery_evidence=sorted(evidence[job])))
    allocations.sort(key=lambda r:r['job_id'])
    grouped={name:total([r for r in allocations if r['category']==name]) for name in
             ['training','preflight','feature_extraction','downstream_cpu','data_preparation_storage','report','recovery']}
    live=[r for r in allocations if not r['terminal']]
    stopped=next(r for r in allocations if r['job_id']==203433)
    output=dict(schema=1,as_of_utc=datetime.now(timezone.utc).isoformat(),
                scope='All evidenced task-owned parent Slurm allocations for this round; old encoder-training rounds excluded; user unrelated interactive job203388 excluded.',
                parent_allocations_only=True,allocation_rule='ElapsedRaw×allocated GPU/CPU count; sacct --allocations; no batch/step double counting. CPU allocation-hours count reserved CPUs including GPU jobs, and are separate from GPU hours.',
                final=not live,provisional_job_ids=[r['job_id'] for r in live],
                totals_so_far=total(allocations),terminal_totals=total([r for r in allocations if r['terminal']]),
                categories=grouped,allocations=allocations,excluded_jobs=excluded,
                missing_job_ids=sorted(set(evidence)-found-set(EXCLUDED)),
                stopped_full_clevr=dict(job_id=203433,gpu_allocation_hours=stopped['gpu_allocation_hours_so_far'],
                                        cpu_allocation_hours=stopped['cpu_allocation_hours_so_far'],
                                        note='Stopped full85k extraction cost is included in recovery. Original completed encoder cache was retained/reused for the lightweight subset; do not claim all1.48h was irrecoverably wasted. Full-probe pending children were cancelled before allocation.'),
                query_command=command,refresh_command='.venv/bin/python -m experiments.lejepa.next_round.round_accounting')
    destination=ROOT/'final-round-accounting.json'
    temporary=destination.with_suffix('.json.tmp');temporary.write_text(json.dumps(output,indent=2)+'\n');temporary.replace(destination)
    print(json.dumps(dict(path=str(destination),final=output['final'],totals=output['totals_so_far'],categories=grouped,
                          provisional=output['provisional_job_ids'],missing=output['missing_job_ids'])))


if __name__=='__main__':main()
