"""Submit an approved bounded external evaluation pipeline.

Run on login (scheduler commands + small JSON only). Dataset verification jobs
remain held until the corresponding transfer-only curl succeeds; release those
jobs explicitly, never before archives are complete. CLEVR requires the authorized scratch symlink and its60GiB free-space guard.
"""
import argparse
import json
from pathlib import Path
import subprocess

ROOT=Path('experiment_outputs/next-round/external')
LAUNCHER='experiments/lejepa/downstream/slurm.sh'
TRAINS={'resnet18-tuned-400':'203253','resnet9-original-400':'203254','resnet9-more-views-200':'203255'}


def submit(name,args,*,gpu=False,dependency=None,hold=False,hours=8):
    cmd=['sbatch','--parsable','--partition='+('gpus' if gpu else 'compute'),'--nodes=1','--ntasks=1',
         '--cpus-per-task='+('4' if gpu else '8'),'--mem='+('24G' if gpu else '32G'),
         '--time='+str(hours)+':00:00','--job-name='+name,'--output='+str(ROOT/(name+'-%j.log'))]
    if gpu:cmd+=['--gres=gpu:1','--constraint=titan_rtx|rtx_a6000|l40|l40s|gtx_2080_ti']
    if dependency:cmd+=['--dependency=afterok:'+':'.join(dependency)]
    if hold:cmd+=['--hold']
    cmd += [LAUNCHER,*args]
    jid=subprocess.check_output(cmd,text=True).strip().split(';')[0]
    return dict(job_id=jid,command=cmd,held=hold,dependencies=dependency or [])


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--dataset',required=True,choices=['dtd','cifar100','clevr'])
    a=p.parse_args();plan=ROOT/(a.dataset+'-jobs.json')
    if plan.exists():raise ValueError('Existing job plan; inspect active jobs instead of resubmitting')
    rows={};record=dict(dataset=a.dataset,jobs=rows)
    def save():plan.write_text(json.dumps(record,indent=2)+'\n')
    rows['prepare']=submit(a.dataset+'-prepare',['experiments.lejepa.next_round.external_download','--dataset',a.dataset,'--approved-large-download'],hold=True,hours=4 if a.dataset=='clevr' else 1);save()
    ready=rows['prepare']['job_id']
    rows['extract']=submit(a.dataset+'-existing-extract',['experiments.lejepa.next_round.external_run','extract','--dataset',a.dataset],gpu=True,dependency=[ready]);save()
    rows['cpu']=submit(a.dataset+'-existing-probe',['experiments.lejepa.next_round.external_run','cpu','--dataset',a.dataset],dependency=[rows['extract']['job_id']],hours=24);save()
    for run,train in TRAINS.items():
        key=run+'-extract'
        rows[key]=submit(a.dataset+'-'+run+'-extract',['experiments.lejepa.next_round.followup','external-extract','--run',run,'--dataset',a.dataset],gpu=True,dependency=[train,ready]);save()
        rows[run+'-cpu']=submit(a.dataset+'-'+run+'-probe',['experiments.lejepa.next_round.followup','external-cpu','--run',run,'--dataset',a.dataset],dependency=[rows[key]['job_id']],hours=12);save()
    print(json.dumps({k:v['job_id'] for k,v in rows.items()}))


if __name__=='__main__':main()
