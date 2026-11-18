"""Launch the fixed3500-image CLEVR branch only; preserve full-run artifacts."""
import json
from pathlib import Path
import subprocess

ROOT=Path('experiment_outputs/next-round/external')
TRAINS={'resnet18-tuned-400':'203253','resnet9-original-400':'203254','resnet9-more-views-200':'203255'}


def submit(name,stage,dependencies,run=None,gpu=False,prepare=False):
    cpus=4 if gpu else (2 if prepare else 8)
    mem='24G' if gpu else ('8G' if prepare else '16G')
    hours='02:00:00' if gpu or not prepare else '00:30:00'
    cmd=['sbatch','--parsable','--partition='+('gpus' if gpu else 'compute'),'--nodes=1','--ntasks=1',
         '--cpus-per-task='+str(cpus),'--mem='+mem,'--time='+hours,'--job-name='+name,
         '--output='+str(ROOT/(name+'-%j.log')),'--dependency=afterok:'+':'.join(dependencies)]
    if gpu:cmd+=['--gres=gpu:1','--constraint=titan_rtx|rtx_a6000|l40|l40s|gtx_2080_ti']
    cmd+=['experiments/lejepa/downstream/slurm.sh','experiments.lejepa.next_round.clevr_light',stage]
    if run:cmd+=['--run',run]
    jid=subprocess.check_output(cmd,text=True).strip().split(';')[0]
    return dict(job_id=jid,command=cmd,dependencies=dependencies)


def main():
    plan=ROOT/'clevr-light-jobs.json'
    if plan.exists():raise ValueError('Existing lightweight job plan; inspect it instead of resubmitting')
    record=dict(protocol='Fixed1000train/500val/2000test;seed42;label-independent;NOT officialVTAB1k',jobs={})
    jobs=record['jobs']
    def save():plan.write_text(json.dumps(record,indent=2)+'\n')
    jobs['prepare']=submit('clevr-light-prepare','prepare',['204000'],prepare=True);save()
    jobs['extract']=submit('clevr-light-existing-extract','extract',[jobs['prepare']['job_id']],gpu=True);save()
    jobs['cpu']=submit('clevr-light-existing-probe','cpu',[jobs['extract']['job_id']]);save()
    for run,train in TRAINS.items():
        key=run+'-extract'
        jobs[key]=submit('clevr-light-'+run+'-extract','extract',[train,jobs['prepare']['job_id']],run=run,gpu=True);save()
        jobs[run+'-cpu']=submit('clevr-light-'+run+'-probe','cpu',[jobs[key]['job_id']],run=run);save()
    print(json.dumps({k:v['job_id'] for k,v in jobs.items()}))


if __name__=='__main__':main()
