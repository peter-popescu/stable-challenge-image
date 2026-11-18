"""Exact full-batch forward/backward preflight for three authorized recipes."""
import argparse
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--repository-storage',action='store_true',help='Store full preflight artifacts in repository after capacity recovery')
    args=parser.parse_args()
    if not os.environ.get('SLURM_JOB_ID'): raise RuntimeError('Slurm only')
    root=Path('experiment_outputs/next-round/training')
    from .storage_guard import require_free_space
    from ..sweep import atomic_json
    # Full ONNX files, test checkpoints and caches stay on this allocated node.
    # Home receives small status/provenance/log copies only.
    job=os.environ['SLURM_JOB_ID']
    if args.repository_storage:
        scratch=root/f'preflight-work-{job}'
    else:
        if not Path('/ltmp').is_dir():
            raise RuntimeError('This preflight requires the authorized node-local /ltmp filesystem')
        scratch=Path('/ltmp/ppopescu/stable-challenge-image-next-round')/f'preflight-{job}'
    local_storage=require_free_space(scratch)
    report_storage=require_free_space(root,256*1024**2)
    scratch.mkdir(parents=True,exist_ok=True)
    for key,relative in [('SPT_CACHE_DIR','stable-pretraining'),('HF_HOME','huggingface'),
                         ('XDG_CACHE_HOME','xdg'),('MPLCONFIGDIR','matplotlib'),
                         ('TMPDIR','tmp')]:
        path=scratch/'cache'/relative
        path.mkdir(parents=True,exist_ok=True)
        os.environ[key]=str(path)
    os.environ['PYTHONDONTWRITEBYTECODE']='1'
    sys.dont_write_bytecode=True
    staging=dict(job_id=job,hostname=socket.gethostname(),status='tests',
                 scratch_root=str(scratch),local_storage=local_storage,
                 report_storage=report_storage,storage_mode='repository' if args.repository_storage else 'node_local',
                 full_artifacts='retained under scratch_root; only small reports copied to recipe preflight directories')
    staging_path=root/f'preflight-staging-{job}.json'
    atomic_json(staging_path,staging)

    def copy_small(source,destination,limit=1024**2):
        if not source.exists(): return
        require_free_space(root,256*1024**2)
        destination.parent.mkdir(parents=True,exist_ok=True)
        # Preserve full logs locally and only their bounded tail in home.
        if source.stat().st_size>limit:
            with source.open('rb') as handle:
                handle.seek(-limit,2)
                destination.write_bytes(b'[truncated; full artifact retained on node-local scratch]\n'+handle.read())
        else:
            shutil.copy2(source,destination)

    tests_log=scratch/'tests.log'
    tests_xml=scratch/'tests.xml'
    with tests_log.open('w') as handle:
        tests=subprocess.run([sys.executable,'-m','pytest','-q','tests/test_lejepa.py','tests/test_architecture_pooling.py','tests/test_augmentation_study.py','tests/test_cross_validate.py',
                              '--basetemp',str(scratch/'tests'),
                              '--junitxml',str(tests_xml)],stdout=handle,stderr=subprocess.STDOUT)
    copy_small(tests_log,root/f'preflight-tests-{job}.log')
    copy_small(tests_xml,root/f'preflight-tests-{job}.xml')
    staging['tests_returncode']=tests.returncode
    staging['status']='tests_passed' if tests.returncode==0 else 'tests_failed'
    atomic_json(staging_path,staging)
    tests.check_returncode()
    audit=json.loads(Path('experiment_outputs/next-round/provenance/checkpoints.json').read_text())['checkpoints']
    refs={r['name']:r for r in audit}
    configs=Path('experiments/lejepa/next_round')
    for name,reference in [('resnet18-tuned-400','resnet18-tuned-200-seed42'),('resnet9-original-400','original-200-seed42'),('resnet9-more-views-200',None)]:
        config=configs/(name+'.json')
        c=json.loads(config.read_text())
        if reference:
            expected=dict(refs[reference]['config'],epochs=400,resource_metrics=True)
            if c!=expected: raise ValueError(f'Historical configuration mismatch: {name}')
        else:
            expected=json.loads(Path('experiments/lejepa/augmentations/more-views.json').read_text())
            expected['resource_metrics']=True
            if c!=expected: raise ValueError('Prepared more-views configuration mismatch')
        local_output=scratch/name
        staging.update(status='preflight',recipe=name)
        atomic_json(staging_path,staging)
        try:
            subprocess.run([sys.executable,'-u','-m','experiments.lejepa.next_round.preflight','--config',str(config),'--output',str(local_output)],check=True)
        finally:
            for file in ['preflight.json','validator.log']:
                copy_small(local_output/file,root/name/'preflight'/file)
    staging.update(status='passed',recipe=None)
    atomic_json(staging_path,staging)


if __name__=='__main__': main()
