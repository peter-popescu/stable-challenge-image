"""Publish allocated-GPU telemetry to the existing SPT filesystem dashboard.

Run as an overlapping step of an existing Slurm job. This imports no Torch,
allocates no CUDA memory, and never edits the experiment's metrics or source.
"""
import argparse
import csv
import importlib.util
import io
import json
import os
from pathlib import Path
import signal
import subprocess
import sysconfig
import time


def parse_gpu(text):
    rows = list(csv.reader(io.StringIO(text)))
    if len(rows) != 1 or len(rows[0]) != 6:
        raise ValueError('Expected exactly one allocated GPU')
    uuid, name, used, free, total, utilization = [v.strip() for v in rows[0]]
    used, free, total, utilization = map(float, [used, free, total, utilization])
    if min(used, free, total, utilization) < 0 or total <= 0:
        raise ValueError('Invalid GPU telemetry')
    return uuid, name, dict(gpu_memory_used_gib=used/1024,
                           gpu_memory_free_gib=free/1024,
                           gpu_memory_total_gib=total/1024,
                           gpu_memory_used_percent=100*used/total,
                           gpu_utilization_percent=utilization)


def sidecar_helpers():
    # Load this stdlib-only helper directly, like dashboard/bootstrap.py, to
    # avoid importing the package's Torch/Lightning training dependencies.
    path = Path(sysconfig.get_paths()['purelib']) / 'stable_pretraining/registry/_sidecar.py'
    spec = importlib.util.spec_from_file_location('spt_telemetry_sidecar', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--interval', type=float, default=10)
    p.add_argument('--samples', type=int, help='Optional bounded verification run')
    a = p.parse_args()
    if not os.environ.get('SLURM_JOB_ID') or not os.environ.get('SLURM_STEP_ID'):
        p.error('Run inside an srun step of the existing GPU allocation')
    if a.interval < 1 or (a.samples is not None and a.samples < 1):
        p.error('Require interval >= 1 second and positive sample count')
    gpu = os.environ.get('SLURM_STEP_GPUS', '')
    if not gpu or ',' in gpu:
        p.error('Require exactly one allocated GPU in SLURM_STEP_GPUS')
    root = Path(__file__).resolve().parents[1]
    output = a.output.resolve()
    if not output.is_relative_to(root / 'experiment_outputs'):
        p.error('Telemetry output must be under repository experiment_outputs')
    output.mkdir(parents=True, exist_ok=True)
    helpers = sidecar_helpers()
    job = os.environ['SLURM_JOB_ID']
    run_id = str(output.relative_to(root))
    sidecar = helpers.make_sidecar(
        run_id=run_id, run_dir=str(output), tags=['gpu-memory', 'telemetry', f'job-{job}'],
        hparams={'slurm.job_id': job, 'node': os.environ.get('SLURMD_NODENAME'),
                 'sample_interval_seconds': a.interval, 'gpu_index': gpu},
        notes='Device VRAM reported by nvidia-smi, sampled periodically. Step is elapsed seconds since monitoring began. This is not PyTorch allocated/reserved memory or a per-step peak; earlier usage is unavailable.')
    active = True

    def stop(signum, frame):
        nonlocal active
        active = False

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    start = time.monotonic()
    stats, count = {}, 0
    failed = False
    try:
        # Exclusive creation prevents two writers corrupting one telemetry run.
        with (output / 'metrics.csv').open('x', newline='', buffering=1) as handle:
            writer = None
            while active and (a.samples is None or count < a.samples):
                raw = subprocess.check_output([
                    'nvidia-smi', '-i', gpu,
                    '--query-gpu=uuid,name,memory.used,memory.free,memory.total,utilization.gpu',
                    '--format=csv,noheader,nounits'], text=True, timeout=10)
                uuid, name, metrics = parse_gpu(raw)
                elapsed = round(time.monotonic() - start, 3)
                if writer is None:
                    writer = csv.DictWriter(handle, fieldnames=['step', *metrics])
                    writer.writeheader()
                    sidecar['hparams'].update(gpu_uuid=uuid, gpu_name=name)
                writer.writerow(dict(step=elapsed, **metrics))
                handle.flush()
                sidecar['summary'] = metrics
                for key, value in metrics.items():
                    previous = stats.get(key)
                    stats[key] = dict(last=value, min=min(value, previous['min']) if previous else value,
                                      max=max(value, previous['max']) if previous else value,
                                      count=count+1)
                helpers.write_sidecar(output, sidecar)
                helpers.atomic_json_write(output / 'summary.json', dict(
                    schema_version=1, run_id=run_id, run_dir=str(output), updated_at=time.time(),
                    step=elapsed, epoch=None, metrics=stats))
                helpers.touch_heartbeat(output)
                if count == 0:
                    print(json.dumps(dict(gpu=name, uuid=uuid, output=str(output), **metrics)), flush=True)
                count += 1
                # Check termination every second so Slurm can end this step promptly.
                until = time.monotonic() + a.interval
                while active and time.monotonic() < until and (a.samples is None or count < a.samples):
                    time.sleep(min(1, max(0, until-time.monotonic())))
    except Exception:
        failed = True
        raise
    finally:
        sidecar.update(status='failed' if failed else 'completed', ended_at=time.time())
        if count:
            helpers.write_sidecar(output, sidecar)


if __name__ == '__main__':
    main()
