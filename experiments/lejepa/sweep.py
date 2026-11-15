"""Sequential, restartable LR/SIGReg screening within one Slurm allocation."""
import argparse
import csv
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
from datetime import datetime, timezone


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def configurations(base, spec):
    """Build the Cartesian grid with the matched control first."""
    if not 1 <= spec['screen_epochs'] <= base['epochs']:
        raise ValueError('Screen epochs must fit within the unchanged schedule horizon')
    if not 1 <= spec['retain_every'] <= spec['screen_epochs']:
        raise ValueError('Retention interval must fit within the screening budget')
    rates, weights = spec['learning_rates'], spec['sigreg_weights']
    for values in [rates, weights]:
        if not values or len(values) != len(set(values)) or not all(math.isfinite(v) and v > 0 for v in values):
            raise ValueError('Grid axes must have distinct positive finite values')
    if base['lr'] not in rates or base['lamb'] not in weights:
        raise ValueError('Grid must include the matched baseline control')
    pairs = list(itertools.product(rates, weights))
    pairs.sort(key=lambda pair: (pair != (base['lr'], base['lamb']), pair))
    runs = []
    for lr, lamb in pairs:
        c = dict(base, lr=lr, lamb=lamb)
        runs.append({'name': f'lr-{lr:g}_lambda-{lamb:g}', 'config': c})
    return runs


def compatible_result(path, model_hash, folds_hash, data_hash):
    """Only reuse CV artifacts produced from exactly these model/data/folds."""
    if not path.exists():
        return False
    r = json.loads(path.read_text())
    return (r['models']['trained']['sha256'] == model_hash and
            r['fold_manifest_sha256'] == folds_hash and r['data_sha256'] == data_hash)


def checkpoint_complete(path, config, budget):
    if not path.exists():
        return False
    import torch
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    if checkpoint['hyper_parameters']['config'] != config:
        raise ValueError(f'Checkpoint configuration mismatch: {path}')
    if checkpoint['epoch'] + 1 > budget:
        raise ValueError('Checkpoint exceeds screening budget; refusing to mix training durations')
    return checkpoint['epoch'] + 1 == budget


def run_command(command, log):
    print('Executing:', ' '.join(map(str, command)), 'log:', log, flush=True)
    with log.open('a') as f:
        f.write('\n' + datetime.now(timezone.utc).isoformat() + '\n')
        f.flush()
        subprocess.run([str(item) for item in command], stdout=f, stderr=subprocess.STDOUT, check=True)


def comparison(output, runs, statuses):
    rows = []
    control_folds = None
    control_path = output / runs[0]['name'] / 'validation/results.json'
    if statuses.get(runs[0]['name'], {}).get('status') == 'completed' and control_path.exists():
        r = json.loads(control_path.read_text())
        control_folds = [f['accuracy'] for f in r['results']['trained']['folds']]
    for run in runs:
        name = run['name']
        row = {'name': name, 'lr': run['config']['lr'], 'sigreg_weight': run['config']['lamb'],
               'status': statuses.get(name, {}).get('status', 'pending')}
        result_path = output / name / 'validation/results.json'
        if row['status'] == 'completed':
            r = json.loads(result_path.read_text())['results']['trained']
            m = r['metrics']
            row.update(outer_accuracy=m['accuracy'], macro_accuracy=m['macro_accuracy'],
                       fold_sd=m['fold_sd_accuracy'], official_local_loo=m['official_local_loo_accuracy'])
            if control_folds is not None:
                delta = [f['accuracy'] - b for f, b in zip(r['folds'], control_folds)]
                row['mean_paired_delta'] = sum(delta) / len(delta)
                row['paired_fold_deltas'] = json.dumps(delta)
        rows.append(row)
    fields = ['name', 'lr', 'sigreg_weight', 'status', 'outer_accuracy', 'macro_accuracy',
              'fold_sd', 'official_local_loo', 'mean_paired_delta', 'paired_fold_deltas']
    tmp = output / 'comparison.csv.tmp'
    with tmp.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(output / 'comparison.csv')
    ranked = sorted((r for r in rows if r['status'] == 'completed'),
                    key=lambda r: (-r['outer_accuracy'], -r['macro_accuracy'], r['name']))
    lines = ['# LR/SIGReg screening sweep', '',
             '100-epoch screens on a fixed 200-epoch schedule. One seed, fixed development folds.', '',
             '| Run | Outer accuracy | Macro accuracy | Official local LOO | Paired change vs 100-epoch control |',
             '|---|---:|---:|---:|---:|']
    for row in ranked:
        delta = f'{100*row["mean_paired_delta"]:+.2f} pp' if 'mean_paired_delta' in row else 'pending control'
        lines.append(f'| {row["name"]} | {row["outer_accuracy"]:.2%} | {row["macro_accuracy"]:.2%} | {row["official_local_loo"]:.2%} | {delta} |')
    lines += ['', f'{len(ranked)}/{len(runs)} configurations completed. See progress.json and per-run logs for remaining/failed runs.',
              'Fold SD and paired fold differences are descriptive, not confidence intervals.',
              'These folds are being used for hyperparameter selection, not an unbiased final estimate.',
              'Compare screening candidates at the same budget. The separate 200-epoch baseline scored 70.0% outer accuracy.',
              'No configurations are automatically extended to 200 epochs and no leaderboard submissions are made.']
    (output / 'REPORT.md').write_text('\n'.join(lines) + '\n')


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--base-config', type=Path, default=Path(__file__).with_name('config.json'))
    p.add_argument('--spec', type=Path, default=Path(__file__).with_name('sweep.json'))
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--folds', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):
        p.error('Run the sweep within a single Slurm GPU allocation')
    base, spec = json.loads(a.base_config.read_text()), json.loads(a.spec.read_text())
    runs = configurations(base, spec)
    a.output.mkdir(parents=True, exist_ok=True)
    if not a.folds.is_file():
        p.error('Use the existing fixed fold manifest; it must already exist')
    data_hash, folds_hash = digest(a.data), digest(a.folds)
    if json.loads(a.folds.read_text())['data_sha256'] != data_hash:
        p.error('Fold manifest belongs to another dataset')
    sources = list(Path(__file__).parent.glob('*.py')) + [Path(f) for f in ['evaluate.py', 'training_data.py', 'export_onnx.py', 'pyproject.toml', 'uv.lock']]
    plan = {'schema_version': 1, 'base_config': base, 'spec': spec, 'runs': runs,
            'data': str(a.data.resolve()), 'data_sha256': data_hash,
            'fold_manifest': str(a.folds.resolve()), 'fold_manifest_sha256': folds_hash,
            'source_hashes': {str(path.resolve().relative_to(Path.cwd().resolve())): digest(path) for path in sorted(sources)}}
    plan_path = a.output / 'plan.json'
    if plan_path.exists():
        if json.loads(plan_path.read_text()) != plan:
            p.error('Sweep plan/source/data changed; refuse to mix experiments in the existing output')
    else:
        atomic_json(plan_path, plan)
        shutil.copytree(Path(__file__).parent, a.output / 'source/lejepa',
                        ignore=shutil.ignore_patterns('__pycache__'))
        for file in ['evaluate.py', 'training_data.py', 'export_onnx.py', 'pyproject.toml', 'uv.lock']:
            shutil.copy2(file, a.output / 'source' / file)
        shutil.copy2(a.folds, a.output / 'folds.json')
    progress_path = a.output / 'progress.json'
    statuses = json.loads(progress_path.read_text())['runs'] if progress_path.exists() else {}
    def update(name, status, **extra):
        statuses[name] = {'status': status, 'updated_at': datetime.now(timezone.utc).isoformat(), **extra}
        atomic_json(progress_path, {'job_id': os.environ['SLURM_JOB_ID'], 'runs': statuses})
        comparison(a.output, runs, statuses)
    python = sys.executable
    for run in runs:
        name, config = run['name'], run['config']
        directory = a.output / name
        directory.mkdir(exist_ok=True)
        config_path = directory / 'config.json'
        if config_path.exists() and json.loads(config_path.read_text()) != config:
            raise ValueError(f'Existing run config differs: {config_path}')
        atomic_json(config_path, config)
        checkpoint = directory / 'checkpoints/last.ckpt'
        model = directory / 'model.onnx'
        result_path = directory / 'validation/results.json'
        try:
            if not checkpoint_complete(checkpoint, config, spec['screen_epochs']):
                update(name, 'training')
                command = [python, '-u', '-m', 'experiments.lejepa.main', '--config', config_path,
                           '--data', a.data, '--output', directory, '--accelerator', 'gpu',
                           '--stop-after-epochs', str(spec['screen_epochs']),
                           '--retain-every', str(spec['retain_every'])]
                if checkpoint.exists():
                    command += ['--resume', checkpoint]
                run_command(command, directory / 'training.log')
                if not checkpoint_complete(checkpoint, config, spec['screen_epochs']):
                    raise RuntimeError('Training returned before completing the screening budget')
            export_path = directory / 'export.json'
            checkpoint_hash = digest(checkpoint)
            export_state = json.loads(export_path.read_text()) if export_path.exists() else {}
            if (not model.exists() or export_state.get('checkpoint_sha256') != checkpoint_hash or
                    export_state.get('model_sha256') != digest(model)):
                update(name, 'exporting')
                run_command([python, '-u', '-m', 'experiments.lejepa.export', checkpoint, model], directory / 'export.log')
                atomic_json(export_path, {'checkpoint_sha256': checkpoint_hash, 'model_sha256': digest(model)})
            if not compatible_result(result_path, digest(model), folds_hash, data_hash):
                update(name, 'evaluating')
                run_command([python, '-u', '-m', 'experiments.lejepa.cross_validate',
                             '--model', model, '--data', a.data, '--folds', a.folds,
                             '--output', directory / 'validation'], directory / 'validation.log')
            update(name, 'completed')
        except (subprocess.CalledProcessError, RuntimeError) as error:
            update(name, 'failed', error=str(error))
            print(f'{name} failed; details in its logs. Continuing to next configuration.', flush=True)
    failed = [r['name'] for r in runs if statuses.get(r['name'], {}).get('status') != 'completed']
    print('Sweep finished:', a.output / 'REPORT.md', 'incomplete:', failed, flush=True)
    if failed:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
