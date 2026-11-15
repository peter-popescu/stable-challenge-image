"""Continue selected screening checkpoints without overwriting screening artifacts."""
import argparse
import json
import os
from pathlib import Path
import shutil
import sys
import subprocess
from datetime import datetime, timezone

from .sweep import atomic_json, digest, run_command, checkpoint_complete, compatible_result


def selected_runs(sweep, names):
    """Require unique names with completed, unchanged screening configurations."""
    if not names or len(names) != len(set(names)):
        raise ValueError('Choose distinct completed screening runs')
    plan = json.loads((sweep / 'plan.json').read_text())
    progress = json.loads((sweep / 'progress.json').read_text())['runs']
    known = {run['name']: run for run in plan['runs']}
    runs = []
    for name in names:
        if name not in known or progress.get(name, {}).get('status') != 'completed':
            raise ValueError(f'Not a completed screening configuration: {name}')
        run = known[name]
        if json.loads((sweep / name / 'config.json').read_text()) != run['config']:
            raise ValueError(f'Screening config changed: {name}')
        if run['config']['epochs'] != 200:
            raise ValueError('These continuations require the original 200-epoch schedule')
        runs.append(run)
    return runs, plan


def write_comparison(output, sweep, runs, statuses, reference):
    rows = []
    for run in runs:
        name = run['name']
        before = json.loads((sweep / name / 'validation/results.json').read_text())['results']['trained']
        row = {'name': name, 'status': statuses.get(name, {}).get('status', 'pending'),
               'accuracy_100': before['metrics']['accuracy']}
        if row['status'] == 'completed':
            after = json.loads((output / name / 'validation/results.json').read_text())['results']['trained']
            m = after['metrics']
            row.update(accuracy_200=m['accuracy'], macro_200=m['macro_accuracy'],
                       official_loo_200=m['official_local_loo_accuracy'], fold_sd_200=m['fold_sd_accuracy'],
                       improvement_from_100=m['accuracy'] - row['accuracy_100'])
            if reference:
                baseline = reference['results']['trained']
                row['difference_from_baseline_200'] = m['accuracy'] - baseline['metrics']['accuracy']
                row['paired_fold_differences'] = [f['accuracy'] - b['accuracy']
                    for f, b in zip(after['folds'], baseline['folds'])]
        rows.append(row)
    atomic_json(output / 'comparison.json', rows)
    lines = ['# Selected configurations continued to 200 epochs', '',
             '| Configuration | 100 epochs | 200 epochs | Macro at 200 | Official local LOO at 200 |',
             '|---|---:|---:|---:|---:|']
    for row in rows:
        if row['status'] == 'completed':
            lines.append(f'| {row["name"]} | {row["accuracy_100"]:.2%} | {row["accuracy_200"]:.2%} | {row["macro_200"]:.2%} | {row["official_loo_200"]:.2%} |')
        else:
            lines.append(f'| {row["name"]} | {row["accuracy_100"]:.2%} | {row["status"]} | — | — |')
    if reference:
        baseline = reference['results']['trained']['metrics']
        lines += ['', f'Original 200-epoch baseline: outer accuracy {baseline["accuracy"]:.2%}, macro {baseline["macro_accuracy"]:.2%}, official local LOO {baseline["official_local_loo_accuracy"]:.2%}.']
    lines += ['', 'The same fixed folds and original optimizer/scheduler state are used.',
              'Single seed; configurations selected on these development folds. Neither fold variation nor the selected improvement establishes seed-robust or held-out performance.',
              'Screening checkpoints, models, scores and logs remain untouched in the parent sweep.',
              'No leaderboard submission is made automatically.']
    (output / 'REPORT.md').write_text('\n'.join(lines) + '\n')


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--sweep', required=True, type=Path)
    p.add_argument('--runs', required=True, nargs='+')
    p.add_argument('--data', required=True, type=Path)
    p.add_argument('--folds', required=True, type=Path)
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--reference-results', type=Path)
    a = p.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):
        p.error('Run inside one Slurm GPU allocation')
    runs, screen_plan = selected_runs(a.sweep, a.runs)
    data_hash, folds_hash = digest(a.data), digest(a.folds)
    if data_hash != screen_plan['data_sha256'] or folds_hash != screen_plan['fold_manifest_sha256']:
        raise ValueError('Continuation data and folds must match screening exactly')
    reference = json.loads(a.reference_results.read_text()) if a.reference_results else None
    if reference and (reference['data_sha256'] != data_hash or reference['fold_manifest_sha256'] != folds_hash):
        raise ValueError('Reference baseline must use the identical data and folds')
    origins = {}
    for run in runs:
        parent = a.sweep / run['name']
        checkpoint = parent / 'checkpoints/last.ckpt'
        if not checkpoint_complete(checkpoint, run['config'], screen_plan['spec']['screen_epochs']):
            raise ValueError('Screening checkpoint has not completed 100 epochs')
        exported = json.loads((parent / 'export.json').read_text())
        cv = json.loads((parent / 'validation/results.json').read_text())
        ckpt_hash = digest(checkpoint)
        if exported['checkpoint_sha256'] != ckpt_hash or cv['models']['trained']['sha256'] != exported['model_sha256']:
            raise ValueError('Screening scores and checkpoint provenance do not match')
        origins[run['name']] = {'checkpoint': str(checkpoint.resolve()), 'sha256': ckpt_hash}
    a.output.mkdir(parents=True, exist_ok=True)
    sources = list(Path(__file__).parent.glob('*.py')) + [Path(f) for f in ['evaluate.py', 'training_data.py', 'export_onnx.py', 'pyproject.toml', 'uv.lock']]
    plan = {'runs': runs, 'origins': origins, 'target_epochs': 200,
            'data_sha256': data_hash, 'fold_manifest_sha256': folds_hash,
            'source_hashes': {str(f.resolve().relative_to(Path.cwd().resolve())): digest(f) for f in sorted(sources)}}
    plan_path = a.output / 'plan.json'
    if plan_path.exists():
        if json.loads(plan_path.read_text()) != plan:
            raise ValueError('Existing continuation plan differs; refusing to mix experiments')
    else:
        atomic_json(plan_path, plan)
        shutil.copytree(Path(__file__).parent, a.output / 'source/lejepa', ignore=shutil.ignore_patterns('__pycache__'))
        for file in ['evaluate.py', 'training_data.py', 'export_onnx.py', 'pyproject.toml', 'uv.lock']:
            shutil.copy2(file, a.output / 'source' / file)
        shutil.copy2(a.folds, a.output / 'folds.json')
    progress_path = a.output / 'progress.json'
    statuses = json.loads(progress_path.read_text())['runs'] if progress_path.exists() else {}
    def update(name, status, **extra):
        statuses[name] = {'status': status, 'updated_at': datetime.now(timezone.utc).isoformat(), **extra}
        atomic_json(progress_path, {'job_id': os.environ['SLURM_JOB_ID'], 'runs': statuses})
        write_comparison(a.output, a.sweep, runs, statuses, reference)
    for run in runs:
        name, config = run['name'], run['config']
        directory = a.output / name
        directory.mkdir(exist_ok=True)
        config_path = directory / 'config.json'
        if config_path.exists() and json.loads(config_path.read_text()) != config:
            raise ValueError('Existing continuation config changed')
        atomic_json(config_path, config)
        checkpoint = directory / 'checkpoints/last.ckpt'
        try:
            if not checkpoint_complete(checkpoint, config, 200):
                resume = checkpoint if checkpoint.exists() else origins[name]['checkpoint']
                update(name, 'training')
                run_command([sys.executable, '-u', '-m', 'experiments.lejepa.main',
                             '--config', config_path, '--data', a.data, '--output', directory,
                             '--resume', resume, '--retain-every', '50', '--accelerator', 'gpu'],
                            directory / 'training.log')
                if not checkpoint_complete(checkpoint, config, 200):
                    raise RuntimeError('Continuation did not complete the 200-epoch budget')
            model = directory / 'model.onnx'
            export_path = directory / 'export.json'
            ckpt_hash = digest(checkpoint)
            exported = json.loads(export_path.read_text()) if export_path.exists() else {}
            if (not model.exists() or exported.get('checkpoint_sha256') != ckpt_hash or
                    exported.get('model_sha256') != digest(model)):
                update(name, 'exporting')
                run_command([sys.executable, '-u', '-m', 'experiments.lejepa.export', checkpoint, model], directory / 'export.log')
                atomic_json(export_path, {'checkpoint_sha256': ckpt_hash, 'model_sha256': digest(model)})
            result_path = directory / 'validation/results.json'
            if not compatible_result(result_path, digest(model), folds_hash, data_hash):
                update(name, 'evaluating')
                run_command([sys.executable, '-u', '-m', 'experiments.lejepa.cross_validate',
                             '--model', model, '--data', a.data, '--folds', a.folds,
                             '--output', directory / 'validation'], directory / 'validation.log')
            update(name, 'completed')
        except (subprocess.CalledProcessError, RuntimeError) as error:
            update(name, 'failed', error=str(error))
            print(f'{name} failed; continuing to remaining selected configuration', flush=True)
    if any(statuses.get(r['name'], {}).get('status') != 'completed' for r in runs):
        raise SystemExit(1)
    print('Continuations complete:', a.output / 'REPORT.md', flush=True)


if __name__ == '__main__':
    main()
