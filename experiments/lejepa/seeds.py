"""Repeat the baseline and selected configuration with matched additional seeds."""
import argparse
import json
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
from datetime import datetime, timezone

from .sweep import atomic_json, digest, run_command, checkpoint_complete, compatible_result


def seed_runs(baseline, winner, seeds):
    """Change only seed within each configuration; keep the same seeds for both."""
    if not seeds or len(set(seeds)) != len(seeds) or any(type(s) is not int or s < 0 for s in seeds):
        raise ValueError('Additional seeds must be distinct nonnegative integers')
    if baseline['seed'] != winner['seed'] or baseline['seed'] in seeds:
        raise ValueError('Reference seeds must match and must not be repeated')
    if baseline['epochs'] != 200 or winner['epochs'] != 200:
        raise ValueError('Both configurations must have the same 200-epoch budget')
    if {k:v for k,v in baseline.items() if k not in ['lr', 'lamb']} != {
            k:v for k,v in winner.items() if k not in ['lr', 'lamb']}:
        raise ValueError('The selected configuration must differ only in LR and SIGReg weight')
    return [{'name': f'{role}-seed-{seed}', 'role': role, 'config': dict(config, seed=seed)}
            for seed in seeds for role, config in [('baseline', baseline), ('winner', winner)]]


def summarize_seeds(records, reference_seed):
    """Aggregate matched seeds only; distinguish selected and confirmation seeds."""
    matched = sorted(seed for seed, record in records.items() if set(record) == {'baseline', 'winner'})
    summary = {'matched_seeds': matched, 'configurations': {}}
    for role in ['baseline', 'winner']:
        summary['configurations'][role] = {}
        for key in ['accuracy', 'macro_accuracy', 'official_local_loo_accuracy']:
            values = [records[seed][role][key] for seed in matched]
            summary['configurations'][role][key] = {
                'mean': statistics.mean(values) if values else None,
                'seed_sd': statistics.stdev(values) if len(values) > 1 else None}
    deltas = {seed: records[seed]['winner']['accuracy'] - records[seed]['baseline']['accuracy'] for seed in matched}
    confirmation = [seed for seed in matched if seed != reference_seed]
    summary['paired_accuracy_differences'] = deltas
    summary['mean_paired_accuracy_difference'] = statistics.mean(deltas.values()) if deltas else None
    summary['confirmation_seeds'] = confirmation
    summary['confirmation_mean_paired_accuracy_difference'] = (
        statistics.mean(deltas[seed] for seed in confirmation) if confirmation else None)
    return summary


def report(output, runs, statuses, references, reference_seed):
    records = {reference_seed: references}
    rows = []
    for role in ['baseline', 'winner']:
        rows.append({'seed': reference_seed, 'role': role, 'status': 'existing reference', **references[role]})
    for run in runs:
        seed, role = run['config']['seed'], run['role']
        status = statuses.get(run['name'], {}).get('status', 'pending')
        row = {'seed': seed, 'role': role, 'status': status}
        if status == 'completed':
            result = json.loads((output / run['name'] / 'validation/results.json').read_text())
            metrics = result['results']['trained']['metrics']
            row.update(metrics)
            records.setdefault(seed, {})[role] = metrics
        rows.append(row)
    summary = summarize_seeds(records, reference_seed)
    atomic_json(output / 'comparison.json', {'runs': rows, 'aggregate': summary})
    lines = ['# Baseline versus tuned configuration: matched seed repeats', '',
             '| Seed | Configuration | Status | Outer accuracy | Macro accuracy | Official local LOO |',
             '|---|---|---|---:|---:|---:|']
    for row in rows:
        if 'accuracy' in row:
            lines.append(f'| {row["seed"]} | {row["role"]} | {row["status"]} | {row["accuracy"]:.2%} | {row["macro_accuracy"]:.2%} | {row["official_local_loo_accuracy"]:.2%} |')
        else:
            lines.append(f'| {row["seed"]} | {row["role"]} | {row["status"]} | — | — | — |')
    lines += ['', f'Aggregate includes only matched completed seeds: {summary["matched_seeds"]}.', '',
              '| Configuration | Mean outer accuracy | SD across seeds (pp) | Mean macro accuracy |',
              '|---|---:|---:|---:|']
    for role, metrics in summary['configurations'].items():
        sd = metrics['accuracy']['seed_sd']
        sd_text = f'{100*sd:.2f}' if sd is not None else 'not yet estimable'
        lines.append(f'| {role} | {metrics["accuracy"]["mean"]:.2%} | {sd_text} | {metrics["macro_accuracy"]["mean"]:.2%} |')
    lines += ['', '## Paired differences (winner minus baseline)', '']
    for seed, delta in summary['paired_accuracy_differences'].items():
        lines.append(f'- Seed {seed}: {100*delta:+.2f} percentage points.')
    confirmation = summary['confirmation_mean_paired_accuracy_difference']
    if confirmation is not None:
        lines += ['', f'Confirmation-only mean paired gain on seeds {summary["confirmation_seeds"]}: {100*confirmation:+.2f} pp.']
    lines += ['', '## Interpretation limits', '',
              f'Seed {reference_seed} was used for hyperparameter selection. Additional seeds confirm initialization/augmentation variation while keeping folds fixed.',
              'Report variability across independent training seeds separately from variability across probe folds; the 15 fold results are not 15 independent training runs.',
              'Three seeds provide a small descriptive comparison, not a strong significance claim.',
              'All scores use development images seen without labels during SSL. Reusing these folds for selection is not an unbiased final test.',
              'All new runs start from scratch with the supplied images; no pretrained weights, resumed seed-42 weights, or external datasets are used.',
              'No leaderboard submission or further hyperparameter search is performed automatically.']
    (output / 'REPORT.md').write_text('\n'.join(lines) + '\n')


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--baseline-config', required=True, type=Path)
    p.add_argument('--winner-config', required=True, type=Path)
    p.add_argument('--baseline-reference', required=True, type=Path)
    p.add_argument('--winner-reference', required=True, type=Path)
    p.add_argument('--seeds', type=int, nargs='+', default=[43, 44])
    p.add_argument('--data', required=True, type=Path)
    p.add_argument('--folds', required=True, type=Path)
    p.add_argument('--output', required=True, type=Path)
    a = p.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):
        p.error('Run all repeats sequentially within one Slurm allocation')
    baseline, winner = json.loads(a.baseline_config.read_text()), json.loads(a.winner_config.read_text())
    runs = seed_runs(baseline, winner, a.seeds)
    data_hash, folds_hash = digest(a.data), digest(a.folds)
    if json.loads(a.folds.read_text())['data_sha256'] != data_hash:
        raise ValueError('Fixed folds belong to another dataset')
    references = {}
    for role, path in [('baseline', a.baseline_reference), ('winner', a.winner_reference)]:
        r = json.loads(path.read_text())
        if r['data_sha256'] != data_hash or r['fold_manifest_sha256'] != folds_hash:
            raise ValueError('Reference and repeat data/folds must match')
        references[role] = r['results']['trained']['metrics']
    a.output.mkdir(parents=True, exist_ok=True)
    sources = list(Path(__file__).parent.glob('*.py')) + [Path(f) for f in ['evaluate.py', 'training_data.py', 'export_onnx.py', 'pyproject.toml', 'uv.lock']]
    plan = {'runs': runs, 'reference_seed': baseline['seed'],
            'reference_result_hashes': {'baseline': digest(a.baseline_reference), 'winner': digest(a.winner_reference)},
            'data_sha256': data_hash, 'fold_manifest_sha256': folds_hash,
            'source_hashes': {str(f.resolve().relative_to(Path.cwd().resolve())): digest(f) for f in sorted(sources)}}
    plan_path = a.output / 'plan.json'
    if plan_path.exists():
        if json.loads(plan_path.read_text()) != plan:
            raise ValueError('Seed-repeat plan changed; refusing to mix experiments')
    else:
        atomic_json(plan_path, plan)
        shutil.copytree(Path(__file__).parent, a.output / 'source/lejepa', ignore=shutil.ignore_patterns('__pycache__'))
        for file in ['evaluate.py', 'training_data.py', 'export_onnx.py', 'pyproject.toml', 'uv.lock']:
            shutil.copy2(file, a.output / 'source' / file)
        shutil.copy2(a.folds, a.output / 'folds.json')
        shutil.copy2(a.baseline_reference, a.output / 'baseline-seed42-reference.json')
        shutil.copy2(a.winner_reference, a.output / 'winner-seed42-reference.json')
    progress_path = a.output / 'progress.json'
    statuses = json.loads(progress_path.read_text())['runs'] if progress_path.exists() else {}
    def update(name, status, **extra):
        statuses[name] = {'status': status, 'updated_at': datetime.now(timezone.utc).isoformat(), **extra}
        atomic_json(progress_path, {'job_id': os.environ['SLURM_JOB_ID'], 'runs': statuses})
        report(a.output, runs, statuses, references, baseline['seed'])
    report(a.output, runs, statuses, references, baseline['seed'])
    for run in runs:
        name, config = run['name'], run['config']
        directory = a.output / name
        directory.mkdir(exist_ok=True)
        config_path = directory / 'config.json'
        if config_path.exists() and json.loads(config_path.read_text()) != config:
            raise ValueError('Existing run config differs')
        atomic_json(config_path, config)
        checkpoint = directory / 'checkpoints/last.ckpt'
        try:
            if not checkpoint_complete(checkpoint, config, config['epochs']):
                update(name, 'training')
                command = [sys.executable, '-u', '-m', 'experiments.lejepa.main', '--config', config_path,
                           '--data', a.data, '--output', directory, '--accelerator', 'gpu', '--retain-every', '50']
                if checkpoint.exists():
                    command += ['--resume', checkpoint]
                run_command(command, directory / 'training.log')
                if not checkpoint_complete(checkpoint, config, config['epochs']):
                    raise RuntimeError('Training ended before its configured epoch budget')
            model = directory / 'model.onnx'
            export_path = directory / 'export.json'
            ckpt_hash = digest(checkpoint)
            exported = json.loads(export_path.read_text()) if export_path.exists() else {}
            if (not model.exists() or exported.get('checkpoint_sha256') != ckpt_hash or
                    exported.get('model_sha256') != digest(model)):
                update(name, 'exporting')
                run_command([sys.executable, '-u', '-m', 'experiments.lejepa.export', checkpoint, model], directory / 'export.log')
                atomic_json(export_path, {'checkpoint_sha256': ckpt_hash, 'model_sha256': digest(model)})
            if not compatible_result(directory / 'validation/results.json', digest(model), folds_hash, data_hash):
                update(name, 'evaluating')
                run_command([sys.executable, '-u', '-m', 'experiments.lejepa.cross_validate',
                             '--model', model, '--data', a.data, '--folds', a.folds,
                             '--output', directory / 'validation'], directory / 'validation.log')
            update(name, 'completed')
        except (subprocess.CalledProcessError, RuntimeError) as error:
            update(name, 'failed', error=str(error))
            print(f'{name} failed; preserving logs and continuing to remaining repeats', flush=True)
    if any(statuses.get(r['name'], {}).get('status') != 'completed' for r in runs):
        raise SystemExit(1)
    print('Seed repeats complete:', a.output / 'REPORT.md', flush=True)


if __name__ == '__main__':
    main()
