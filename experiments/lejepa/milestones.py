"""Compare retained 150-epoch checkpoints with existing 200-epoch CV results."""
import argparse
import csv
import json
import os
from pathlib import Path
import statistics
import sys

from .sweep import atomic_json, digest, compatible_result


def summarize(rows):
    """Summarize only available within-run pairs, without imputing missing epochs."""
    result = {}
    for role in ['baseline', 'winner']:
        paired = [r for r in rows if r['role'] == role and r['status'] == 'completed']
        if not paired:
            continue
        gains = [r['accuracy_200'] - r['accuracy_150'] for r in paired]
        result[role] = {'seeds': [r['seed'] for r in paired], 'pairs': len(paired),
                        'mean_accuracy_150': statistics.mean(r['accuracy_150'] for r in paired),
                        'mean_accuracy_200': statistics.mean(r['accuracy_200'] for r in paired),
                        'mean_gain': statistics.mean(gains),
                        'gain_seed_sd': statistics.stdev(gains) if len(gains) > 1 else None,
                        'improved_seeds': sum(g > 0 for g in gains),
                        'mean_macro_gain': statistics.mean(r['macro_200'] - r['macro_150'] for r in paired),
                        'mean_official_loo_gain': statistics.mean(r['official_loo_200'] - r['official_loo_150'] for r in paired)}
    return result


def loss_windows(directory):
    path = directory / 'metrics/metrics.csv'
    if not path.exists():
        return {}
    with path.open() as f:
        rows = [r for r in csv.DictReader(f) if r.get('loss_epoch')]
    values = {}
    for name, lo, hi in [('near150', 140, 149), ('near200', 190, 199)]:
        subset = [r for r in rows if lo <= int(float(r['epoch'])) <= hi]
        if subset:
            values[name] = {k: statistics.mean(float(r[k]) for r in subset)
                            for k in ['loss_epoch', 'invariance_epoch', 'sigreg_epoch']}
    return values


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--baseline-root', type=Path, default=Path('experiment_outputs/lejepa-resnet9-seed42-retry'))
    p.add_argument('--winner-root', type=Path, default=Path('experiment_outputs/sweeps/lejepa-lr-sigreg-seed42/continued-200/lr-0.003_lambda-0.005'))
    p.add_argument('--repeat-root', type=Path, default=Path('experiment_outputs/seed-repeats/baseline-vs-tuned'))
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--folds', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):
        p.error('Run checkpoint extraction and probe evaluation within Slurm')
    import torch
    from .export import export
    from . import cross_validate

    a.output.mkdir(parents=True, exist_ok=True)
    data_hash, folds_hash = digest(a.data), digest(a.folds)
    if json.loads(a.folds.read_text())['data_sha256'] != data_hash:
        raise ValueError('Fold manifest and dataset differ')
    cases = []
    for seed in [42, 43, 44]:
        for role in ['baseline', 'winner']:
            directory = ((a.baseline_root if role == 'baseline' else a.winner_root)
                         if seed == 42 else a.repeat_root / f'{role}-seed-{seed}')
            validation = directory / ('cross_validation' if seed == 42 and role == 'baseline' else 'validation') / 'results.json'
            final = json.loads(validation.read_text())
            if final['data_sha256'] != data_hash or final['fold_manifest_sha256'] != folds_hash:
                raise ValueError('Epoch-200 data/folds differ from the requested comparison')
            paths = list((directory / 'checkpoints').glob('retained-epoch-149-step-*.ckpt'))
            if len(paths) > 1:
                raise ValueError('Ambiguous 150-epoch checkpoints')
            cases.append({'seed': seed, 'role': role, 'directory': directory,
                          'checkpoint': paths[0] if paths else None, 'final': final,
                          'reference_hash': digest(validation)})
    plan = {'data_sha256': data_hash, 'fold_manifest_sha256': folds_hash,
            'cases': [{'seed': c['seed'], 'role': c['role'],
                       'checkpoint': str(c['checkpoint'].resolve()) if c['checkpoint'] else None,
                       'checkpoint_sha256': digest(c['checkpoint']) if c['checkpoint'] else None,
                       'epoch200_results_sha256': c['reference_hash']} for c in cases]}
    plan_path = a.output / 'plan.json'
    if plan_path.exists() and json.loads(plan_path.read_text()) != plan:
        raise ValueError('Existing milestone comparison belongs to different artifacts')
    atomic_json(plan_path, plan)
    rows = []
    for case in cases:
        role, seed = case['role'], case['seed']
        final = case['final']['results']['trained']
        checkpoint = case['checkpoint']
        row = {'role': role, 'seed': seed, 'accuracy_200': final['metrics']['accuracy'],
               'macro_200': final['metrics']['macro_accuracy'],
               'official_loo_200': final['metrics']['official_local_loo_accuracy'],
               'loss_windows': loss_windows(case['directory'])}
        if checkpoint is None:
            row.update(status='missing epoch150 checkpoint', reason='Original baseline predates milestone retention')
            rows.append(row)
            print(role, seed, row['status'], flush=True)
            continue
        saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
        if saved['epoch'] != 149 or saved['hyper_parameters']['config']['seed'] != seed:
            raise ValueError('Checkpoint epoch/seed does not match the requested comparison')
        if saved['hyper_parameters']['config']['epochs'] != 200:
            raise ValueError('Checkpoint does not use the original 200-epoch schedule')
        row['global_step_150'] = saved['global_step']
        row['lr_at_checkpoint_150'] = saved['optimizer_states'][0]['param_groups'][0]['lr']
        del saved
        destination = a.output / f'{role}-seed-{seed}'
        destination.mkdir(exist_ok=True)
        model = destination / 'epoch150.onnx'
        marker = destination / 'export.json'
        checkpoint_hash = digest(checkpoint)
        exported = json.loads(marker.read_text()) if marker.exists() else {}
        if (not model.exists() or exported.get('checkpoint_sha256') != checkpoint_hash or
                exported.get('model_sha256') != digest(model)):
            export(checkpoint, model)
            atomic_json(marker, {'checkpoint_sha256': checkpoint_hash, 'model_sha256': digest(model)})
        cv_path = destination / 'validation/results.json'
        if not compatible_result(cv_path, digest(model), folds_hash, data_hash):
            sys.argv = ['cross_validate', '--model', str(model), '--data', str(a.data),
                        '--folds', str(a.folds), '--output', str(destination / 'validation')]
            cross_validate.main()
        before = json.loads(cv_path.read_text())['results']['trained']
        row.update(status='completed', accuracy_150=before['metrics']['accuracy'],
                   macro_150=before['metrics']['macro_accuracy'],
                   official_loo_150=before['metrics']['official_local_loo_accuracy'],
                   gain=final['metrics']['accuracy'] - before['metrics']['accuracy'],
                   paired_fold_gains=[f['accuracy'] - b['accuracy'] for f, b in zip(final['folds'], before['folds'])])
        rows.append(row)
        atomic_json(a.output / 'progress.json', rows)
        print('PAIRED RESULT', json.dumps(row), flush=True)
    summary = summarize(rows)
    atomic_json(a.output / 'results.json', {'runs': rows, 'summary': summary, 'plan': plan})
    lines = ['# Assessing the final 50 epochs', '',
             'All comparisons use the same fixed outer folds and original 200-epoch cosine schedule. No encoder is retrained.', '',
             '| Configuration | Seed | Epoch 150 | Epoch 200 | Gain (pp) | Macro 150 | Macro 200 |',
             '|---|---:|---:|---:|---:|---:|---:|']
    for r in rows:
        if r['status'] == 'completed':
            lines.append(f'| {r["role"]} | {r["seed"]} | {r["accuracy_150"]:.2%} | {r["accuracy_200"]:.2%} | {100*r["gain"]:+.2f} | {r["macro_150"]:.2%} | {r["macro_200"]:.2%} |')
        else:
            lines.append(f'| {r["role"]} | {r["seed"]} | unavailable | {r["accuracy_200"]:.2%} | — | — | {r["macro_200"]:.2%} |')
    lines += ['', '## Means across available within-seed pairs', '',
              '| Configuration | Seeds | Mean at 150 | Mean at 200 | Mean gain (pp) | SD of seed gains (pp) |',
              '|---|---|---:|---:|---:|---:|']
    for role, s in summary.items():
        sd = f'{100*s["gain_seed_sd"]:.2f}' if s['gain_seed_sd'] is not None else 'unavailable'
        lines.append(f'| {role} | {s["seeds"]} | {s["mean_accuracy_150"]:.2%} | {s["mean_accuracy_200"]:.2%} | {100*s["mean_gain"]:+.2f} | {sd} |')
    lines += ['', '## Interpretation limits', '',
              'Baseline seed 42 has no saved epoch-150 checkpoint and is excluded from paired aggregates, not imputed.',
              'The tuned configuration was selected using seed 42; seeds 43/44 are additional repetitions on the same development folds.',
              'Use within-configuration, within-seed changes to assess late training gains. Baseline and tuned aggregates have different seed counts.',
              'Fold differences and seed SD are descriptive, not independent test replicates or confidence intervals.',
              'Improvement from 150 to 200 does not establish improvement beyond 200. The original cosine LR reaches zero at 200; longer training requires an explicitly specified new schedule.',
              'All images were available without labels during SSL; these are development probe scores, not unseen-image/new-class performance.',
              'Raw fold differences, losses near 150/200 and official LOO changes are preserved in results.json.']
    (a.output / 'REPORT.md').write_text('\n'.join(lines) + '\n')
    print('Milestone assessment complete:', a.output / 'REPORT.md', flush=True)


if __name__ == '__main__':
    main()
