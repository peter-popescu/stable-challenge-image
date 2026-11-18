"""Compare completed augmentation milestones without fitting new probes."""
import argparse
import csv
import json
import os
from pathlib import Path

from .sweep import atomic_json, digest


def report(output):
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from PIL import Image, ImageDraw
    plan = json.loads((output / 'plan.json').read_text())
    rows, classes, folds, invariances = [], [], [], []
    lines = ['# Augmentation study: seed 42, planned 200-epoch cosine', '',
             'All comparisons use the same development population. Neither fixed-fold nor local LOO gains establish hidden-class/domain transfer.', '',
             '| Policy | Epoch | Five-fold | Macro | Local LOO | Five-fold change vs control |',
             '|---|---:|---:|---:|---:|---:|']
    if any(r['config']['local_views'] == 6 for r in plan['runs']):
        # Keep the table rows adjacent; add the caveat after its header/rows below.
        more_views = True
    else:
        more_views = False
    available = {}
    for run in plan['runs']:
        for epoch in plan['spec']['milestones']:
            directory = output / run['name'] / f'epoch{epoch}'
            path = directory / 'validation/results.json'
            if not path.exists():
                continue
            result = json.loads(path.read_text())
            if (result['data_sha256'] != plan['data_sha256'] or
                    result['fold_manifest_sha256'] != plan['fold_manifest_sha256'] or
                    result['models']['trained']['sha256'] != digest(directory / 'model.onnx')):
                raise ValueError('Refusing a report with unmatched data/folds/model')
            available[run['name'], epoch] = result['results']['trained']
    for (name, epoch), result in available.items():
        m = result['metrics']
        control = available.get(('control', epoch))
        delta = m['accuracy'] - control['metrics']['accuracy'] if control else None
        rows.append(dict(name=name, epoch=epoch, accuracy=m['accuracy'], macro_accuracy=m['macro_accuracy'],
                         official_local_loo_accuracy=m['official_local_loo_accuracy'], delta=delta))
        change = f'{100*delta:+.2f} pp' if delta is not None else 'pending'
        lines.append(f'| {name} | {epoch} | {m["accuracy"]:.2%} | {m["macro_accuracy"]:.2%} | {m["official_local_loo_accuracy"]:.2%} | {change} |')
        control_classes = {c['class_id']: c for c in control['metrics']['per_class']} if control else {}
        for c in m['per_class']:
            baseline = control_classes.get(c['class_id'])
            classes.append(dict(name=name, epoch=epoch, **c,
                                delta=c['accuracy'] - baseline['accuracy'] if baseline else None))
        control_folds = {f['fold']: f for f in control['folds']} if control else {}
        for f in result['folds']:
            baseline = control_folds.get(f['fold'])
            folds.append(dict(name=name, epoch=epoch, **f,
                              delta=f['accuracy'] - baseline['accuracy'] if baseline else None))
        path = output / name / f'epoch{epoch}/invariance/invariance.json'
        if path.exists():
            diagnostic = json.loads(path.read_text())
            control_path = output / 'control' / f'epoch{epoch}/invariance/invariance.json'
            reference_diagnostic = json.loads(control_path.read_text())
            for key in ['indices', 'seed', 'transforms', 'crop_position', 'crop_output', 'crop_ratio', 'data_sha256']:
                if diagnostic[key] != reference_diagnostic[key]:
                    raise ValueError('Controlled invariance measurements differ in images or settings')
            for transform, values in diagnostic['measurements'].items():
                invariances.append(dict(name=name, epoch=epoch, transform=transform, **values,
                                        mismatched_pair_cosine=diagnostic['mismatched_pair_cosine'],
                                        mean_variance=diagnostic['mean_variance']))
    for filename, records in [('comparison', rows), ('per_class_changes', classes),
                              ('per_fold_changes', folds), ('invariance', invariances)]:
        if records:
            with (output / (filename + '.csv')).open('w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=list(records[0]))
                writer.writeheader()
                writer.writerows(records)
    atomic_json(output / 'comparison.json', rows)
    if rows:
        fig, axes = plt.subplots(1, 3, figsize=(14, 4))
        for axis, metric in zip(axes, ['accuracy', 'macro_accuracy', 'official_local_loo_accuracy']):
            for run in plan['runs']:
                selected = [r for r in rows if r['name'] == run['name']]
                axis.plot([r['epoch'] for r in selected], [100*r[metric] for r in selected], marker='o', label=run['name'])
            axis.set(xlabel='Completed epochs', ylabel='Percent', title=metric)
        axes[0].legend()
        fig.tight_layout()
        fig.savefig(output / 'learning_curves.png')
        plt.close(fig)
        lines += ['', '![Probe learning curves](learning_curves.png)', '',
                  'Per-fold and per-class paired differences are in per_fold_changes.csv and per_class_changes.csv; positive changes favor the variant.']
    # Plot both historical curve segments; keep losses separate from probe curves.
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for run in plan['runs']:
        directories = ([Path(plan['spec']['control'][key]) for key in ['screen', 'continued']]
                       if run['name'] == 'control' else [output / run['name']])
        by_epoch = {}
        for directory in directories:
            path = directory / 'metrics/metrics.csv'
            if path.exists():
                for row in csv.DictReader(path.open()):
                    if row.get('loss_epoch') and row.get('epoch'):
                        by_epoch[int(float(row['epoch'])) + 1] = row
        for axis, key in zip(axes, ['loss_epoch', 'invariance_epoch', 'sigreg_epoch']):
            epochs = sorted(by_epoch)
            axis.plot(epochs, [float(by_epoch[e][key]) for e in epochs], label=run['name'])
            axis.set(xlabel='Completed epochs', title=key)
            axis.set_yscale('log')
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(output / 'training_curves.png')
    plt.close(fig)
    lines += ['', '![SSL loss curves](training_curves.png)', '',
              'Losses across augmentation policies represent different prediction tasks; lower loss alone does not establish better representations.']
    if more_views:
        lines += ['', 'More-views uses 2 global + 6 local views versus 2 + 2: equal epochs/optimizer steps are not equal compute. Invariance global/local residual weights shift from 50/50 to 25/75; SIGReg pools 1024 instead of 512 projections at batch128 and retains its sample-count multiplier. Lambda and standard objective are unchanged. BatchNorm statistics also change. Read memory-preflight.json and per-phase logs for device/resource evidence; runtime estimates in the proposal are not measured benchmarks.']
    final_classes = [c for c in classes if c['epoch'] == 200 and c['name'] != 'control' and c['delta'] is not None]
    for run in plan['runs'][1:]:
        changes = sorted([c for c in final_classes if c['name'] == run['name']], key=lambda c: c['delta'])
        if changes:
            lines += ['', f'## {run["name"]}: final per-class changes', '']
            for c in changes:
                lines.append(f'- Class {c["class_id"]}: {100*c["delta"]:+.2f} pp; accuracy {c["accuracy"]:.2%}, support {c["support"]}.')
    for epoch in plan['spec']['milestones']:
        panels, names, indices = [], [], None
        for run in plan['runs']:
            directory = output / run['name'] / f'epoch{epoch}/spatial'
            if not (directory / 'spatial.png').exists():
                continue
            metadata = json.loads((directory / 'diagnostics.json').read_text())
            if indices is not None and metadata['indices'] != indices:
                raise ValueError('PCA source images differ across policies')
            indices = metadata['indices']
            panels.append(Image.open(directory / 'spatial.png').convert('RGB'))
            names.append(run['name'])
        if panels:
            canvas = Image.new('RGB', (sum(im.width for im in panels), max(im.height for im in panels)+24), 'white')
            draw, left = ImageDraw.Draw(canvas), 0
            for name, im in zip(names, panels):
                canvas.paste(im, (left, 24)); draw.text((left, 4), name, fill='black'); left += im.width
            canvas.save(output / f'spatial-epoch{epoch}.png')
            lines += ['', f'## Spatial PCA, epoch {epoch}', '', f'![Common images and settings](spatial-epoch{epoch}.png)']
    lines += ['', 'PCA uses identical seeded rows, shared-within-model basis, 1st/99th percentile scaling, sign convention and 0.5 overlay. Bases are fitted separately per model; RGB colors are not aligned across encoders or semantic labels.',
              'Each epoch directory also contains neighbors.png, diagnostic variance/effective rank and paired per-image invariance measurements. High cosine alone can indicate collapse; compare mismatched pairs and variance.',
              'No documented class-to-domain mapping exists in the supplied dataset documentation; report numeric classes only. No labels select training augmentations.',
              'Epoch 200 is the primary endpoint; earlier epochs show trajectories. Fold SD is descriptive; one selected seed and repeated development scoring do not establish transfer or statistical significance. Promising policies require additional seeds and a separate 400-epoch schedule.']
    (output / 'REPORT.md').write_text('\n'.join(lines) + '\n')


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):
        p.error('Render comparisons inside a Slurm allocation')
    report(a.output)


if __name__ == '__main__':
    main()
