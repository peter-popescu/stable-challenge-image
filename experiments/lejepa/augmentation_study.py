"""Prepare or execute a small augmentation study using existing train/eval entrypoints."""
import argparse
import json
import os
from pathlib import Path
import shutil
import sys

from .augmentation_policy import resolve_augmentation, augmentation_metadata
from .sweep import digest, atomic_json, run_command, compatible_result
from .lr_history import verify_lr_history


def canonical(config):
    return dict(config, **resolve_augmentation(config))


def load_matrix(spec):
    runs = [dict(run, config=json.loads(Path(run['config']).read_text())) for run in spec['runs']]
    names = [run['name'] for run in runs]
    allowed = [['control', 'local-min10', 'jitter-strong'],
               ['control', 'local-min50'], ['control', 'more-views']]
    if names not in allowed:
        raise ValueError('Unexpected study matrix; review a new protocol before expanding it')
    base = runs[0]['config']
    if base['seed'] != 42 or base['epochs'] != 200 or base['lr'] != .003 or base['lamb'] != .005:
        raise ValueError('Require the tuned seed-42 200-epoch control')
    expected = {'control': {}, 'local-min10': {'local_scale': [.1, .6]},
                'local-min50': {'local_scale': [.5, .6]}, 'more-views': {'local_views': 6},
                'jitter-strong': {'jitter': .4, 'jitter_contrast': .4, 'jitter_saturation': .4,
                                  'jitter_hue': .1, 'jitter_probability': .8}}
    for run in runs:
        changes = expected[run['name']]
        if canonical(run['config']) != canonical(dict(base, **changes)):
            raise ValueError('A variant changes parameters outside its declared intervention')
    if spec['milestones'] != [50, 100, 150, 200]:
        raise ValueError('Require the reviewed periodic evaluation budget')
    return runs


def retained(directory, epoch):
    matches = list((directory / 'checkpoints').glob(f'retained-epoch-{epoch-1:03d}-step-*.ckpt'))
    if len(matches) != 1:
        raise ValueError(f'Missing or ambiguous checkpoint: {directory}, epoch {epoch}')
    return matches[0]


def prepare(spec, runs, data, folds):
    data_hash = digest(data)
    if json.loads(folds.read_text())['data_sha256'] != data_hash:
        raise ValueError('Fold manifest belongs to another dataset')
    control = spec['control']
    histories = []
    for key, filename in [('screen', 'resolved.json'), ('continued', 'resume.json')]:
        directory = Path(control[key])
        history = json.loads((directory / filename).read_text())
        if canonical(history['config']) != canonical(runs[0]['config']) or history['data_sha256'] != data_hash:
            raise ValueError('Historical control differs from the proposed control')
        histories.append(history)
    if histories[0]['runtime']['max_epochs'] != 100 or histories[1]['runtime']['max_epochs'] != 200:
        raise ValueError('Historical execution budgets differ')
    if Path(histories[1]['resume']).resolve() != (Path(control['screen']) / 'checkpoints/last.ckpt').resolve():
        raise ValueError('Control continuation does not resume the recorded screen')
    lr_audit = verify_lr_history(Path(control['screen']) / 'metrics/metrics.csv',
                                 Path(control['continued']) / 'metrics/metrics.csv',
                                 histories[0]['config'], histories[0]['dataset_size'])
    checkpoints = {str(epoch): str(retained(Path(control['screen' if epoch <= 100 else 'continued']), epoch))
                   for epoch in spec['milestones']}
    files = list(Path(__file__).parent.rglob('*.py')) + list(Path(__file__).with_name('augmentations').glob('*.json'))
    files += list(Path(__file__).parent.glob('*.sh'))
    files += [Path(f) for f in ['evaluate.py', 'training_data.py', 'export_onnx.py', 'pyproject.toml', 'uv.lock']]
    files += [Path('tests/test_augmentation_study.py'), Path('tests/test_lejepa.py')]
    references = [Path(control['screen']) / 'resolved.json', Path(control['continued']) / 'resume.json',
                  Path(control['continued']) / 'validation/results.json',
                  Path(control['continued']).parent / 'plan.json']
    loss_files = list(Path('.venv/lib').glob('python*/site-packages/stable_pretraining/methods/lejepa.py'))
    if len(loss_files) != 1:
        raise ValueError('Require one provisioned pinned stable-pretraining implementation')
    return dict(schema_version=2, spec=spec, runs=runs,
                augmentations={r['name']: augmentation_metadata(r['config']) for r in runs},
                data=str(data.resolve()), data_sha256=data_hash,
                folds=str(folds.resolve()), fold_manifest_sha256=digest(folds),
                control_checkpoints=checkpoints,
                control_checkpoint_hashes={epoch: digest(path) for epoch, path in checkpoints.items()},
                reference_hashes={str(p): digest(p) for p in references},
                control_lr_audit=lr_audit,
                loss_implementation_sha256=digest(loss_files[0]),
                source_hashes={str(p.resolve().relative_to(Path.cwd().resolve())): digest(p)
                               for p in sorted(set(files))})


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--spec', type=Path, default=Path(__file__).with_name('augmentations') / 'study.json')
    p.add_argument('--data', type=Path, default=Path('data/train.parquet'))
    p.add_argument('--folds', type=Path, default=Path('experiment_outputs/validation/stratified-5fold-seed42.json'))
    p.add_argument('--output', type=Path, default=Path('experiment_outputs/augmentation-study/seed42-200-diversity'))
    p.add_argument('--prepare-only', action='store_true', help='Freeze plan without importing Torch or executing experiments')
    a = p.parse_args()
    if not a.prepare_only and not os.environ.get('SLURM_JOB_ID'):
        p.error('Execution requires one Slurm allocation; --prepare-only is safe on a login host')
    spec = json.loads(a.spec.read_text())
    runs = load_matrix(spec)
    plan = prepare(spec, runs, a.data, a.folds)
    a.output.mkdir(parents=True, exist_ok=True)
    plan_path = a.output / 'plan.json'
    if plan_path.exists():
        if json.loads(plan_path.read_text()) != plan:
            raise ValueError('Plan/source/reference changed; use a new study output directory')
    else:
        atomic_json(plan_path, plan)
    for name, expected_hash in plan['source_hashes'].items():
        destination = a.output / 'source' / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists() and digest(destination) != expected_hash:
            raise ValueError('Saved source snapshot was modified')
        if not destination.exists():
            shutil.copy2(name, destination)
    if not (a.output / 'folds.json').exists():
        shutil.copy2(a.folds, a.output / 'folds.json')
    if digest(a.output / 'folds.json') != plan['fold_manifest_sha256']:
        raise ValueError('Saved fold snapshot was modified')
    atomic_json(a.output / 'control_lr_audit.json', plan['control_lr_audit'])
    for run in runs:
        directory = a.output / run['name']
        directory.mkdir(exist_ok=True)
        atomic_json(directory / 'config.json', run['config'])
        atomic_json(directory / 'augmentation.json', augmentation_metadata(run['config']))
    if a.prepare_only:
        print('Prepared; no jobs submitted or computation started:', plan_path)
        return

    import torch
    from .augmentation_report import report
    progress_path = a.output / 'progress.json'
    progress = json.loads(progress_path.read_text()) if progress_path.exists() else {}

    def update(name, epoch, phase):
        progress[name] = dict(epoch=epoch, phase=phase, job_id=os.environ['SLURM_JOB_ID'])
        atomic_json(progress_path, progress)

    def completed(path, config):
        if not path.exists():
            return 0
        state = torch.load(path, map_location='cpu', weights_only=False)
        if canonical(state['hyper_parameters']['config']) != canonical(config):
            raise ValueError('Checkpoint configuration differs from the study plan')
        epoch = state['epoch'] + 1
        if epoch > config['epochs']:
            raise ValueError('Checkpoint exceeds schedule horizon')
        return epoch

    for run in runs:
        name, config = run['name'], run['config']
        directory = a.output / name
        if name != 'control' and not (directory / 'checkpoints/last.ckpt').exists():
            update(name, 0, 'memory-preflight')
            # Exercise the exact view count/batch via the existing GPU preflight.
            old_config = os.environ.get('LEJEPA_MEMORY_CONFIG')
            os.environ['LEJEPA_MEMORY_CONFIG'] = str(directory / 'config.json')
            try:
                run_command([sys.executable, '-m', 'pytest',
                             'tests/test_lejepa.py::test_full_screening_batch_fits_cuda', '-q',
                             '--basetemp', directory / 'memory-preflight-tests'],
                            directory / 'memory-preflight.log')
            finally:
                if old_config is None:
                    os.environ.pop('LEJEPA_MEMORY_CONFIG', None)
                else:
                    os.environ['LEJEPA_MEMORY_CONFIG'] = old_config
            memory_report = Path(os.environ['EXPERIMENT_STORAGE']) / f'gpu-memory-preflight-{os.environ["SLURM_JOB_ID"]}.json'
            shutil.copy2(memory_report, directory / 'memory-preflight.json')
        for epoch in spec['milestones']:
            if name == 'control':
                checkpoint = Path(plan['control_checkpoints'][str(epoch)])
            else:
                last = directory / 'checkpoints/last.ckpt'
                if completed(last, config) < epoch:
                    update(name, epoch, 'training')
                    command = [sys.executable, '-u', '-m', 'experiments.lejepa.main',
                               '--config', directory / 'config.json', '--data', a.data,
                               '--output', directory, '--accelerator', 'gpu',
                               '--retain-every', '50', '--stop-after-epochs', str(epoch)]
                    if last.exists():
                        command += ['--resume', last]
                    run_command(command, directory / 'training.log')
                    if completed(last, config) != epoch:
                        raise RuntimeError('Training returned before the milestone')
                checkpoint = retained(directory, epoch)
            if completed(checkpoint, config) != epoch:
                raise ValueError('Wrong milestone checkpoint')
            destination = directory / f'epoch{epoch}'
            destination.mkdir(exist_ok=True)
            model = destination / 'model.onnx'
            provenance_path = destination / 'provenance.json'
            provenance = json.loads(provenance_path.read_text()) if provenance_path.exists() else {}
            if not (provenance.get('checkpoint_sha256') == digest(checkpoint) and model.exists()
                    and provenance.get('onnx_sha256') == digest(model)
                    and compatible_result(destination / 'validation/results.json', digest(model),
                                          plan['fold_manifest_sha256'], plan['data_sha256'])):
                update(name, epoch, 'evaluating')
                run_command([sys.executable, '-u', '-m', 'experiments.lejepa.checkpoint_eval',
                             '--checkpoint', checkpoint, '--expected-epoch', str(epoch),
                             '--data', a.data, '--folds', a.folds, '--output', destination],
                            destination / 'evaluation.log')
            marker = destination / 'diagnostics_complete.json'
            expected = dict(checkpoint_sha256=digest(checkpoint), plan_sha256=digest(plan_path))
            if not marker.exists() or json.loads(marker.read_text()) != expected:
                update(name, epoch, 'diagnostics')
                run_command([sys.executable, '-u', '-m', 'experiments.lejepa.diagnostics', checkpoint,
                             '--data', a.data, '--output', destination / 'spatial', '--shared',
                             '--count', str(spec['pca_count']), '--seed', str(spec['diagnostic_seed'])],
                            destination / 'diagnostics.log')
                run_command([sys.executable, '-u', '-m', 'experiments.lejepa.invariance', checkpoint,
                             '--data', a.data, '--output', destination / 'invariance',
                             '--count', str(spec['invariance_count']), '--seed', str(spec['diagnostic_seed'])],
                            destination / 'invariance.log')
                atomic_json(marker, expected)
            update(name, epoch, 'completed')
            report(a.output)
    print('Study complete:', a.output / 'REPORT.md')


if __name__ == '__main__':
    main()
