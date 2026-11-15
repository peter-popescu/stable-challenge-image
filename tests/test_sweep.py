"""Screening-plan and real epoch-boundary continuation checks."""
import copy
import json
import os
from pathlib import Path
import sys
import pytest
from experiments.lejepa.sweep import configurations, compatible_result


def base():
    return json.loads(Path('experiments/lejepa/config.json').read_text())


def spec():
    return json.loads(Path('experiments/lejepa/sweep.json').read_text())


def test_grid_contains_control_and_only_changes_lr_and_lambda():
    original = base()
    runs = configurations(original, spec())
    assert len(runs) == len({r['name'] for r in runs}) == 9
    assert runs[0]['config'] == original
    assert {(r['config']['lr'], r['config']['lamb']) for r in runs} == {
        (lr, weight) for lr in [.0003, .001, .003] for weight in [.005, .02, .08]}
    for r in runs:
        assert {k:v for k,v in r['config'].items() if k not in ['lr', 'lamb']} == {
            k:v for k,v in original.items() if k not in ['lr', 'lamb']}
        assert r['config']['epochs'] == 200
    assert original == base()


def test_invalid_grid_and_budgets_are_rejected():
    for key, value in [('screen_epochs', 201), ('retain_every', 0),
                       ('learning_rates', [.001, .001]), ('sigreg_weights', [float('nan')]),
                       ('learning_rates', [.003])]:
        changed = copy.deepcopy(spec())
        changed[key] = value
        with pytest.raises(ValueError):
            configurations(base(), changed)


def test_cv_cache_requires_identical_model_folds_and_data(tmp_path):
    path = tmp_path / 'results.json'
    assert not compatible_result(path, 'model', 'folds', 'data')
    path.write_text(json.dumps({'models': {'trained': {'sha256':'model'}},
                               'fold_manifest_sha256':'folds', 'data_sha256':'data'}))
    assert compatible_result(path, 'model', 'folds', 'data')
    assert not compatible_result(path, 'other', 'folds', 'data')
    assert not compatible_result(path, 'model', 'other', 'data')
    assert not compatible_result(path, 'model', 'folds', 'other')


def test_execution_budget_does_not_change_configured_schedule():
    from experiments.lejepa.main import training_budget
    c = base()
    assert training_budget(c, 100) == 100
    assert training_budget(c) == 200
    assert c['epochs'] == 200
    for value in [0, 201]:
        with pytest.raises(ValueError):
            training_budget(c, value)


@pytest.mark.skipif(not os.environ.get('SLURM_JOB_ID'), reason='Training requires Slurm')
def test_real_checkpoint_continuation_keeps_schedule_and_milestone(tmp_path, monkeypatch):
    import torch
    from experiments.lejepa.main import main
    c = base()
    c.update(epochs=3, batch_size=2, workers=0, n_slices=8, warmup_epochs=1)
    config_path = tmp_path / 'config.json'
    config_path.write_text(json.dumps(c))
    output = tmp_path / 'run'
    args = ['train', '--config', str(config_path), '--data', 'data/sample.parquet',
            '--output', str(output), '--accelerator', 'cpu', '--retain-every', '2']
    monkeypatch.setattr(sys, 'argv', args + ['--stop-after-epochs', '2'])
    main()
    checkpoint = output / 'checkpoints/last.ckpt'
    partial = torch.load(checkpoint, map_location='cpu', weights_only=False)
    assert partial['hyper_parameters']['config']['epochs'] == 3
    assert partial['epoch'] == 1 and partial['global_step'] == 2
    assert partial['optimizer_states'][0]['param_groups'][0]['lr'] == pytest.approx(.0005)
    milestones = list((output / 'checkpoints').glob('retained-*'))
    assert len(milestones) == 1
    retained_bytes = milestones[0].read_bytes()
    monkeypatch.setattr(sys, 'argv', args + ['--resume', str(checkpoint)])
    main()
    complete = torch.load(checkpoint, map_location='cpu', weights_only=False)
    assert complete['epoch'] == 2 and complete['global_step'] == 3
    assert complete['optimizer_states'][0]['param_groups'][0]['lr'] == pytest.approx(0.)
    assert milestones[0].read_bytes() == retained_bytes
    assert not torch.equal(partial['state_dict']['method.backbone.conv.0.weight'],
                           complete['state_dict']['method.backbone.conv.0.weight'])
