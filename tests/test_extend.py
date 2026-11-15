"""Selection safety before continuing costly screening runs."""
import json
import pytest
from experiments.lejepa.extend import selected_runs


def screening(tmp_path):
    config = {'epochs': 200, 'lr': .003, 'lamb': .005}
    (tmp_path / 'plan.json').write_text(json.dumps({'runs': [{'name':'winner', 'config':config}]}))
    (tmp_path / 'progress.json').write_text(json.dumps({'runs': {'winner': {'status':'completed'}}}))
    (tmp_path / 'winner').mkdir()
    (tmp_path / 'winner/config.json').write_text(json.dumps(config))
    return config


def test_selection_preserves_configuration(tmp_path):
    config = screening(tmp_path)
    runs, _ = selected_runs(tmp_path, ['winner'])
    assert runs == [{'name':'winner', 'config':config}]


def test_selection_rejects_unknown_duplicate_and_failed_runs(tmp_path):
    screening(tmp_path)
    for names in [[], ['other'], ['winner', 'winner']]:
        with pytest.raises(ValueError):
            selected_runs(tmp_path, names)
    (tmp_path / 'progress.json').write_text(json.dumps({'runs': {'winner': {'status':'failed'}}}))
    with pytest.raises(ValueError):
        selected_runs(tmp_path, ['winner'])


def test_selection_rejects_modified_config(tmp_path):
    config = screening(tmp_path)
    config['epochs'] = 100
    (tmp_path / 'winner/config.json').write_text(json.dumps(config))
    with pytest.raises(ValueError):
        selected_runs(tmp_path, ['winner'])
