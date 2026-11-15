"""Matched-seed design and independent-seed aggregation checks."""
import json
from pathlib import Path
import pytest
from experiments.lejepa.seeds import seed_runs, summarize_seeds


def configs():
    baseline = json.loads(Path('experiments/lejepa/config.json').read_text())
    return baseline, dict(baseline, lr=.003, lamb=.005)


def test_four_runs_change_only_seed_and_pair_configurations():
    baseline, winner = configs()
    runs = seed_runs(baseline, winner, [43, 44])
    assert [r['name'] for r in runs] == ['baseline-seed-43', 'winner-seed-43', 'baseline-seed-44', 'winner-seed-44']
    for run in runs:
        expected = baseline if run['role'] == 'baseline' else winner
        assert {k:v for k,v in run['config'].items() if k != 'seed'} == {k:v for k,v in expected.items() if k != 'seed'}
        assert run['config']['seed'] in [43, 44]
        assert run['config']['epochs'] == 200
    assert baseline['seed'] == winner['seed'] == 42


def test_rejects_reused_seed_and_unmatched_configurations():
    baseline, winner = configs()
    for seeds in [[], [42], [43, 43], [-1]]:
        with pytest.raises(ValueError):
            seed_runs(baseline, winner, seeds)
    with pytest.raises(ValueError):
        seed_runs(baseline, dict(winner, batch_size=64), [43, 44])


def metrics(value):
    return {'accuracy': value, 'macro_accuracy': value, 'official_local_loo_accuracy': value}


def test_aggregate_only_matched_seeds_and_separate_confirmation():
    records = {42: {'baseline': metrics(.70), 'winner': metrics(.73)},
               43: {'baseline': metrics(.68), 'winner': metrics(.72)},
               44: {'baseline': metrics(.71)}}
    summary = summarize_seeds(records, 42)
    assert summary['matched_seeds'] == [42, 43]
    assert summary['configurations']['baseline']['accuracy']['mean'] == pytest.approx(.69)
    assert summary['configurations']['baseline']['accuracy']['seed_sd'] == pytest.approx(2**.5 * .01)
    assert summary['confirmation_seeds'] == [43]
    assert summary['confirmation_mean_paired_accuracy_difference'] == pytest.approx(.04)
    records[44]['winner'] = metrics(.72)
    summary = summarize_seeds(records, 42)
    assert summary['matched_seeds'] == [42, 43, 44]
    assert summary['confirmation_mean_paired_accuracy_difference'] == pytest.approx(.025)
    assert summary['mean_paired_accuracy_difference'] == pytest.approx(.08 / 3)


def test_single_reference_does_not_claim_a_seed_sd():
    summary = summarize_seeds({42: {'baseline': metrics(.70), 'winner': metrics(.73)}}, 42)
    assert summary['configurations']['winner']['accuracy']['seed_sd'] is None
    assert summary['confirmation_mean_paired_accuracy_difference'] is None
