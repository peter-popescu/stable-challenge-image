"""Within-seed milestone comparisons must exclude missing checkpoints."""
import pytest
from experiments.lejepa.milestones import summarize


def row(role, seed, a, b):
    return {'role':role, 'seed':seed, 'status':'completed',
            'accuracy_150':a, 'accuracy_200':b, 'macro_150':a, 'macro_200':b,
            'official_loo_150':a, 'official_loo_200':b}


def test_missing_checkpoint_is_not_imputed_or_counted():
    rows = [{'role':'baseline', 'seed':42, 'status':'missing epoch150 checkpoint', 'accuracy_200':.7},
            row('baseline',43,.68,.70), row('baseline',44,.67,.68),
            row('winner',42,.71,.73), row('winner',43,.70,.72), row('winner',44,.71,.72)]
    summary = summarize(rows)
    assert summary['baseline']['seeds'] == [43,44]
    assert summary['winner']['seeds'] == [42,43,44]
    assert summary['baseline']['pairs'] == 2
    assert summary['baseline']['mean_gain'] == pytest.approx(.015)
    assert summary['winner']['mean_gain'] == pytest.approx(.05/3)


def test_negative_gains_and_single_seed_uncertainty():
    result = summarize([row('winner',42,.74,.72)])['winner']
    assert result['mean_gain'] == pytest.approx(-.02)
    assert result['improved_seeds'] == 0
    assert result['gain_seed_sd'] is None
