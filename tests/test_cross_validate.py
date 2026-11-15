"""Fixed-fold coverage, immutable plan and outer-split isolation checks."""
import json
import numpy as np
import pytest
from experiments.lejepa.cross_validate import make_folds, fixed_plan, evaluate_folds


def test_stratification_coverage_balance_and_order_independence():
    labels = np.repeat([2, 7, 19], [9, 17, 24])
    ids = np.arange(100, 150)
    folds = make_folds(ids, labels)
    assert set(folds) == set(range(5))
    assert np.ptp(np.bincount(folds, minlength=5)) <= 1
    for c in np.unique(labels):
        counts = np.bincount(folds[labels == c], minlength=5)
        assert counts.min() > 0 and np.ptp(counts) <= 1
    order = np.random.default_rng(123).permutation(len(ids))
    reordered = make_folds(ids[order], labels[order])
    np.testing.assert_array_equal(reordered, folds[order])
    np.testing.assert_array_equal(make_folds(ids, labels), folds)
    assert not np.array_equal(make_folds(ids, labels, seed=43), folds)


def test_rejects_rare_classes_unlabeled_rows_and_duplicate_ids():
    with pytest.raises(ValueError):
        make_folds(np.arange(8), np.repeat([2, 7], 4))
    with pytest.raises(ValueError):
        make_folds(np.arange(10), np.full(10, -1))
    with pytest.raises(ValueError):
        make_folds(np.zeros(10, dtype=int), np.ones(10, dtype=int))


def test_manifest_is_immutable_and_validated(tmp_path):
    path = tmp_path / 'folds.json'
    ids, labels = np.arange(20), np.repeat([2, 7], 10)
    first = fixed_plan(path, ids, labels, 'data-hash')
    original = path.read_bytes()
    np.testing.assert_array_equal(fixed_plan(path, ids[::-1], labels[::-1], 'data-hash'), first[::-1])
    assert path.read_bytes() == original
    with pytest.raises(ValueError):
        fixed_plan(path, ids, labels, 'other-dataset')
    with pytest.raises(ValueError):
        fixed_plan(path, ids, labels, 'data-hash', seed=43)
    corrupted = json.loads(path.read_text())
    corrupted['assignments'][0]['fold'] = (corrupted['assignments'][0]['fold'] + 1) % 5
    path.write_text(json.dumps(corrupted))
    with pytest.raises(ValueError):
        fixed_plan(path, ids, labels, 'data-hash')


def test_each_probe_receives_only_its_outer_training_rows():
    labels = np.repeat([2, 7], 10)
    x = np.arange(60, dtype=float).reshape(20, 3)
    folds = make_folds(np.arange(20), labels)
    fitted, queries = [], []
    class SpyProbe:
        def fit(self, train_x, train_y):
            fitted.append((train_x.copy(), train_y.copy()))
            self.lam = 1.
            self.label = train_y[0]
            return self
        def predict(self, test_x):
            queries.append(test_x.copy())
            return np.full(len(test_x), self.label)
    summary, records, predictions, confusion = evaluate_folds(x, labels, folds, probe_factory=SpyProbe)
    assert len(fitted) == len(queries) == 5
    for fold, ((train_x, train_y), test_x) in enumerate(zip(fitted, queries)):
        np.testing.assert_array_equal(train_x, x[folds != fold])
        np.testing.assert_array_equal(train_y, labels[folds != fold])
        np.testing.assert_array_equal(test_x, x[folds == fold])
        assert records[fold]['train_images'] == 16
        assert records[fold]['validation_images'] == 4
    assert confusion.sum() == len(labels)
    assert summary['accuracy'] == np.mean(predictions == labels)


def test_heldout_features_and_labels_do_not_affect_fold_zero_fit():
    from evaluate import Ridge
    rng = np.random.default_rng(42)
    labels = np.repeat([2, 7], 15)
    x = rng.normal(size=(30, 8)) + labels[:, None] / 7
    folds = make_folds(np.arange(30), labels)
    fitted = []
    class RecordedRidge(Ridge):
        def fit(self, train_x, train_y):
            super().fit(train_x, train_y)
            fitted.append((self.mu.copy(), self.sd.copy(), self.lam))
            return self
    evaluate_folds(x, labels, folds, probe_factory=RecordedRidge)
    first = fitted[0]
    changed_x, changed_y = x.copy(), labels.copy()
    changed_x[folds == 0] += 1000
    changed_y[folds == 0] = np.where(changed_y[folds == 0] == 2, 7, 2)
    fitted.clear()
    evaluate_folds(changed_x, changed_y, folds, probe_factory=RecordedRidge)
    for before, after in zip(first, fitted[0]):
        np.testing.assert_array_equal(before, after)
