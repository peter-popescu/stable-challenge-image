import numpy as np
import pytest

from probe_diagnostics import domain_details, low_support_probes


def test_domain_errors_distinguish_within_domain_confusion():
    report = domain_details([18, 19, 21, 1], [12, 19, 12, 1])
    assert report["galaxy"]["images"] == 2
    assert report["galaxy"]["accuracy"] == 0.5
    assert report["galaxy"]["cross_domain_errors"] == 0
    assert report["natural"]["cross_domain_errors"] == 1
    assert report["satellite"]["accuracy"] == 1
    assert domain_details([99], [99])["unmapped"]["accuracy"] == 1


def test_low_support_is_reproducible_and_disjoint_in_single_pool():
    rng = np.random.default_rng(19)
    labels = np.repeat([0, 1, 2], 8)
    features = rng.normal(size=(24, 5)) + labels[:, None] * 2
    options = dict(shots=5, repeats=3, seed=2026, same_pool=True,
                   support_indices=np.arange(100, 124), query_indices=np.arange(100, 124))
    first = low_support_probes(features, labels, features, labels, **options)
    assert first == low_support_probes(features, labels, features, labels, **options)
    for row in first["results"]:
        support = row["support_row_indices"]
        assert len(support) == 15
        assert not set(support) & set(row["query_row_indices"])
        assert row["images"] == 9
    assert first["accuracy_mean"] == np.mean([row["score"] for row in first["results"]])


def test_low_support_fit_never_receives_query_features(monkeypatch):
    fits = []
    class SpyRidge:
        lam = 1.0
        def fit(self, features, labels):
            fits.append((features.copy(), labels.copy()))
            return self
        def predict(self, features):
            assert (features == 999).all()
            return np.array([0, 1])
    monkeypatch.setattr("probe_diagnostics.Ridge", SpyRidge)
    support = np.arange(24).reshape(8, 3)
    labels = np.repeat([0, 1], 4)
    result = low_support_probes(support, labels, np.full((2, 3), 999), [0, 1],
                                shots=2, repeats=3)
    assert result["accuracy_mean"] == 1
    for features, selected_labels in fits:
        assert features.shape == (4, 3)
        assert (features < 999).all()
        assert np.bincount(selected_labels).tolist() == [2, 2]


def test_low_support_requires_enough_examples():
    with pytest.raises(ValueError, match="at least 6"):
        low_support_probes(np.ones((5, 2)), np.zeros(5), np.ones((5, 2)), np.zeros(5),
                           shots=5, same_pool=True)


def test_diagnostics_cache_matches_scored_features(eval_data, tmp_path):
    from conftest import MODEL
    from diagnose import diagnose, file_hash
    path = tmp_path / "embeddings.npz"
    result = diagnose(MODEL, eval_data, low_shot=5, repeats=2, embeddings_out=path)
    with np.load(path) as cached:
        assert cached["features"].shape == (36, 1024)
        assert cached["same_pool"].item()
        assert cached["model_sha256"].item() == file_hash(MODEL)
        assert cached["data_sha256"].item() == result["data_sha256"]
        assert (cached["labels"] >= 0).all()
        np.testing.assert_array_equal(cached["features"], cached["support_features"])
    assert len(result["low_support"]["results"]) == 2
    assert all(row["images"] == 21 for row in result["low_support"]["results"])
