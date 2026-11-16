import numpy as np
import pytest

from compare_probes import (BalancedRidge, centroid_predict, compare, load_cache,
                            stratified_folds)
from evaluate import Ridge


def cache_file(path, same_pool=True):
    rng = np.random.default_rng(1)
    labels = np.repeat([0, 1, 2], 10)
    features = rng.normal(size=(30, 6)).astype(np.float32)
    features[:, 0] += labels * 3
    features = np.pad(features, ((0, 0), (0, 3)))
    query = features if same_pool else features[:9] + 0.1
    query_labels = labels if same_pool else np.tile([0, 1, 2], 3)
    np.savez(path, features=query, labels=query_labels, row_indices=np.arange(len(query)),
             support_features=features, support_labels=labels, support_row_indices=np.arange(30),
             same_pool=same_pool, model_sha256="model", data_sha256="data", support_sha256="support")
    return features, labels


def test_balanced_ridge_matches_direct_weighted_least_squares_with_intercept():
    rng = np.random.default_rng(12)
    x = rng.normal(size=(13, 4))
    y = np.array([0] * 9 + [1] * 4)
    query = rng.normal(size=(7, 4))
    alpha = 0.3
    model = BalancedRidge().fit(x, y)
    weights = len(y) / (2 * np.bincount(y)[y])
    features = np.column_stack([(x - model.mu) / model.sd, np.ones(len(x))])
    penalty = np.diag([alpha * model.scale] * 4 + [0])
    coefficients = np.linalg.solve(features.T @ (weights[:, None] * features) + penalty,
                                   features.T @ (weights[:, None] * np.eye(2)[y]))
    expected = np.column_stack([(query - model.mu) / model.sd, np.ones(len(query))]) @ coefficients
    np.testing.assert_allclose(model.scores(query, alpha), expected, atol=1e-12)


def test_centroid_support_standardization_and_no_query_fitting():
    x = np.array([[0., 100.], [0., 102.], [10., 100.], [10., 102.]])
    y = np.array([0, 0, 1, 1])
    query = np.array([[0., 101.], [10., 101.]])
    np.testing.assert_array_equal(centroid_predict(x, y, query), [0, 1])
    enlarged = np.vstack([query, [1e9, -1e9]])
    np.testing.assert_array_equal(centroid_predict(x, y, enlarged)[:2], [0, 1])


def test_outer_folds_are_disjoint_and_cover_every_image_once():
    y = np.repeat([0, 1, 2], [9, 13, 17])
    folds = stratified_folds(y, 5, 2026)
    assert len(folds) == 5
    queries = []
    for support, query in folds:
        assert not set(support) & set(query)
        assert set(y[support]) == set(y[query]) == {0, 1, 2}
        queries.extend(query)
    assert sorted(queries) == list(range(len(y)))
    assert [q.tolist() for _, q in folds] == [q.tolist() for _, q in stratified_folds(y, 5, 2026)]


def test_single_pool_reference_matches_official_and_all_probes_share_folds(tmp_path):
    path = tmp_path / "cache.npz"
    x, y = cache_file(path)
    report, records = compare(path, folds=3, inner_folds=2)
    expected = Ridge().fit(x, y)
    assert report["reference"]["score"] == np.mean(expected.loo_pred == y)
    np.testing.assert_array_equal([r["reference"] for r in records], expected.loo_pred)
    assert report["protocol"] == "stratified_out_of_fold"
    assert len(records) == 30
    assert set(r["fold"] for r in records) == {0, 1, 2}
    assert sum(row["query_images"] for row in report["fold_settings"]) == 30
    assert set(report["methods"]) == {"ridge", "nearest_centroid", "balanced_ridge"}


def test_support_query_does_not_use_query_labels_to_fit_or_select_alpha(tmp_path):
    path = tmp_path / "cache.npz"
    cache_file(path, same_pool=False)
    first, records = compare(path, inner_folds=2)
    with np.load(path) as cache:
        arrays = {key: cache[key] for key in cache.files}
    arrays["labels"] = np.roll(arrays["labels"], 1)  # same classes; different query ground truth
    np.savez(path, **arrays)
    second, changed = compare(path, inner_folds=2)
    assert first["fold_settings"] == second["fold_settings"]
    for name in ("reference", "ridge", "nearest_centroid", "balanced_ridge"):
        assert [r[name] for r in records] == [r[name] for r in changed]


def test_cache_rejects_inconsistent_same_pool(tmp_path):
    path = tmp_path / "cache.npz"
    cache_file(path, same_pool=False)
    with np.load(path) as cache:
        arrays = {key: cache[key] for key in cache.files}
    arrays["same_pool"] = True
    np.savez(path, **arrays)
    with pytest.raises(ValueError, match="same_pool"):
        load_cache(path)
