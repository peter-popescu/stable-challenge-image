"""Tests for class accounting and retrieval self-exclusion."""
import numpy as np
import pytest
from experiments.lejepa.error_analysis import classification_summary, neighbors


def test_class_summary_uses_support_and_original_ids():
    metrics, confusion = classification_summary([3, 3, 3, 7], [3, 7, 7, 7])
    np.testing.assert_array_equal(confusion, [[1, 2], [0, 1]])
    assert metrics['accuracy'] == .5
    assert metrics['macro_accuracy'] == pytest.approx((1 / 3 + 1) / 2)
    assert metrics['per_class'] == [
        {'class_id': 3, 'support': 3, 'correct': 1, 'accuracy': 1 / 3},
        {'class_id': 7, 'support': 1, 'correct': 1, 'accuracy': 1.}]


def test_summary_rejects_unknown_prediction():
    with pytest.raises(ValueError):
        classification_summary([3, 7], [3, 8])


def test_neighbors_exclude_self_and_handle_identical_vectors():
    x = np.array([[1., 0.], [1., 0.], [0., 1.], [0., 1.]])
    order, similarity = neighbors(x, k=5)
    assert order.shape == (4, 3)
    assert not np.any(order == np.arange(4)[:, None])
    np.testing.assert_array_equal(order[:, 0], [1, 0, 3, 2])
    assert np.isfinite(similarity).all()


def test_neighbors_invariant_to_positive_feature_scale_and_offset():
    x = np.random.default_rng(42).normal(size=(12, 5))
    before, _ = neighbors(x)
    after, _ = neighbors(x * np.arange(1, 6) + 4)
    np.testing.assert_array_equal(before, after)
