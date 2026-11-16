"""Monitoring-only diagnostics; no labels from here enter SSL training.

Domain groups are manually inferred from inspected examples, not official tags.
Low-support repeats measure sensitivity to support selection, not confidence
intervals or transfer to classes absent from SSL pretraining.
"""

import numpy as np

from evaluate import Ridge


DOMAIN_CLASSES = {
    "galaxy": [0, 3, 12, 13, 18, 19, 20, 25, 29],
    "natural": [2, 4, 6, 10, 11, 16, 21, 23, 26, 28],
    "satellite": [1, 5, 7, 9, 14, 15, 17, 22, 24, 27],
}


def domain_details(labels, predictions):
    labels, predictions = np.asarray(labels), np.asarray(predictions)
    groups = {**DOMAIN_CLASSES}
    known = set(sum(DOMAIN_CLASSES.values(), []))
    unknown = sorted(set(map(int, np.concatenate([labels, predictions]))) - known)
    if unknown:
        groups["unmapped"] = unknown
    result = {}
    for name, classes in groups.items():
        mask = np.isin(labels, classes)
        if not mask.any():
            continue
        result[name] = {
            "class_ids": classes, "images": int(mask.sum()),
            "correct": int(np.sum(labels[mask] == predictions[mask])),
            "accuracy": float(np.mean(labels[mask] == predictions[mask])),
            "cross_domain_errors": int(np.sum(~np.isin(predictions[mask], classes))),
        }
    return result


def low_support_probes(support_features, support_labels, query_features, query_labels,
                       shots=5, repeats=10, seed=2026, same_pool=False,
                       support_indices=None, query_indices=None):
    """Select K/class only from support; fit preprocessing/lambda on support.

    For a single file, exclude sampled support rows from that repeat's queries.
    With separate files the caller must ensure image-disjoint support/query data.
    """
    if shots < 1 or repeats < 1:
        raise ValueError("shots and repeats must be positive")
    support_labels, query_labels = np.asarray(support_labels), np.asarray(query_labels)
    classes = np.unique(query_labels)
    pools = [np.flatnonzero(support_labels == label) for label in classes]
    required = shots + int(same_pool)
    if any(len(pool) < required for pool in pools):
        raise ValueError(f"Every evaluated class needs at least {required} labelled support images")
    support_indices = np.arange(len(support_labels)) if support_indices is None else np.asarray(support_indices)
    query_indices = np.arange(len(query_labels)) if query_indices is None else np.asarray(query_indices)
    rng = np.random.default_rng(seed)
    results = []
    for repeat in range(repeats):
        selected = np.concatenate([rng.choice(pool, shots, replace=False) for pool in pools])
        query = np.arange(len(query_labels))
        if same_pool:
            query = np.setdiff1d(query, selected)
        probe = Ridge().fit(support_features[selected], support_labels[selected])
        predictions = probe.predict(query_features[query])
        truth = query_labels[query]
        results.append({
            "repeat": repeat, "score": float(np.mean(predictions == truth)),
            "macro_accuracy": float(np.mean([np.mean(predictions[truth == label] == label)
                                              for label in classes])),
            "images": len(query), "lambda": float(probe.lam),
            "support_row_indices": support_indices[selected].tolist(),
            "query_row_indices": query_indices[query].tolist(),
            "domains": domain_details(truth, predictions),
        })
    return {
        "shots": shots, "repeats": repeats, "seed": seed,
        "protocol": "sampled_support_query",
        "interpretation": "Support-selection variability; not a confidence interval or unseen-class test",
        "accuracy_mean": float(np.mean([row["score"] for row in results])),
        "accuracy_std": float(np.std([row["score"] for row in results])),
        "macro_accuracy_mean": float(np.mean([row["macro_accuracy"] for row in results])),
        "results": results,
    }
