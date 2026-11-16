"""CPU-only readout diagnostics from diagnose.py's embeddings.npz cache.

The reference ridge score remains unchanged. For a single-image pool, additional
probes share stratified outer folds; queries never fit preprocessing, centroids,
class weights, or balanced-ridge hyperparameters. These are diagnostic scores,
not replacements for the challenge score or tests of encoder generalization.
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from diagnose import classification_details, file_hash
from evaluate import LAMBDAS, Ridge


def stratified_folds(labels, folds=5, seed=2026):
    labels = np.asarray(labels)
    classes, counts = np.unique(labels, return_counts=True)
    folds = min(folds, int(counts.min()))
    if folds < 2:
        raise ValueError("Each class needs at least two examples for stratified cross-validation")
    rng = np.random.default_rng(seed)
    parts = [[] for _ in range(folds)]
    for label in classes:
        for fold, indices in enumerate(np.array_split(rng.permutation(np.flatnonzero(labels == label)), folds)):
            parts[fold].extend(indices.tolist())
    all_indices = np.arange(len(labels))
    return [(np.setdiff1d(all_indices, query), np.sort(query)) for query in parts]


def centroid_predict(x, y, query):
    """Euclidean nearest class mean, with support-only feature standardization."""
    mu, sd = x.mean(0), x.std(0) + 1e-6
    classes = np.unique(y)
    z = (x - mu) / sd
    centers = np.stack([z[y == label].mean(0) for label in classes])
    q = (query - mu) / sd
    distances = ((q[:, None] - centers[None]) ** 2).sum(2)
    return classes[distances.argmin(1)]


class BalancedRidge:
    """Weighted squared-loss ridge, inverse-frequency sample weights.

    Weights average one; the intercept is unpenalized. The ridge multiplier is
    relative to the mean eigenvalue of the weighted sample Gram matrix. There
    is no query-dependent balancing or feature normalization.
    """

    def fit(self, x, y):
        x = np.asarray(x, dtype=np.float64)
        self.classes, inverse, counts = np.unique(y, return_inverse=True, return_counts=True)
        weights = len(y) / (len(counts) * counts[inverse])
        self.active = x.std(0) > 1e-12  # discard zero-padding, without learning from queries
        x = x[:, self.active]
        self.mu = np.average(x, axis=0, weights=weights)
        self.sd = x.std(0) + 1e-6
        x = (x - self.mu) / self.sd
        targets = np.eye(len(counts))[inverse]
        self.intercept = np.average(targets, axis=0, weights=weights)
        gram = x.T @ (weights[:, None] * x)
        self.eigenvalues, self.vectors = np.linalg.eigh(gram)
        self.eigenvalues = np.maximum(self.eigenvalues, 0)
        self.scale = max(float(np.trace(gram) / len(y)), 1e-12)
        self.rhs = self.vectors.T @ (x.T @ (weights[:, None] * (targets - self.intercept)))
        return self

    def scores(self, query, relative_alpha):
        coefficients = self.vectors @ (self.rhs / (self.eigenvalues + relative_alpha * self.scale)[:, None])
        return ((query[:, self.active] - self.mu) / self.sd) @ coefficients + self.intercept

    def predict(self, query, relative_alpha):
        return self.classes[self.scores(query, relative_alpha).argmax(1)]


def choose_balanced_alpha(x, y, folds=3, seed=2026):
    """Select by class-balanced prediction MSE using inner support-only folds."""
    errors = np.zeros(len(LAMBDAS))
    for support, validation in stratified_folds(y, folds, seed):
        model = BalancedRidge().fit(x[support], y[support])
        targets = (y[validation, None] == model.classes).astype(float)
        for i, alpha in enumerate(LAMBDAS):
            squared_error = ((model.scores(x[validation], alpha) - targets) ** 2).mean(1)
            errors[i] += np.mean([squared_error[y[validation] == label].mean()
                                  for label in model.classes])
    return float(LAMBDAS[errors.argmin()])


def load_cache(path):
    with np.load(path, allow_pickle=False) as cache:
        required = {"features", "labels", "row_indices", "support_features", "support_labels",
                    "support_row_indices", "same_pool", "model_sha256", "data_sha256", "support_sha256"}
        missing = required - set(cache.files)
        if missing:
            raise ValueError(f"Cache is missing {sorted(missing)}; regenerate with diagnose.py --save-embeddings")
        arrays = {key: cache[key].copy() for key in required}
    for prefix in ("", "support_"):
        x, y, rows = arrays[prefix + "features"], arrays[prefix + "labels"], arrays[prefix + "row_indices"]
        if (x.ndim != 2 or y.ndim != 1 or rows.ndim != 1 or len(x) != len(y)
                or len(rows) != len(y) or len(y) < 2 or np.any(y < 0)
                or not np.isfinite(x).all()):
            raise ValueError("Cache must contain finite features and aligned labelled rows")
    if arrays["features"].shape[1] != arrays["support_features"].shape[1]:
        raise ValueError("Support and query embedding dimensions differ")
    same_pool = bool(arrays["same_pool"].item())
    if same_pool and any(not np.array_equal(arrays[key], arrays["support_" + key])
                         for key in ("features", "labels", "row_indices")):
        raise ValueError("same_pool cache contains different support/query arrays")
    if not set(arrays["labels"]).issubset(set(arrays["support_labels"])):
        raise ValueError("Every query class needs labelled support")
    return arrays


def compare(cache_path, folds=5, inner_folds=3, seed=2026):
    arrays = load_cache(cache_path)
    x, y = arrays["features"].astype(np.float64), arrays["labels"]
    sx, sy = arrays["support_features"].astype(np.float64), arrays["support_labels"]
    same_pool = bool(arrays["same_pool"].item())
    # Preserve the cached dtype for an exact reference-evaluator comparison.
    reference = Ridge().fit(arrays["support_features"], sy)
    reference_predictions = reference.loo_pred if same_pool else reference.predict(arrays["features"])
    predictions = {name: np.empty_like(y) for name in ("ridge", "nearest_centroid", "balanced_ridge")}
    fold_ids = np.full(len(y), -1, dtype=int)
    settings = []
    partitions = stratified_folds(y, folds, seed) if same_pool else [(None, np.arange(len(y)))]
    for fold, (support, query) in enumerate(partitions):
        train_x, train_y = (sx[support], sy[support]) if same_pool else (sx, sy)
        alpha = choose_balanced_alpha(train_x, train_y, inner_folds, seed + fold)
        predictions["ridge"][query] = Ridge().fit(train_x, train_y).predict(x[query])
        predictions["nearest_centroid"][query] = centroid_predict(train_x, train_y, x[query])
        predictions["balanced_ridge"][query] = BalancedRidge().fit(train_x, train_y).predict(x[query], alpha)
        fold_ids[query] = fold
        settings.append({"fold": fold, "support_images": len(train_y), "query_images": len(query),
                         "balanced_relative_alpha": alpha})
    methods = {}
    for name, pred in predictions.items():
        methods[name] = classification_details(y, pred, arrays["row_indices"])
        methods[name]["predicted_class_counts"] = {str(int(c)): int(np.sum(pred == c)) for c in np.unique(y)}
    report = {
        "cache": str(cache_path), "cache_sha256": file_hash(cache_path),
        "model_sha256": str(arrays["model_sha256"].item()),
        "data_sha256": str(arrays["data_sha256"].item()),
        "support_sha256": str(arrays["support_sha256"].item()),
        "protocol": "stratified_out_of_fold" if same_pool else "support_query",
        "seed": seed, "outer_folds": len(partitions) if same_pool else None,
        "inner_folds_requested": inner_folds,
        "note": "Compare diagnostic methods within the same outer protocol. Reference LOO uses more support. "
                "Alternative classifiers cannot replace official scoring. Training-image caches do not test encoder generalization.",
        "balanced_alpha_selection": "Inner stratified support-only CV; class-balanced squared prediction error",
        "reference_protocol": "official_loo" if same_pool else "official_support_query",
        "reference": classification_details(y, reference_predictions, arrays["row_indices"]),
        "reference_lambda": float(reference.lam), "fold_settings": settings, "methods": methods,
    }
    records = [{"row_index": int(arrays["row_indices"][i]), "true_label": int(y[i]),
                "fold": int(fold_ids[i]), "reference": int(reference_predictions[i]),
                **{name: int(pred[i]) for name, pred in predictions.items()}} for i in range(len(y))]
    return report, records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cache", type=Path, help="embeddings.npz from diagnose.py --save-embeddings")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--inner-folds", type=int, default=3)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--model", type=Path, help="Optionally verify the cache belongs to this ONNX file")
    args = parser.parse_args()
    if args.folds < 2 or args.inner_folds < 2 or args.seed < 0:
        parser.error("fold counts must be >= 2 and seed nonnegative")
    if args.out.exists():
        parser.error("Output directory already exists; choose a new directory to preserve earlier reports")
    if args.model and file_hash(args.model) != str(load_cache(args.cache)["model_sha256"].item()):
        parser.error("Cache model hash differs from --model; regenerate that cache before comparing")
    report, records = compare(args.cache, args.folds, args.inner_folds, args.seed)
    args.out.mkdir(parents=True)
    (args.out / "comparison.json").write_text(json.dumps(report, indent=2))
    with (args.out / "predictions.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    ref = report["reference"]
    print(f"Reference {report['reference_protocol']}: accuracy {ref['score']:.2%}; macro {ref['macro_accuracy']:.2%}")
    print(f"Diagnostic protocol: {report['protocol']}; compare methods below with each other")
    print(f"{'Method':20} {'Accuracy':>10} {'Macro':>10} {'Galaxy':>10}")
    for name, result in report["methods"].items():
        galaxy = result["domains"].get("galaxy", {}).get("accuracy")
        galaxy_text = f"{galaxy:.2%}" if galaxy is not None else "n/a"
        print(f"{name:20} {result['score']:10.2%} {result['macro_accuracy']:10.2%} {galaxy_text:>10}")
    print("Focus classes: correct/total, followed by total predictions of that class")
    for c in (18, 19, 21, 25, 16, 26):
        cells = []
        for name, result in report["methods"].items():
            row = next((row for row in result["per_class"] if row["class_id"] == c), None)
            if row and row["images"]:
                cells.append(f"{name}: {row['correct']}/{row['images']}, predicted={result['predicted_class_counts'][str(c)]}")
        if cells:
            print(f"class {c}: " + "; ".join(cells))
    print(f"Saved comparison.json and predictions.csv to {args.out}")


if __name__ == "__main__":
    main()
