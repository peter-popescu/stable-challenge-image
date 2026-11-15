"""Fixed outer-fold evaluation around the unchanged challenge Ridge probe."""
import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
from .error_analysis import classification_summary, write_csv


def make_folds(row_ids, labels, n_splits=5, seed=42):
    """Assign balanced stratified folds, independent of input row ordering."""
    row_ids, labels = np.asarray(row_ids), np.asarray(labels)
    if row_ids.ndim != 1 or labels.shape != row_ids.shape or len(labels) == 0:
        raise ValueError('Require nonempty aligned one-dimensional IDs and labels')
    if len(np.unique(row_ids)) != len(row_ids) or not np.issubdtype(row_ids.dtype, np.integer):
        raise ValueError('Row IDs must be unique integers')
    if not np.issubdtype(labels.dtype, np.integer) or np.any(labels < 0):
        raise ValueError('Use only labeled examples with nonnegative integer class IDs')
    classes, counts = np.unique(labels, return_counts=True)
    if n_splits < 2 or np.any(counts < n_splits):
        raise ValueError('Every class must have at least n_splits examples')
    rng = np.random.default_rng(seed)
    folds = np.empty(len(labels), dtype=np.int64)
    offset = int(rng.integers(n_splits))
    for c in classes:
        positions = np.flatnonzero(labels == c)
        positions = positions[np.argsort(row_ids[positions])]
        positions = rng.permutation(positions)
        folds[positions] = (np.arange(len(positions)) + offset) % n_splits
        offset = (offset + len(positions)) % n_splits
    return folds


def fixed_plan(path, row_ids, labels, data_sha256, n_splits=5, seed=42):
    """Create a manifest once, or reject an incompatible existing manifest."""
    folds = make_folds(row_ids, labels, n_splits, seed)
    assignments = [{'row_id': int(row_ids[i]), 'label': int(labels[i]), 'fold': int(folds[i])}
                   for i in np.argsort(row_ids)]
    expected = {'schema_version': 1, 'algorithm': 'sorted-class-shuffle-rotating-round-robin-v1',
                'seed': seed, 'n_splits': n_splits, 'data_sha256': data_sha256,
                'assignments': assignments}
    path = Path(path)
    if path.exists():
        if json.loads(path.read_text()) != expected:
            raise ValueError('Existing fold manifest differs in dataset, labels, IDs, seed, or assignments; refusing to overwrite')
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        # Exclusive creation prevents simultaneous jobs replacing each other's plan.
        with path.open('x') as f:
            json.dump(expected, f, indent=2)
            f.write('\n')
    return folds


def evaluate_folds(features, labels, folds, n_splits=5, probe_factory=None):
    """Fit a fresh official Ridge on each training split and predict its test split."""
    if probe_factory is None:
        from evaluate import Ridge
        probe_factory = Ridge
    features, labels, folds = np.asarray(features), np.asarray(labels), np.asarray(folds)
    if features.ndim != 2 or features.shape[0] != len(labels) or labels.shape != folds.shape:
        raise ValueError('Features, labels, and folds must be aligned')
    if not np.array_equal(np.unique(folds), np.arange(n_splits)):
        raise ValueError('Every outer fold must be present')
    if not np.isfinite(features).all():
        raise ValueError('Embeddings must be finite')
    predictions = np.empty_like(labels)
    records = []
    for fold in range(n_splits):
        train, test = folds != fold, folds == fold
        if not np.array_equal(np.unique(labels[train]), np.unique(labels)):
            raise ValueError('Every training split must contain all classes')
        probe = probe_factory().fit(features[train], labels[train])
        predictions[test] = probe.predict(features[test])
        metrics, _ = classification_summary(labels[test], predictions[test])
        records.append({'fold': fold, 'train_images': int(train.sum()),
                        'validation_images': int(test.sum()), 'ridge_lambda': float(probe.lam),
                        'accuracy': metrics['accuracy'], 'macro_accuracy': metrics['macro_accuracy']})
    summary, confusion = classification_summary(labels, predictions)
    summary.update(fold_mean_accuracy=float(np.mean([r['accuracy'] for r in records])),
                   fold_sd_accuracy=float(np.std([r['accuracy'] for r in records], ddof=1)))
    return summary, records, predictions, confusion


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--model', required=True, type=Path)
    p.add_argument('--untrained-model', type=Path)
    p.add_argument('--data', required=True, type=Path)
    p.add_argument('--folds', required=True, type=Path)
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--n-splits', type=int, default=5)
    a = p.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):
        p.error('Run evaluation in a Slurm allocation')
    import onnxruntime as ort
    from training_data import load_training_data
    from evaluate import Ridge, embed

    a.output.mkdir(parents=True, exist_ok=True)
    ds = load_training_data(str(a.data))
    all_labels = np.asarray(ds['label'])
    row_ids = np.flatnonzero(all_labels != -1)
    labels = all_labels[row_ids]
    ds = ds.select(row_ids)
    data_hash = file_hash(a.data)
    folds = fixed_plan(a.folds, row_ids, labels, data_hash, a.n_splits, a.seed)
    print('Fixed fold sizes:', np.bincount(folds).tolist(), flush=True)
    options = ort.SessionOptions()
    options.intra_op_num_threads = int(os.environ.get('SLURM_CPUS_PER_TASK', '2'))
    options.inter_op_num_threads = 1
    features, models = {}, {}
    for name, model in [('trained', a.model), ('untrained', a.untrained_model)]:
        if model is None:
            continue
        session = ort.InferenceSession(str(model), options, providers=['CPUExecutionProvider'])
        features[name] = embed(session, ds, batch=32)
        models[name] = {'path': str(model.resolve()), 'sha256': file_hash(model)}
        print(f'Extracted {name} embeddings', flush=True)
    images = np.stack([np.asarray(im.convert('RGB'), dtype=np.float32) / 255 for im in ds['image']])
    features['rgb_stats'] = np.concatenate([images.mean((1, 2)), images.std((1, 2))], axis=1)
    del images
    results, records, predictions, official_predictions = {}, [], {}, {}
    for name, embeddings in features.items():
        metrics, fold_results, pred, confusion = evaluate_folds(embeddings, labels, folds, a.n_splits)
        official = Ridge().fit(embeddings, labels)
        metrics['official_local_loo_accuracy'] = float(np.mean(official.loo_pred == labels))
        metrics['official_local_ridge_lambda'] = float(official.lam)
        results[name] = {'metrics': metrics, 'folds': fold_results}
        predictions[name], official_predictions[name] = pred, official.loo_pred
        records += [{'representation': name, **r} for r in fold_results]
        np.save(a.output / f'{name}_outer_confusion.npy', confusion)
        print(name, json.dumps({k:v for k,v in metrics.items() if k != 'per_class'}), flush=True)
    write_csv(a.output / 'fold_metrics.csv', records)
    prediction_rows = []
    for i, row_id in enumerate(row_ids):
        row = {'row_id': int(row_id), 'label': int(labels[i]), 'fold': int(folds[i])}
        for name in features:
            row[f'{name}_outer_prediction'] = int(predictions[name][i])
            row[f'{name}_official_loo_prediction'] = int(official_predictions[name][i])
        prediction_rows.append(row)
    write_csv(a.output / 'predictions.csv', prediction_rows)
    per_class = []
    for i, c in enumerate(np.unique(labels)):
        row = {'class_id': int(c), 'support': int((labels == c).sum())}
        for name in features:
            row[f'{name}_outer_accuracy'] = results[name]['metrics']['per_class'][i]['accuracy']
        per_class.append(row)
    write_csv(a.output / 'per_class.csv', per_class)
    np.savez_compressed(a.output / 'embeddings.npz', row_ids=row_ids, labels=labels, folds=folds, **features)
    result = {'protocol': 'Fixed stratified outer folds; each official Ridge.fit selects lambda by inner analytic LOO on its training split.',
              'data': str(a.data.resolve()), 'data_sha256': data_hash,
              'fold_manifest': str(a.folds.resolve()), 'fold_manifest_sha256': file_hash(a.folds),
              'seed': a.seed, 'n_splits': a.n_splits, 'images': len(labels),
              'classes': np.unique(labels).tolist(), 'models': models, 'results': results,
              'limitations': 'Frozen encoder saw all images without labels during SSL. Outer folds assess probe generalization, not unseen-image SSL transfer or new classes. Reusing these folds for encoder selection makes them development data. Fold SD is descriptive, not a confidence interval. Official local LOO also uses more probe training images than each outer fold.'}
    (a.output / 'results.json').write_text(json.dumps(result, indent=2))
    lines = ['# Fixed outer-fold probe evaluation', '', result['protocol'], '',
             f'{len(labels)} labeled images, {len(np.unique(labels))} classes, {a.n_splits} folds, seed {a.seed}.',
             f'Fold sizes: {np.bincount(folds).tolist()}. Fold manifest: `{a.folds}`.', '',
             '| Representation | Official local LOO | Outer accuracy | Outer macro | Fold SD (pp) |',
             '|---|---:|---:|---:|---:|']
    for name, r in results.items():
        m = r['metrics']
        lines.append(f'| {name} | {m["official_local_loo_accuracy"]:.2%} | {m["accuracy"]:.2%} | {m["macro_accuracy"]:.2%} | {100*m["fold_sd_accuracy"]:.2f} |')
    lines += ['', '## Fold results', '', '| Representation | Fold | Train | Validation | Accuracy | Ridge lambda |',
              '|---|---:|---:|---:|---:|---:|']
    for r in records:
        lines.append(f'| {r["representation"]} | {r["fold"]} | {r["train_images"]} | {r["validation_images"]} | {r["accuracy"]:.2%} | {r["ridge_lambda"]:.6g} |')
    lines += ['', '## Interpretation', '', result['limitations'], '',
              'No encoder is retrained and validation labels do not enter Ridge.fit or SSL gradients.',
              'Standardization and regularization selection are fitted on the outer training images only.',
              'This reuses the official inner LOO implementation, including its training-split-wide standardization; it is not a fully refitted inner preprocessing pipeline.',
              'RGB statistics are the six channel means/stds and are an analysis reference, not an ONNX submission.',
              'The same fold manifest must be passed to every future configuration. Do not search seeds/folds to improve a score.']
    (a.output / 'REPORT.md').write_text('\n'.join(lines) + '\n')
    print('Evaluation complete:', a.output / 'REPORT.md', flush=True)


if __name__ == '__main__':
    main()
