"""Deterministic error analysis using the unchanged official embedding/probe code."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np
from PIL import Image, ImageDraw


def classification_summary(labels, predictions):
    """Return supports and a true-row/predicted-column confusion matrix."""
    labels, predictions = np.asarray(labels), np.asarray(predictions)
    classes = np.unique(labels)
    if labels.shape != predictions.shape or not np.isin(predictions, classes).all():
        raise ValueError('Predictions must match label shape and use observed classes')
    lookup = {int(c): i for i, c in enumerate(classes)}
    confusion = np.zeros((len(classes), len(classes)), dtype=np.int64)
    for actual, predicted in zip(labels, predictions):
        confusion[lookup[int(actual)], lookup[int(predicted)]] += 1
    support = confusion.sum(1)
    per_class = [{'class_id': int(c), 'support': int(support[i]),
                  'correct': int(confusion[i, i]),
                  'accuracy': float(confusion[i, i] / support[i])}
                 for i, c in enumerate(classes)]
    return {'accuracy': float(np.mean(labels == predictions)),
            'macro_accuracy': float(np.mean([r['accuracy'] for r in per_class])),
            'per_class': per_class}, confusion


def neighbors(features, k=5):
    """Cosine retrieval after the official probe's dimension standardization."""
    standardized = (features - features.mean(0)) / (features.std(0) + 1e-6)
    normalized = standardized / np.maximum(np.linalg.norm(standardized, axis=1, keepdims=True), 1e-12)
    similarity = normalized @ normalized.T
    np.fill_diagonal(similarity, -np.inf)
    order = np.argsort(-similarity, axis=1, kind='stable')[:, :min(k, len(features) - 1)]
    return order, np.take_along_axis(similarity, order, axis=1)


def write_csv(path, rows):
    if not rows:
        return
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def sheet(path, selections, images, indices, labels, predictions, nearest=None):
    """Save fixed-size pages with row IDs and actual/predicted numeric classes."""
    pages = []
    for page, start in enumerate(range(0, len(selections), 12)):
        rows = selections[start:start + 12]
        columns = 4 if nearest is not None else 1
        canvas = Image.new('RGB', (columns * 180, len(rows) * 142), 'white')
        draw = ImageDraw.Draw(canvas)
        for row, query in enumerate(rows):
            cells = [int(query)]
            if nearest is not None:
                cells += nearest[query, :3].tolist()
            for col, item in enumerate(cells):
                x, y = col * 180, row * 142
                canvas.paste(images[item].resize((96, 96)), (x, y + 40))
                title = f'{"query" if col == 0 else "neighbor"} row {indices[item]}\ny={labels[item]} probe={predictions[item]}'
                draw.multiline_text((x + 2, y + 2), title, fill='black', spacing=2)
        dest = path.with_name(f'{path.stem}-{page + 1:02d}.png')
        canvas.save(dest)
        pages.append(dest.name)
    return pages


def file_hash(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            result.update(chunk)
    return result.hexdigest()


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--analysis', type=Path, required=True, help='Directory containing trained and untrained ONNX files')
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--data', required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--seed', type=int, default=42)
    a = p.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):
        p.error('Run embedding extraction and analysis within Slurm')
    import onnxruntime as ort
    from training_data import load_training_data
    from evaluate import Ridge, embed
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    a.output.mkdir(parents=True, exist_ok=True)
    ds = load_training_data(a.data)
    all_labels = np.asarray(ds['label'])
    indices = np.flatnonzero(all_labels != -1)
    labeled = ds.select(indices)
    labels = all_labels[indices]
    classes = np.unique(labels)
    images = [im.convert('RGB') for im in labeled['image']]
    rng = np.random.default_rng(a.seed)
    options = ort.SessionOptions()
    options.intra_op_num_threads = int(os.environ.get('SLURM_CPUS_PER_TASK', '2'))
    options.inter_op_num_threads = 1
    features = {}
    model_hashes = {}
    for name, filename in [('trained', 'model.onnx'), ('untrained', 'untrained.onnx')]:
        path = a.analysis / filename
        session = ort.InferenceSession(str(path), options, providers=['CPUExecutionProvider'])
        features[name] = embed(session, labeled, batch=32)
        model_hashes[name] = file_hash(path)
        print(f'Embedded {name}: {features[name].shape}', flush=True)
    pixels = np.stack([np.asarray(im, dtype=np.float32) / 255 for im in images])
    features['rgb_stats'] = np.concatenate([pixels.mean((1, 2)), pixels.std((1, 2))], axis=1)
    del pixels
    predictions, summaries, confusion, nearest, similarity = {}, {}, {}, {}, {}
    for name, z in features.items():
        probe = Ridge().fit(z, labels)
        predictions[name] = probe.loo_pred
        summaries[name], confusion[name] = classification_summary(labels, probe.loo_pred)
        summaries[name]['ridge_lambda'] = float(probe.lam)
        nearest[name], similarity[name] = neighbors(z)
        agreements = labels[nearest[name]] == labels[:, None]
        summaries[name]['nearest_neighbor_accuracy'] = float(agreements[:, 0].mean())
        summaries[name]['top5_neighbor_label_fraction'] = float(agreements.mean())
        np.save(a.output / f'{name}_confusion.npy', confusion[name])
        print(name, json.dumps({k:v for k,v in summaries[name].items() if k != 'per_class'}), flush=True)
    prior_path = a.analysis / 'report.json'
    if prior_path.exists():
        prior = json.loads(prior_path.read_text())
        for name in ['trained', 'untrained']:
            if not np.isclose(summaries[name]['accuracy'], prior[name]['score'], atol=1e-12):
                raise AssertionError(f'{name} score differs from the recorded official evaluation')
    correct = {name: pred == labels for name, pred in predictions.items()}
    categories = {'errors': np.flatnonzero(~correct['trained']),
                  'corrected': np.flatnonzero(correct['trained'] & ~correct['untrained']),
                  'regressions': np.flatnonzero(~correct['trained'] & correct['untrained']),
                  'both_wrong': np.flatnonzero(~correct['trained'] & ~correct['untrained'])}
    rows = []
    for i, row_id in enumerate(indices):
        row = {'row_id': int(row_id), 'label': int(labels[i])}
        for name in features:
            row[f'{name}_prediction'] = int(predictions[name][i])
            row[f'{name}_correct'] = bool(correct[name][i])
            row[f'{name}_neighbor_row_ids'] = ' '.join(str(indices[j]) for j in nearest[name][i])
            row[f'{name}_neighbor_labels'] = ' '.join(str(labels[j]) for j in nearest[name][i])
        rows.append(row)
    write_csv(a.output / 'predictions.csv', rows)
    per_class = []
    for i, c in enumerate(classes):
        row = {'class_id': int(c), 'support': summaries['trained']['per_class'][i]['support']}
        for name in features:
            row[f'{name}_accuracy'] = summaries[name]['per_class'][i]['accuracy']
        row['trained_minus_untrained'] = row['trained_accuracy'] - row['untrained_accuracy']
        per_class.append(row)
    write_csv(a.output / 'per_class.csv', per_class)
    pairs = []
    for i, actual in enumerate(classes):
        for j, predicted in enumerate(classes):
            count = int(confusion['trained'][i, j])
            if i != j and count:
                pairs.append({'true_class': int(actual), 'predicted_class': int(predicted),
                              'count': count, 'fraction_of_true_class': count / int((labels == actual).sum())})
    pairs.sort(key=lambda r: (-r['count'], r['true_class'], r['predicted_class']))
    write_csv(a.output / 'confusion_pairs.csv', pairs)
    fig, axes = plt.subplots(1, 2, figsize=(17, 7))
    matrix = confusion['trained']
    for ax, values, title in zip(axes, [matrix, matrix / matrix.sum(1, keepdims=True)],
                                 ['Counts', 'Fraction within true class']):
        image = ax.imshow(values, cmap='Blues', vmin=0)
        ax.set(xticks=range(len(classes)), yticks=range(len(classes)),
               xticklabels=classes, yticklabels=classes, xlabel='Predicted class', ylabel='True class', title=title)
        ax.tick_params(labelsize=7)
        fig.colorbar(image, ax=ax, fraction=.046)
    fig.tight_layout()
    fig.savefig(a.output / 'confusion.png', dpi=160)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(13, 5))
    position = np.arange(len(classes))
    for offset, name in zip([-.25, 0, .25], features):
        ax.bar(position + offset, [r[f'{name}_accuracy'] for r in per_class], width=.25, label=name)
    ax.set(xticks=position, xticklabels=classes, ylabel='Official LOO accuracy', xlabel='Class ID', ylim=(0, 1))
    ax.legend()
    fig.tight_layout()
    fig.savefig(a.output / 'per_class.png', dpi=160)
    plt.close(fig)
    galleries = {}
    chosen_errors = []
    for category, candidates in categories.items():
        selected = rng.permutation(candidates)[:24].tolist()
        galleries[category] = sheet(a.output / category, selected, images, indices, labels,
                                    predictions['trained'], nearest['trained'])
    for pair in pairs[:12]:
        candidates = np.flatnonzero((labels == pair['true_class']) &
                                   (predictions['trained'] == pair['predicted_class']))
        selected = rng.permutation(candidates)[:6].tolist()
        key = f'confusion-{pair["true_class"]}-to-{pair["predicted_class"]}'
        galleries[key] = sheet(a.output / key, selected, images, indices, labels,
                               predictions['trained'], nearest['trained'])
        chosen_errors += selected[:1]
    class_examples = []
    for c in classes:
        class_examples += rng.permutation(np.flatnonzero(labels == c))[:3].tolist()
    galleries['class_examples'] = sheet(a.output / 'class-examples', class_examples, images,
                                      indices, labels, predictions['trained'])
    result = {'seed': a.seed, 'images': len(labels), 'classes': classes.tolist(),
              'models_sha256': model_hashes, 'data_sha256': file_hash(a.data),
              'metrics': summaries, 'categories': {k:len(v) for k,v in categories.items()},
              'both_correct': int((correct['trained'] & correct['untrained']).sum()),
              'top_confusions': pairs[:20], 'galleries': galleries,
              'domain_mapping': 'Unavailable; class IDs are not assumed to identify domains.',
              'retrieval': 'Cosine of standardized embeddings, among labeled images only, excluding self.',
              'limitations': 'One seed; development images seen unlabeled during SSL; no outer CV. RGB statistics are a six-dimensional mean/std baseline, not a causal test of color reliance.'}
    np.savez_compressed(a.output / 'embeddings.npz', row_ids=indices, labels=labels,
                        **features, **{f'{k}_predictions':v for k,v in predictions.items()})
    (a.output / 'summary.json').write_text(json.dumps(result, indent=2))
    lines = ['# LeJEPA error analysis', '',
             f'{len(labels)} labeled training images; {len(classes)} observed classes. Labels used only for analysis.', '',
             '| Representation | Official LOO | Macro accuracy | Standardized cosine 1-NN |',
             '|---|---:|---:|---:|']
    for name, m in summaries.items():
        lines.append(f'| {name} | {m["accuracy"]:.1%} | {m["macro_accuracy"]:.1%} | {m["nearest_neighbor_accuracy"]:.1%} |')
    lines += ['', f'Compared with untrained features: {len(categories["corrected"])} examples fixed, '
              f'{len(categories["regressions"])} regressions; {len(categories["both_wrong"])} wrong under both probes.',
              '', '## Weakest classes', '', '| Class | Count | Trained | Untrained | RGB statistics |', '|---|---:|---:|---:|---:|']
    for r in sorted(per_class, key=lambda r: (r['trained_accuracy'], r['class_id']))[:10]:
        lines.append(f'| {r["class_id"]} | {r["support"]} | {r["trained_accuracy"]:.1%} | {r["untrained_accuracy"]:.1%} | {r["rgb_stats_accuracy"]:.1%} |')
    lines += ['', '## Largest directed confusions', '', '| True class | Predicted | Errors | Fraction of true class |', '|---|---|---:|---:|']
    for r in pairs[:12]:
        lines.append(f'| {r["true_class"]} | {r["predicted_class"]} | {r["count"]} | {r["fraction_of_true_class"]:.1%} |')
    lines += ['', '## Interpretation limits', '', result['limitations'], '', result['domain_mapping'],
              'Class supports and macro accuracy matter: small classes can have highly variable accuracy.',
              'Retrieval panels are diagnostic examples, not independent test cases or confidence-ranked errors.',
              'The probe is standardized and tuned exactly as in evaluate.py; its LOO estimates are not outer-CV scores.',
              '', 'See predictions.csv, per_class.csv, confusion_pairs.csv and summary.json for complete numerical results.',
              'Contact sheets show row IDs, actual labels and probe predictions. Neighbors exclude the query itself.']
    (a.output / 'REPORT.md').write_text('\n'.join(lines) + '\n')
    import html
    contents = ['<!doctype html><meta charset="utf-8"><title>LeJEPA error analysis</title>',
                '<style>body{font:16px sans-serif;max-width:1100px;margin:30px auto}img{max-width:100%}pre{white-space:pre-wrap}</style>',
                '<h1>LeJEPA error analysis</h1><pre>' + html.escape('\n'.join(lines)) + '</pre>',
                '<h2>Per-class accuracy</h2><img src="per_class.png">',
                '<h2>Confusion matrix</h2><img src="confusion.png">']
    for name, pages in galleries.items():
        contents.append('<h2>' + html.escape(name) + '</h2>')
        contents.extend('<img loading="lazy" src="' + page + '">' for page in pages)
    (a.output / 'index.html').write_text('\n'.join(contents))
    if chosen_errors:
        from . import diagnostics
        row_ids = ','.join(str(indices[i]) for i in dict.fromkeys(chosen_errors))
        for shared in [False, True]:
            sys.argv = ['diagnostics', a.checkpoint, '--data', a.data,
                        '--indices', row_ids, '--seed', str(a.seed), '--output',
                        str(a.output / ('error_pca_shared' if shared else 'error_pca'))] + (['--shared'] if shared else [])
            diagnostics.main()
    print('Analysis complete:', a.output / 'REPORT.md', flush=True)


if __name__ == '__main__':
    main()
