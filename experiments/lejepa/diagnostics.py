"""Deterministic qualitative PCA and retrieval; no segmentation claims."""
import argparse
import json
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
import torch
from training_data import load_training_data
from .data import preprocessing
from .model import Encoder, load_encoder


def fit_pca(x):
    x = x.astype(np.float64)
    mean = x.mean(0)
    _, s, vt = np.linalg.svd(x - mean, full_matrices=False)
    basis = vt[:3].copy()
    # Resolve SVD sign ambiguity for repeatable colors.
    for row in basis:
        row *= np.sign(row[np.abs(row).argmax()]) or 1
    projected = (x - mean) @ basis.T
    lo, hi = np.percentile(projected, [1, 99], axis=0)
    return mean, basis, lo, hi, (s[:3]**2 / max((s**2).sum(), 1e-12)).tolist()


def color(x, fit):
    mean, basis, lo, hi, _ = fit
    return np.uint8(255 * np.clip(((x - mean) @ basis.T - lo) / np.maximum(hi - lo, 1e-8), 0, 1))


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('checkpoint')
    p.add_argument('--data', required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--count', type=int, default=12)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--shared', action='store_true', help='Shared basis and scaling across images')
    p.add_argument('--indices', help='Comma separated row IDs; overrides random selection')
    a = p.parse_args()
    torch.set_num_threads(1)
    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    ds = load_training_data(a.data)
    ids = ([int(i) for i in a.indices.split(',')] if a.indices else
           rng.choice(len(ds), min(a.count, len(ds)), replace=False).tolist())
    images = [ds[i]['image'].convert('RGB') for i in ids]
    x = torch.stack([preprocessing()(im) for im in images])
    trained = load_encoder(a.checkpoint)
    torch.manual_seed(a.seed)
    untrained = Encoder().eval()
    with torch.no_grad():
        maps = []
        for model in [trained, untrained]:
            maps.append(torch.cat([model.spatial(v) for v in x.split(2)]).permute(0, 2, 3, 1).numpy())
        embeddings = torch.cat([trained(v) for v in x.split(2)]).numpy()
    a.output.mkdir(parents=True, exist_ok=True)
    report = {'indices': ids, 'seed': a.seed, 'shared': a.shared, 'explained_variance': {}}
    panels = [[im] for im in images]
    for name, features in [('trained', maps[0]), ('untrained', maps[1]),
                           ('rgb', np.stack([np.asarray(im) / 255 for im in images]))]:
        flat = features.reshape(-1, features.shape[-1])
        common = fit_pca(flat[rng.choice(len(flat), min(10000, len(flat)), replace=False)]) if a.shared else None
        report['explained_variance'][name] = []
        for j, f in enumerate(features):
            vectors = f.reshape(-1, f.shape[-1])
            fit = common if common is not None else fit_pca(vectors)
            report['explained_variance'][name].append(fit[-1])
            rgb = color(vectors, fit).reshape(*f.shape[:2], 3)
            panel = Image.fromarray(rgb).resize((96, 96), Image.Resampling.BILINEAR)
            panels[j].append(panel)
            if name == 'trained':
                panels[j].append(Image.blend(images[j], panel, 0.5))
    canvas = Image.new('RGB', (5 * 96, len(ids) * 116), 'white')
    draw = ImageDraw.Draw(canvas)
    for j, row in enumerate(panels):
        for k, panel in enumerate(row):
            canvas.paste(panel, (k * 96, j * 116 + 20))
        draw.text((0, j * 116), f'row {ids[j]}: image | trained | overlay | random | RGB', fill='black')
    canvas.save(a.output / 'spatial.png')
    centered = embeddings - embeddings.mean(0)
    singular = np.linalg.svd(centered, compute_uv=False)
    probabilities = singular / max(singular.sum(), 1e-12)
    report.update(mean_variance=float(embeddings.var(0).mean()),
                  effective_rank=float(np.exp(-(probabilities * np.log(probabilities + 1e-12)).sum())),
                  embedding_count=len(ids))
    normalized = embeddings / np.maximum(np.linalg.norm(embeddings, axis=1, keepdims=True), 1e-12)
    similarity = normalized @ normalized.T
    np.fill_diagonal(similarity, -np.inf)
    retrieval = Image.new('RGB', (4 * 96, len(ids) * 96), 'white')
    neighbors = np.argsort(-similarity, axis=1)[:, :min(3, len(ids) - 1)]
    for j in range(len(ids)):
        for k, index in enumerate([j] + neighbors[j].tolist()):
            retrieval.paste(images[index], (k * 96, j * 96))
    retrieval.save(a.output / 'neighbors.png')
    report['neighbor_indices'] = [[ids[k] for k in row] for row in neighbors]
    (a.output / 'diagnostics.json').write_text(json.dumps(report, indent=2))
    np.savez(a.output / 'embeddings.npz', embeddings=embeddings, indices=ids)


if __name__ == '__main__':
    main()
