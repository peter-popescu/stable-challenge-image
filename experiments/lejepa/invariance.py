"""Paired controlled-transform diagnostics; no labels or training objectives."""
import argparse
import json
import os
from pathlib import Path


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('checkpoint', type=Path)
    p.add_argument('--data', required=True)
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--count', type=int, default=128)
    p.add_argument('--seed', type=int, default=42)
    a = p.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):
        p.error('Run diagnostics within Slurm')
    import numpy as np
    import torch
    from torchvision import transforms as T
    from torchvision.transforms import functional as F
    from training_data import load_training_data
    from .data import preprocessing
    from .model import load_encoder
    from .sweep import digest

    torch.set_num_threads(int(os.environ.get('SLURM_CPUS_PER_TASK', '2')))
    ds = load_training_data(a.data).select_columns(['image'])
    ids = np.random.default_rng(a.seed).choice(len(ds), min(a.count, len(ds)), replace=False)
    if len(ids) < 2:
        raise ValueError('Need at least two diagnostic images')
    images = [ds[int(i)]['image'].convert('RGB') for i in ids]
    model = load_encoder(a.checkpoint)

    def embed(images):
        x = torch.stack([preprocessing()(im) for im in images])
        with torch.inference_mode():
            return torch.cat([model(batch) for batch in x.split(16)]).numpy()

    base = embed(images)
    normalized = base / np.maximum(np.linalg.norm(base, axis=1, keepdims=True), 1e-12)
    records, arrays = {}, {'row_ids': ids, 'base': base}
    transforms = {'jitter': dict(brightness=.4, contrast=.4, saturation=.4, hue=.1),
                  'center_crop_area': [.3, .6, .9], 'grayscale': True,
                  'solarize_threshold': 128}
    transformed = {}
    jittered = []
    for row_id, image in zip(ids, images):
        # Reset per source image, independent of model construction and batch order.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(a.seed + int(row_id))
            jittered.append(T.ColorJitter(**transforms['jitter'])(image))
    transformed['jitter'] = jittered
    transformed['grayscale'] = [F.rgb_to_grayscale(im, num_output_channels=3) for im in images]
    transformed['solarize'] = [F.solarize(im, 128) for im in images]
    for area in transforms['center_crop_area']:
        transformed[f'crop-{area:g}'] = [
            F.resize(F.center_crop(im, [round(im.height * area**.5), round(im.width * area**.5)]),
                     [96, 96], interpolation=T.InterpolationMode.BILINEAR, antialias=True)
            for im in images]
    for name, altered in transformed.items():
        features = embed(altered)
        norm = features / np.maximum(np.linalg.norm(features, axis=1, keepdims=True), 1e-12)
        cosine = (normalized * norm).sum(1)
        relative_l2 = np.linalg.norm(features - base, axis=1) / np.maximum(np.linalg.norm(base, axis=1), 1e-12)
        arrays[name + '_cosine'] = cosine
        arrays[name + '_relative_l2'] = relative_l2
        records[name] = dict(mean_cosine=float(cosine.mean()),
                             cosine_sd=float(cosine.std(ddof=1)),
                             mean_relative_l2=float(relative_l2.mean()))
    report = dict(seed=a.seed, indices=ids.tolist(), transforms=transforms,
                  crop_position='center', crop_output=96, crop_ratio='square',
                  checkpoint_sha256=digest(a.checkpoint), data_sha256=digest(a.data),
                  measurements=records,
                  mean_variance=float(base.var(0).mean()),
                  mismatched_pair_cosine=float((normalized * np.roll(normalized, 1, axis=0)).sum(1).mean()),
                  limitations='High cosine can reflect collapse or anisotropy. Read with mismatched-pair similarity, variance, PCA rank and probe metrics. Central crops are controlled perturbations, not the random training crop distribution.')
    a.output.mkdir(parents=True, exist_ok=True)
    (a.output / 'invariance.json').write_text(json.dumps(report, indent=2) + '\n')
    np.savez_compressed(a.output / 'invariance.npz', **arrays)


if __name__ == '__main__':
    main()
