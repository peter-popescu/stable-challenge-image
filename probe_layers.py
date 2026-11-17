"""Which layer of a trained MAE encoder gives the most linearly useful features?

    uv run probe_layers.py runs/mae-baseline                       # score every layer / pooling / combination
    uv run probe_layers.py runs/mae-baseline --export 6,7,8,9      # also write that choice as an ONNX model

No training: loads <run>/last.pt, embeds the 750 labelled images with every transformer block, and scores
each option with the same local challenge score as challenge_score.py (in-domain + few-shot, per domain).

Why: MAE's last layers specialise in repainting pixels, so for a linear probe the best features often
sit in the middle of the network. The submitted embedding can be up to 1024 numbers, so we can also
concatenate several blocks (5 blocks x 192 = 960).
"""

import argparse
import json
import os
from types import SimpleNamespace

import numpy as np
import torch

from challenge_score import DOMAIN_OF_CLASS, few_shot_accuracy, in_domain_accuracy
from evaluate import UNLABELLED
from export_onnx import export_onnx
from mae_train import Encoder, block_outputs, build_mae, load_images, normalise, parse_layers, pool_tokens


def local_score(features, labels, domains):
    ind = np.mean(list(in_domain_accuracy(features, labels, domains).values()))
    few = np.mean(list(few_shot_accuracy(features, labels, domains).values()))
    return (ind + few) / 2, ind, few


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run", help="run directory containing last.pt and args.json")
    p.add_argument("--data", default="data/train.parquet")
    p.add_argument("--export", default="", help="block numbers to concatenate and export, e.g. 6,7,8,9")
    p.add_argument("--pool", default="mean", choices=["mean", "cls", "cls+mean"], help="pooling for --export")
    a = p.parse_args()

    with open(os.path.join(a.run, "args.json")) as f:
        run_args = SimpleNamespace(**json.load(f))
    mae, vit = build_mae(run_args)
    mae.load_state_dict(torch.load(os.path.join(a.run, "last.pt"), map_location="cpu", weights_only=False)["model"])
    vit.eval()

    images, labels = load_images(a.data)
    keep = labels != UNLABELLED
    images, labels = images[keep], labels[keep]
    domains = np.array([DOMAIN_OF_CLASS[y] for y in labels])

    # Embed once, keep every block's pooled features
    depth = len(vit.blocks)
    feats = {}
    with torch.no_grad():
        for i in range(0, len(images), 128):
            outs = block_outputs(vit, normalise(images[i:i + 128]))
            named = {f"block {b + 1}": t for b, t in enumerate(outs)}
            named["final (normed)"] = vit.norm(outs[-1])
            for name, tokens in named.items():
                for pool in ("mean", "cls"):
                    feats.setdefault((name, pool), []).append(pool_tokens(tokens, pool).numpy())
    feats = {k: np.concatenate(v).astype(np.float64) for k, v in feats.items()}

    candidates = {f"{name} | {pool}": f for (name, pool), f in feats.items()}
    dim = feats[("block 1", "mean")].shape[1]
    for k in range(2, 1024 // dim + 1):  # concatenations of k consecutive blocks, mean-pooled
        for start in range(1, depth - k + 2):
            blocks = range(start, start + k)
            candidates[f"blocks {start}-{start + k - 1} | mean ({k * dim}d)"] = np.concatenate(
                [feats[(f"block {b}", "mean")] for b in blocks], axis=1)

    rows = []
    for name, f in candidates.items():
        rows.append((name, *local_score(f, labels, domains)))
        print(f"  scored {name}", flush=True)

    print("\nPer block (mean-pooled patch tokens):")
    print(f"{'layer':<34}{'score':>8}{'in_dom':>8}{'few':>8}")
    for name, s, i, fs in rows:
        if name.endswith("| mean") and not name.startswith("blocks"):
            print(f"{name:<34}{s:8.4f}{i:8.4f}{fs:8.4f}")
    print("\nTop 15 overall:")
    for name, s, i, fs in sorted(rows, key=lambda r: -r[1])[:15]:
        print(f"{name:<34}{s:8.4f}{i:8.4f}{fs:8.4f}")
    with open(os.path.join(a.run, "probe_layers.json"), "w") as f:
        json.dump([dict(zip(("option", "score", "in_domain", "few_shot"), r)) for r in rows], f, indent=2)

    if a.export:
        layers = parse_layers(a.export)
        path = os.path.join(a.run, f"model_layers{'-'.join(map(str, layers))}_{a.pool}.onnx")
        export_onnx(Encoder(vit, a.pool, layers), path)


if __name__ == "__main__":
    main()
