"""Create a fixed image-disjoint diagnostic split; never used for official scoring."""

import argparse
import json
from pathlib import Path

import numpy as np
from datasets import Dataset

from diagnose import file_hash


def prepare_holdout(data, output, fraction=0.2, seed=2026):
    if not 0 < fraction < 1:
        raise ValueError("fraction must be between 0 and 1")
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"{output} already exists; reuse its files or choose a new directory")
    dataset = Dataset.from_parquet(str(data))
    labels = np.asarray(dataset["label"])
    rng = np.random.default_rng(seed)
    training, holdout = [], []
    # Stratification is only for constructing evaluation partitions, never a
    # label-dependent SSL loss. Unlabelled images are split in the same way.
    for label in np.unique(labels):
        indices = rng.permutation(np.flatnonzero(labels == label))
        n = min(len(indices) - 1, max(1, round(len(indices) * fraction))) if len(indices) > 1 else 0
        holdout.extend(indices[:n].tolist())
        training.extend(indices[n:].tolist())
    if not training or not holdout:
        raise ValueError("Not enough images to create two nonempty partitions")
    training, holdout = sorted(training), sorted(holdout)
    output.mkdir(parents=True)
    dataset.select(training).to_parquet(str(output / "train.parquet"))
    dataset.select(holdout).to_parquet(str(output / "holdout.parquet"))
    manifest = {"source": str(data), "source_sha256": file_hash(data),
                "seed": seed, "fraction": fraction,
                "train_indices": training, "holdout_indices": holdout,
                "train_sha256": file_hash(output / "train.parquet"),
                "holdout_sha256": file_hash(output / "holdout.parquet"),
                "train_images": len(training), "holdout_images": len(holdout),
                "train_labelled": int(np.sum(labels[training] != -1)),
                "holdout_labelled": int(np.sum(labels[holdout] != -1))}
    (output / "split.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/train.parquet"))
    parser.add_argument("--out", type=Path, default=Path("data/holdout-v1"))
    parser.add_argument("--fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    manifest = prepare_holdout(args.data, args.out, args.fraction, args.seed)
    print(f"Training: {manifest['train_images']} images ({manifest['train_labelled']} labelled); "
          f"holdout: {manifest['holdout_images']} images ({manifest['holdout_labelled']} labelled)")
    print(f"Saved split and source indices to {args.out}. Train fresh models on train.parquet only.")


if __name__ == "__main__":
    main()
