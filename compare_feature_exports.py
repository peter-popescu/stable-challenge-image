"""Compare final-stage and multiscale features from the exact same saved encoder on CPU."""

import argparse
import json
from pathlib import Path

import torch

from diagnose import diagnose, file_hash, save_mistake_sheet
from main import ChallengeDINO
from submission import export_teacher


def compare_exports(checkpoint, data, output, support_data=None, threads=2):
    checkpoint, output = Path(checkpoint), Path(output)
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    # Keep original run artifacts and any previous comparison intact.
    if output.exists():
        raise FileExistsError(f"{output} already exists; use a new output directory")
    model = ChallengeDINO.load_from_checkpoint(str(checkpoint), map_location="cpu")
    if not str(model.hparams.backbone).startswith("resnet"):
        raise ValueError("This comparison requires a compact ResNet checkpoint")
    output.mkdir(parents=True)
    results = {}
    for features in ("final", "multiscale"):
        directory = output / features
        path = export_teacher(model, directory / "model.onnx", features=features)
        result = diagnose(path, data, support_data=support_data, threads=threads)
        result["export_features"] = features
        result["model_size_mib"] = path.stat().st_size / 1024**2
        (directory / "diagnostics.json").write_text(json.dumps(result, indent=2))
        save_mistake_sheet(data, result["mistakes"], directory / "mistakes.png")
        results[features] = result
        print(f"{features}: {result['score']:.2%} accuracy; {result['macro_accuracy']:.2%} macro", flush=True)
    result = {
        "checkpoint": str(checkpoint), "checkpoint_sha256": file_hash(checkpoint),
        "protocol": results["final"]["protocol"],
        "final": results["final"], "multiscale": results["multiscale"],
        "accuracy_change": results["multiscale"]["score"] - results["final"]["score"],
        "macro_accuracy_change": results["multiscale"]["macro_accuracy"] - results["final"]["macro_accuracy"],
    }
    (output / "comparison.json").write_text(json.dumps(result, indent=2))
    print(f"Multiscale change: {100 * result['accuracy_change']:+.2f} percentage points; "
          f"macro {100 * result['macro_accuracy_change']:+.2f}", flush=True)
    print(f"Saved both models, per-class reports, and comparison.json to {output}", flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--data", type=Path, default=Path("data/train.parquet"))
    parser.add_argument("--support-data", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("threads must be positive")
    torch.set_num_threads(args.threads)
    compare_exports(args.checkpoint, args.data, args.out, args.support_data, args.threads)


if __name__ == "__main__":
    main()
