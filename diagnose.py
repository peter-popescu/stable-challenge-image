"""Local per-class diagnostics using the unchanged official embedding/ridge code."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
from datasets import Dataset
from PIL import Image, ImageDraw

from evaluate import Ridge, UNLABELLED, embed
from probe_diagnostics import domain_details, low_support_probes


def file_hash(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def labelled_data(path):
    dataset = Dataset.from_parquet(str(path))
    indices = np.flatnonzero(np.asarray(dataset["label"]) != UNLABELLED)
    return dataset.select(indices), indices


def classification_details(labels, predictions, row_indices):
    classes = np.unique(np.concatenate([labels, predictions]))
    positions = {int(label): i for i, label in enumerate(classes)}
    matrix = np.zeros((len(classes), len(classes)), dtype=int)
    for actual, predicted in zip(labels, predictions):
        matrix[positions[int(actual)], positions[int(predicted)]] += 1
    per_class = []
    for i, label in enumerate(classes):
        count = int(matrix[i].sum())
        per_class.append({"class_id": int(label), "images": count,
                          "correct": int(matrix[i, i]),
                          "accuracy": float(matrix[i, i] / count) if count else None})
    return {
        "score": float(np.mean(labels == predictions)),
        "macro_accuracy": float(np.mean([row["accuracy"] for row in per_class if row["images"]])),
        "images": len(labels), "classes": int(len(np.unique(labels))),
        "class_ids": classes.tolist(), "per_class": per_class,
        "confusion_matrix": matrix.tolist(),
        "domains": domain_details(labels, predictions),
        "domain_grouping": "Manually inferred class groups; not official dataset metadata",
        "mistakes": [{"row_index": int(row_indices[i]), "true_label": int(labels[i]),
                      "predicted_label": int(predictions[i])}
                     for i in np.flatnonzero(labels != predictions)],
    }


def diagnose(model, data, support_data=None, threads=2, low_shot=0, repeats=10,
             seed=2026, embeddings_out=None):
    """LOO on data, or fit on support_data and predict data without query labels."""
    dataset, indices = labelled_data(data)
    if len(dataset) < 2:
        raise ValueError("Diagnostics require at least two labelled evaluation images")
    if support_data and Path(support_data).resolve() == Path(data).resolve():
        raise ValueError("Support and query files must differ; use LOO for one file")
    options = ort.SessionOptions()
    options.intra_op_num_threads = threads
    session = ort.InferenceSession(str(model), options, providers=["CPUExecutionProvider"])
    features = embed(session, dataset)
    labels = np.asarray(dataset["label"])
    support_features, support_labels, support_indices = features, labels, indices
    if support_data:
        support, support_indices = labelled_data(support_data)
        if len(support) < 2:
            raise ValueError("Support data requires at least two labelled images")
        support_labels = np.asarray(support["label"])
        if not set(labels).issubset(set(support_labels)):
            raise ValueError("Every query class must have labelled support examples")
        support_features = embed(session, support)
        probe = Ridge().fit(support_features, support_labels)
        predictions = probe.predict(features)
    else:
        probe = Ridge().fit(features, labels)
        predictions = probe.loo_pred
    result = classification_details(labels, predictions, indices)
    result.update({"protocol": "support_query" if support_data else "loo",
                   "lambda": float(probe.lam), "model_sha256": file_hash(model),
                   "data": str(data), "data_sha256": file_hash(data),
                   "support_data": str(support_data) if support_data else None,
                   "support_sha256": file_hash(support_data) if support_data else None})
    if low_shot:
        result["low_support"] = low_support_probes(
            support_features, support_labels, features, labels,
            shots=low_shot, repeats=repeats, seed=seed, same_pool=not support_data,
            support_indices=support_indices, query_indices=indices)
    if embeddings_out:
        path = Path(embeddings_out)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, features=features, labels=labels, row_indices=indices,
                            support_features=support_features, support_labels=support_labels,
                            support_row_indices=support_indices,
                            model_sha256=result["model_sha256"], data_sha256=result["data_sha256"],
                            support_sha256=result["support_sha256"] or "",
                            same_pool=not support_data)
    return result


def save_mistake_sheet(data, mistakes, path, limit=24):
    """Show deterministic examples of errors; labels are class IDs, not names."""
    mistakes = mistakes[:limit]
    if not mistakes:
        Path(path).unlink(missing_ok=True)
        return
    dataset = Dataset.from_parquet(str(data))
    sheet = Image.new("RGB", (4 * 150, ((len(mistakes) + 3) // 4) * 140), "white")
    draw = ImageDraw.Draw(sheet)
    for i, error in enumerate(mistakes):
        x, y = (i % 4) * 150, (i // 4) * 140
        picture = dataset[error["row_index"]]["image"].convert("RGB")
        picture.thumbnail((96, 96))
        sheet.paste(picture, (x, y))
        draw.text((x, y + 98), f"row {error['row_index']}", fill="black")
        draw.text((x, y + 113), f"true {error['true_label']} / pred {error['predicted_label']}", fill="black")
    sheet.save(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("--data", type=Path, default=Path("data/train.parquet"))
    parser.add_argument("--support-data", type=Path)
    parser.add_argument("--out", type=Path, required=True, help="Directory for diagnostics.json and mistakes.png")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--low-shot", type=int, default=0, help="Also fit K-support/class diagnostic probes")
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--seed", type=int, default=2026, help="Support sampling seed, independent of training")
    parser.add_argument("--save-embeddings", action="store_true", help="Cache features for CPU-only reanalysis")
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("threads must be positive")
    if args.low_shot < 0 or args.repeats < 1:
        parser.error("low-shot must be nonnegative and repeats positive")
    result = diagnose(args.model, args.data, args.support_data, args.threads,
                      low_shot=args.low_shot, repeats=args.repeats, seed=args.seed,
                      embeddings_out=args.out / "embeddings.npz" if args.save_embeddings else None)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "diagnostics.json").write_text(json.dumps(result, indent=2))
    save_mistake_sheet(args.data, result["mistakes"], args.out / "mistakes.png")
    print(f"{result['protocol']}: accuracy {result['score']:.2%}; "
          f"macro accuracy {result['macro_accuracy']:.2%}")
    for name, domain in result["domains"].items():
        print(f"{name}: {domain['accuracy']:.2%} ({domain['correct']}/{domain['images']})")
    if "low_support" in result:
        low = result["low_support"]
        print(f"{low['shots']}-support/class diagnostic: {low['accuracy_mean']:.2%} mean; "
              f"{low['accuracy_std']:.2%} support-draw std ({low['repeats']} draws)")
    for row in sorted(result["per_class"], key=lambda row: row["accuracy"] if row["images"] else 2):
        if row["images"]:
            print(f"class {row['class_id']:>3}: {row['accuracy']:6.1%} "
                  f"({row['correct']}/{row['images']})")
    print(f"Saved diagnostics to {args.out}")


if __name__ == "__main__":
    main()
