"""CPU evaluation of an ONNX encoder with a closed-form linear (ridge) probe.

    python evaluate.py model.onnx [--data data/sample.parquet] [--out results.json]

`--data` is any parquet file with an `image` (96x96 RGB) and a `label` column, such as data/sample.parquet or
data/train.parquet. Rows labelled -1 (unlabelled) are skipped. The score is the leave-one-out accuracy of a ridge
probe on the embeddings of the labelled images: each image is classified by a probe fit on all the others. It is
exact and closed form, so there is no train/test split to choose.

The leaderboard uses the same embedding and probe, on held-out images and new classes (see CHALLENGE.md).

Problems with the submitted model are reported as {"error": ...} on stdout. Prints {"score": <float>} as JSON on
stdout; the detailed report goes to stderr.
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import onnxruntime as ort
from datasets import Dataset

SIZE, DIM = 96, 1024
MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)

LAMBDAS = np.logspace(-4, 2, 13)  # relative to the mean eigenvalue of the Gram matrix
UNLABELLED = -1


class SubmissionError(Exception):
    """A problem with the submitted model, shown to the participant."""


def embed(session, ds, batch=256):
    x = np.stack([np.asarray(im, dtype=np.float32) for im in ds["image"]]) / 255.0
    x = ((x - MEAN) / STD).astype(np.float32).transpose(0, 3, 1, 2)
    out = []
    for i in range(0, len(x), batch):
        try:
            z = session.run(None, {"image": x[i : i + batch]})[0]
        except Exception as e:
            raise SubmissionError(
                f"Model failed on float32 input of shape {x[i : i + batch].shape}: {e}"
            ) from None
        if z.shape != (len(x[i : i + batch]), DIM):
            raise SubmissionError(f"Embedding shape {z.shape}, expected (batch, {DIM})")
        if not np.isfinite(z).all():
            raise SubmissionError("Embeddings contain NaN or infinite values")
        out.append(z)
    return np.concatenate(out)


class Ridge:
    """Closed-form ridge regression onto one-hot labels; argmax gives the class.

    Solved in the dual (n x n Gram matrix, one eigendecomposition). Lambda is picked by the exact
    leave-one-out error, which is also closed form, so there is no iterative optimisation anywhere.
    `loo_pred` holds each training point's prediction from the probe fit on all the other points.
    """

    def fit(self, x, y):
        self.mu, self.sd = x.mean(0), x.std(0) + 1e-6
        x = (x - self.mu) / self.sd
        self.classes = np.unique(y)
        Y = (y[:, None] == self.classes).astype(np.float64)
        self.ybar = Y.mean(0)
        Yc = Y - self.ybar
        e, U = np.linalg.eigh(x @ x.T)
        e = np.clip(e, 0, None)
        UtY = U.T @ Yc
        best = None
        for lam in LAMBDAS * e.mean():
            h = e / (e + lam)
            H_diag = (U**2) @ h + 1 / len(x)  # leverage, incl. the intercept
            loo = (Yc - U @ (h[:, None] * UtY)) / (1 - H_diag)[:, None]
            err = (loo**2).sum()
            if best is None or err < best[0]:
                best = (err, lam, loo)
        _, self.lam, loo = best
        self.loo_pred = self.classes[(Yc - loo + self.ybar).argmax(1)]
        self.W = x.T @ (U @ (UtY / (e + self.lam)[:, None]))
        return self

    def predict(self, x):
        return self.classes[(((x - self.mu) / self.sd) @ self.W + self.ybar).argmax(1)]


def evaluate(model_path, data="data/sample.parquet", threads=None):
    t0 = time.time()
    ds = Dataset.from_parquet(data)
    ds = ds.select(np.flatnonzero(np.asarray(ds["label"]) != UNLABELLED))
    if len(ds) < 2:
        raise RuntimeError(f"{data!r} has {len(ds)} labelled images; the probe needs at least 2")
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = threads if threads is not None else os.cpu_count()
    try:
        sess = ort.InferenceSession(
            model_path, opts, providers=["CPUExecutionProvider"]
        )
    except Exception as e:
        raise SubmissionError(f"Could not load the ONNX model: {e}") from None
    if [i.name for i in sess.get_inputs()] != ["image"]:
        raise SubmissionError("ONNX model must have exactly one input, named 'image'")
    z, y = embed(sess, ds), np.asarray(ds["label"])
    t_embed = time.time() - t0
    probe = Ridge().fit(z, y)
    return {
        "score": float((probe.loo_pred == y).mean()),
        "images": len(y),
        "classes": len(probe.classes),
        "lambda": float(probe.lam),
        "seconds": {"embed": round(t_embed, 1), "total": round(time.time() - t0, 1)},
    }


def print_report(r, file=sys.stderr):
    print(
        f"leave-one-out probe accuracy {100 * r['score']:.1f}  "
        f"({r['images']} images, {r['classes']} classes, chance {100 / r['classes']:.1f})\n"
        f"time: {r['seconds']}",
        file=file,
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("model", help="path to the ONNX model")
    ap.add_argument(
        "--data", default="data/sample.parquet", help="parquet file with image and label columns"
    )
    ap.add_argument("--out", default=None, help="also write the full results JSON here")
    a = ap.parse_args()
    if not os.path.isfile(a.data):
        sys.exit(f"Evaluation data not found at {a.data!r}; pass --data")
    try:
        r = evaluate(a.model, a.data)
    except SubmissionError as e:
        print(json.dumps({"error": str(e)}))
        sys.exit(1)
    print_report(r)
    if a.out:
        with open(a.out, "w") as f:
            json.dump(r, f, indent=2)
    print(json.dumps({"score": round(r["score"], 4)}))
