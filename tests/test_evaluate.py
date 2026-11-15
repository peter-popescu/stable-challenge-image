import json
import os

import numpy as np
import onnxruntime as ort
import pytest
import torch
from conftest import CLASSES, MODEL, SAMPLE, last_json, run
from datasets import Dataset

from evaluate import DIM, Ridge, SubmissionError, embed, evaluate
from export_onnx import export_onnx


def test_ridge_separates_classes():
    rng = np.random.default_rng(0)
    y = np.repeat(np.arange(4), 25)
    x = np.eye(4)[y] * 5 + rng.normal(size=(len(y), 4))
    x = np.concatenate([x, rng.normal(size=(len(y), 60))], 1)  # plus noise features
    assert (Ridge().fit(x, y).predict(x) == y).mean() > 0.95


def test_ridge_keeps_original_labels():
    rng = np.random.default_rng(0)
    y = np.repeat([3, 7, 11], 10)
    x = rng.normal(size=(len(y), 8)) + y[:, None]
    assert set(Ridge().fit(x, y).predict(x)) <= {3, 7, 11}


def test_embed_sample_data():
    sess = ort.InferenceSession(str(MODEL), providers=["CPUExecutionProvider"])
    z = embed(sess, Dataset.from_parquet(str(SAMPLE)), batch=2)  # batch < len exercises the batching
    assert z.shape == (3, DIM) and np.isfinite(z).all()


def test_ridge_leave_one_out_matches_refitting():
    rng = np.random.default_rng(0)
    y = np.repeat(np.arange(4), 8)
    x = np.eye(4)[y] @ rng.normal(size=(4, 20)) + 1.5 * rng.normal(size=(len(y), 20))
    p = Ridge().fit(x, y)
    xs, Y = (x - p.mu) / p.sd, (y[:, None] == p.classes).astype(float)
    for i in range(len(y)):  # refit without point i (same standardisation and lambda), unpenalised intercept
        X, T = np.delete(xs, i, 0), np.delete(Y, i, 0)
        w = np.linalg.solve((X - X.mean(0)).T @ (X - X.mean(0)) + p.lam * np.eye(X.shape[1]),
                            (X - X.mean(0)).T @ (T - T.mean(0)))
        assert p.classes[((xs[i] - X.mean(0)) @ w + T.mean(0)).argmax()] == p.loo_pred[i]


def test_example_submission_on_synthetic_data(eval_data, tmp_path):
    out = tmp_path / "results.json"
    r = run("evaluate.py", MODEL, "--data", eval_data, "--out", out, cwd=tmp_path)
    assert r.returncode == 0, r.stderr
    res = json.loads(out.read_text())
    assert last_json(r.stdout)["score"] == res["score"] > 0.6  # 3 classes separated by colour; chance is 1/3
    assert res["images"] == 12 * CLASSES and res["classes"] == CLASSES  # unlabelled rows are skipped


def test_sample_data_runs(tmp_path):
    r = run("evaluate.py", MODEL, "--data", SAMPLE, cwd=tmp_path)
    assert r.returncode == 0, r.stderr
    assert 0 <= last_json(r.stdout)["score"] <= 1


class Flat(torch.nn.Module):
    def __init__(self, dim=DIM, nan=False):
        super().__init__()
        self.lin, self.nan = torch.nn.Linear(3, dim), nan

    def forward(self, x):
        z = self.lin(x.mean((2, 3)))
        return z * float("nan") if self.nan else z


def export(model, path, input_name="image"):
    torch.onnx.export(model.eval(), (torch.randn(2, 3, 96, 96),), path, input_names=[input_name],
                      output_names=["embedding"], dynamic_axes={input_name: {0: "batch"}}, opset_version=17,
                      dynamo=False)
    return path


@pytest.mark.parametrize("model, message", [
    (Flat(dim=256), "expected (batch, 1024)"),
    (Flat(nan=True), "NaN or infinite"),
])
def test_embed_rejects_bad_outputs(tmp_path, model, message):
    sess = ort.InferenceSession(str(export(model, tmp_path / "bad.onnx")), providers=["CPUExecutionProvider"])
    with pytest.raises(SubmissionError, match=message.replace("(", r"\(").replace(")", r"\)")):
        embed(sess, Dataset.from_parquet(str(SAMPLE)))


@pytest.mark.parametrize("model, input_name, message", [
    (Flat(), "pixels", "exactly one input, named 'image'"),
    (Flat(dim=256), "image", "expected (batch, 1024)"),
    (Flat(nan=True), "image", "NaN or infinite"),
])
def test_bad_submissions_report_error(eval_data, tmp_path, model, input_name, message):
    path = export(model, tmp_path / "bad.onnx", input_name)
    r = run("evaluate.py", path, "--data", eval_data, cwd=tmp_path)
    assert r.returncode == 1
    assert message in last_json(r.stdout)["error"]


def test_unloadable_model_reports_error(eval_data, tmp_path):
    path = tmp_path / "junk.onnx"
    path.write_bytes(os.urandom(64))
    r = run("evaluate.py", path, "--data", eval_data, cwd=tmp_path)
    assert r.returncode == 1
    assert "Could not load the ONNX model" in last_json(r.stdout)["error"]


def test_missing_data_dir(tmp_path):
    r = run("evaluate.py", MODEL, "--data", tmp_path / "nope", cwd=tmp_path)
    assert r.returncode == 1 and "pass --data" in r.stderr


def test_too_few_labelled_images_raises(eval_data, tmp_path):
    data = tmp_path / "one.parquet"
    Dataset.from_parquet(str(eval_data)).select([0]).to_parquet(data)
    with pytest.raises(RuntimeError, match="the probe needs at least 2"):
        evaluate(str(MODEL), str(data))


def test_export_onnx_pads_small_encoder(tmp_path):
    path = export_onnx(Flat(dim=256), str(tmp_path / "model.onnx"))
    z = ort.InferenceSession(path, providers=["CPUExecutionProvider"]).run(
        None, {"image": np.zeros((1, 3, 96, 96), np.float32)})[0]
    assert z.shape == (1, DIM) and not z[:, 256:].any()


def test_export_onnx_rejects_too_many_features(tmp_path):
    with pytest.raises(ValueError, match="more than the 1024 allowed"):
        export_onnx(Flat(dim=2048), str(tmp_path / "model.onnx"))


@pytest.mark.parametrize("allocation,expected", [(None, 2), ("1", 1), ("8", 4)])
def test_export_validation_bounds_threads_to_allocation(tmp_path, monkeypatch, allocation, expected):
    import export_onnx as exporter
    if allocation is None:
        monkeypatch.delenv("SLURM_CPUS_PER_TASK", raising=False)
    else:
        monkeypatch.setenv("SLURM_CPUS_PER_TASK", allocation)
    original = ort.InferenceSession
    captured = []
    def session(*args, **kwargs):
        captured.append(kwargs["sess_options"])
        return original(*args, **kwargs)
    monkeypatch.setattr(exporter.ort, "InferenceSession", session)
    previous = torch.get_num_threads()
    torch.set_num_threads(4)
    try:
        export_onnx(Flat(dim=256), str(tmp_path / "threads.onnx"))
        assert captured[0].intra_op_num_threads == expected
        assert captured[0].inter_op_num_threads == 1
    finally:
        torch.set_num_threads(previous)
