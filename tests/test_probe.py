import shutil
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from conftest import MODEL
from evaluate import evaluate
from probe import ChallengeProbe


def test_probe_matches_official_evaluator_and_cleans_up(eval_data, monkeypatch):
    exported = []

    def export(model, path):
        exported.append(Path(path))
        shutil.copyfile(MODEL, path)
        return path

    monkeypatch.setattr("probe.export_teacher", export)
    records = []
    logger = SimpleNamespace(log_metrics=lambda metrics, step: records.append((metrics, step)))
    trainer = SimpleNamespace(current_epoch=0, max_epochs=6, global_step=10, loggers=[logger])
    callback = ChallengeProbe(eval_data, every_n_epochs=5)
    callback.on_train_epoch_end(trainer, object())
    expected = evaluate(str(MODEL), data=str(eval_data), threads=2)
    metrics, step = records[0]
    assert step == 10
    assert metrics["probe/loo_accuracy"] == expected["score"]
    assert metrics["probe/ridge_lambda"] == expected["lambda"]
    assert metrics["probe/images"] == expected["images"]
    assert metrics["probe/classes"] == expected["classes"]
    assert not exported[0].exists()
    trainer.current_epoch = 1
    callback.on_train_epoch_end(trainer, object())
    assert len(records) == 1
    trainer.current_epoch = 4
    callback.on_train_epoch_end(trainer, object())
    trainer.current_epoch = 5
    callback.on_train_epoch_end(trainer, object())
    assert len(records) == 3


def test_probe_rejects_invalid_interval():
    with pytest.raises(ValueError, match="positive"):
        ChallengeProbe("unused.parquet", every_n_epochs=0)


def test_best_export_keeps_peak_instead_of_final(monkeypatch, tmp_path):
    scores = iter([0.6, 0.7, 0.65])
    exports = []
    def export(model, path):
        path.write_bytes(str(len(exports)).encode())
        exports.append(path)
        return path
    monkeypatch.setattr("probe.export_teacher", export)
    monkeypatch.setattr("probe.evaluate", lambda *a, **kw: {
        "score": next(scores), "images": 36, "classes": 3, "lambda": 1.0,
    })
    callback = ChallengeProbe("labels.parquet", output_dir=tmp_path)
    trainer = SimpleNamespace(current_epoch=0, max_epochs=6, global_step=10, loggers=[])
    for epoch in (0, 4, 5):
        trainer.current_epoch = epoch
        callback.on_train_epoch_end(trainer, object())
    assert (tmp_path / "best.onnx").read_bytes() == b"1"
    assert json.loads((tmp_path / "best.json").read_text())["epoch"] == 5
    assert all(not path.exists() for path in exports)
    resumed = ChallengeProbe("labels.parquet", output_dir=tmp_path)
    resumed.on_fit_start(trainer, object())
    assert resumed.best_score == 0.7


def test_support_query_metrics_are_separate_from_official_loo(monkeypatch, tmp_path):
    monkeypatch.setattr("probe.export_teacher", lambda model, path: path)
    monkeypatch.setattr("probe.diagnose", lambda *a, **kw: {
        "score": 0.5, "macro_accuracy": 0.4, "images": 6, "classes": 3, "lambda": 1,
    })
    records = []
    logger = SimpleNamespace(log_metrics=lambda metrics, step: records.append(metrics))
    trainer = SimpleNamespace(current_epoch=0, max_epochs=1, global_step=2, loggers=[logger])
    ChallengeProbe("query.parquet", support_data="support.parquet").on_train_epoch_end(trainer, object())
    assert records[0]["holdout/accuracy"] == 0.5
    assert records[0]["holdout/macro_accuracy"] == 0.4
    assert "probe/loo_accuracy" not in records[0]
