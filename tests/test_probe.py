import shutil
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
