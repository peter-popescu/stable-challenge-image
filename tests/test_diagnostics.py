import json

import numpy as np
import pytest

from conftest import MODEL
from diagnose import classification_details, diagnose, save_mistake_sheet
from evaluate import evaluate
from prepare_holdout import prepare_holdout


def test_classification_details_counts_and_original_indices():
    result = classification_details(np.array([1, 1, 1, 2]), np.array([1, 2, 1, 1]),
                                    np.array([2, 5, 8, 12]))
    assert result["score"] == 0.5
    assert result["macro_accuracy"] == pytest.approx(1 / 3)
    assert result["confusion_matrix"] == [[2, 1], [1, 0]]
    assert [row["row_index"] for row in result["mistakes"]] == [5, 12]


def test_diagnostics_preserve_official_loo(eval_data):
    actual = diagnose(MODEL, eval_data)
    expected = evaluate(str(MODEL), data=str(eval_data), threads=2)
    assert actual["score"] == expected["score"]
    assert actual["lambda"] == expected["lambda"]
    assert actual["images"] == 36  # the five unlabelled rows are never scored
    assert sum(row["correct"] for row in actual["per_class"]) == round(actual["score"] * 36)
    assert len(actual["model_sha256"]) == 64


def test_holdout_split_is_disjoint_reproducible_and_supports_queries(eval_data, tmp_path):
    first = prepare_holdout(eval_data, tmp_path / "first")
    second = prepare_holdout(eval_data, tmp_path / "second")
    assert first["train_indices"] == second["train_indices"]
    assert first["holdout_indices"] == second["holdout_indices"]
    assert not set(first["train_indices"]) & set(first["holdout_indices"])
    assert sorted(first["train_indices"] + first["holdout_indices"]) == list(range(41))
    result = diagnose(MODEL, tmp_path / "first/holdout.parquet", tmp_path / "first/train.parquet")
    assert result["protocol"] == "support_query"
    assert result["images"] == first["holdout_labelled"]
    assert 0 <= result["score"] <= 1
    assert json.loads((tmp_path / "first/split.json").read_text())["source_sha256"] == first["source_sha256"]
    with pytest.raises(FileExistsError):
        prepare_holdout(eval_data, tmp_path / "first")
    with pytest.raises(ValueError, match="must differ"):
        diagnose(MODEL, eval_data, eval_data)


def test_mistake_sheet_uses_original_rows(eval_data, tmp_path):
    path = tmp_path / "mistakes.png"
    save_mistake_sheet(eval_data, [{"row_index": 1, "true_label": 0, "predicted_label": 1}], path)
    from PIL import Image
    with Image.open(path) as image:
        assert image.size == (600, 140)
