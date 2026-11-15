import sys

import lightning as pl
import pytest
import torch
from torchvision import transforms as T

from main import ChallengeDINO, DEFAULT_AUGMENTATION, DINOCrops, parse_args


def test_augmentation_settings_reach_views_without_exposing_labels(eval_data):
    dataset = DINOCrops(eval_data, local_crops=2, augmentation={
        "local_crop_min": 0.15, "color_jitter_strength": 0.5, "solarize_probability": 0,
    })
    assert dataset.images.column_names == ["image"]
    assert dataset.local_transform.transforms[0].scale == (0.15, 0.4)
    for transform in [*dataset.global_transforms, dataset.local_transform]:
        jitter = next(item for item in transform.transforms if isinstance(item, T.RandomApply)).transforms[0]
        assert jitter.brightness == (0.8, 1.2)
    solarize = [item for item in dataset.global_transforms[1].transforms if isinstance(item, T.RandomSolarize)]
    assert solarize[0].p == 0
    sample = dataset[0]
    assert set(sample) == {"global_views", "local_views"}
    assert all(view.shape == (3, 96, 96) for view in sample["global_views"])
    assert all(view.shape == (3, 48, 48) for view in sample["local_views"])


def test_explicit_defaults_reproduce_default_views(eval_data):
    first = DINOCrops(eval_data, local_crops=2)
    second = DINOCrops(eval_data, local_crops=2, augmentation=DEFAULT_AUGMENTATION)
    torch.manual_seed(42)
    a = first[0]
    torch.manual_seed(42)
    b = second[0]
    for key in a:
        for x, y in zip(a[key], b[key]):
            torch.testing.assert_close(x, y, rtol=0, atol=0)


def test_augmentation_checkpoint_roundtrip(tmp_path):
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    try:
        model = ChallengeDINO(backbone="resnet18_wide", n_prototypes=16,
                              augmentation={"local_crop_min": 0.15, "color_jitter_strength": 0.5})
        path = tmp_path / "aug.ckpt"
        torch.save({"state_dict": model.state_dict(), "hyper_parameters": dict(model.hparams),
                    "pytorch-lightning_version": pl.__version__}, path)
        restored = ChallengeDINO.load_from_checkpoint(path)
        assert dict(restored.hparams.augmentation) == dict(model.hparams.augmentation)
        assert restored.hparams.augmentation["solarize_probability"] == 0.2
    finally:
        torch.set_num_threads(previous)


@pytest.mark.parametrize("options", [["--local-crop-min", "0.5"],
    ["--local-crop-min", "nan"], ["--color-jitter-strength", "6"],
    ["--solarize-probability", "-1"], ["--probe-support-data", "unused.parquet"]])
def test_invalid_options_rejected(options, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["main.py", *options])
    with pytest.raises(SystemExit) as error:
        parse_args()
    assert error.value.code == 2
