import sys
import pickle

import lightning as pl
import numpy as np
import pytest
import torch
from PIL import Image
from torchvision import transforms as T

from main import (ChallengeDINO, DEFAULT_AUGMENTATION, DINOCrops,
                  RandomRightAngleRotation, crop_transform, parse_args)
from export_onnx import MEAN, STD


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
                              augmentation={"local_crop_min": 0.15, "color_jitter_strength": 0.5,
                                            "rotate90_probability": 0.5})
        path = tmp_path / "aug.ckpt"
        torch.save({"state_dict": model.state_dict(), "hyper_parameters": dict(model.hparams),
                    "pytorch-lightning_version": pl.__version__}, path)
        restored = ChallengeDINO.load_from_checkpoint(path)
        assert dict(restored.hparams.augmentation) == dict(model.hparams.augmentation)
        assert restored.hparams.augmentation["solarize_probability"] == 0.2
        assert restored.hparams.augmentation["rotate90_probability"] == 0.5
        legacy = dict(model.hparams)
        legacy["augmentation"] = dict(model.hparams.augmentation)
        legacy["augmentation"].pop("rotate90_probability")
        torch.save({"state_dict": model.state_dict(), "hyper_parameters": legacy,
                    "pytorch-lightning_version": pl.__version__}, path)
        restored = ChallengeDINO.load_from_checkpoint(path)
        assert restored.hparams.augmentation["rotate90_probability"] == 0
        assert restored.hparams.augmentation["color_jitter_strength"] == 0.5
    finally:
        torch.set_num_threads(previous)


@pytest.mark.parametrize("options", [["--local-crop-min", "0.5"],
    ["--local-crop-min", "nan"], ["--color-jitter-strength", "6"],
    ["--solarize-probability", "-1"], ["--probe-support-data", "unused.parquet"],
    ["--rotate90-probability", "1.1"], ["--rotate90-probability", "nan"]])
def test_invalid_options_rejected(options, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["main.py", *options])
    with pytest.raises(SystemExit) as error:
        parse_args()
    assert error.value.code == 2


@pytest.mark.parametrize("solarize", [False, True])
def test_disabled_rotation_matches_original_pipeline_and_rng(solarize):
    image = Image.fromarray(np.random.default_rng(0).integers(0, 256, (96, 96, 3), dtype=np.uint8))
    # Frozen pre-rotation reference: compatibility includes random-number consumption.
    operations = [T.RandomResizedCrop(48, scale=(0.05, 0.4), interpolation=T.InterpolationMode.BICUBIC),
                  T.RandomHorizontalFlip(),
                  T.RandomApply([T.ColorJitter(0.4, 0.4, 0.2, 0.1)], p=0.8),
                  T.RandomGrayscale(p=0.2),
                  T.RandomApply([T.GaussianBlur(9, sigma=(0.1, 2.0))], p=0.5)]
    if solarize:
        operations.append(T.RandomSolarize(128, p=0.2))
    legacy = T.Compose([*operations, T.ToTensor(), T.Normalize(MEAN, STD)])
    current = crop_transform(48, (0.05, 0.4), 0.5, solarize=solarize, rotate90_probability=0)
    torch.manual_seed(42)
    expected = legacy(image)
    expected_rng = torch.get_rng_state().clone()
    torch.manual_seed(42)
    actual = current(image)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert torch.equal(torch.get_rng_state(), expected_rng)


def test_right_angle_rotation_is_exact_and_worker_picklable():
    pixels = np.arange(36, dtype=np.uint8).reshape(6, 6)
    image = Image.fromarray(pixels)
    transform = pickle.loads(pickle.dumps(RandomRightAngleRotation(1)))
    expected = {np.rot90(pixels, k).tobytes() for k in (1, 2, 3)}
    torch.manual_seed(42)
    outputs = {np.asarray(transform(image)).tobytes() for _ in range(40)}
    assert outputs == expected


def test_rotation_applies_to_all_views_and_preserves_shapes(eval_data):
    dataset = DINOCrops(eval_data, local_crops=2, augmentation={"rotate90_probability": 0.5})
    for transform in [*dataset.global_transforms, dataset.local_transform]:
        assert sum(isinstance(item, RandomRightAngleRotation) for item in transform.transforms) == 1
    sample = dataset[0]
    assert all(view.shape == (3, 96, 96) for view in sample["global_views"])
    assert all(view.shape == (3, 48, 48) for view in sample["local_views"])
