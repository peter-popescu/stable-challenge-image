import sys

import lightning as pl
import pytest
import torch
from torchvision import transforms as T

from main import ChallengeDINO, DINOCrops, koleo_loss, parse_args


def test_new_view_settings_and_zero_locals(eval_data):
    dataset = DINOCrops(eval_data, local_crops=0,
                        augmentation={"global_crop_min": 0.6, "blur_sigma_max": 0.8})
    assert dataset[0]["local_views"] == []
    for transform in dataset.global_transforms:
        assert transform.transforms[0].scale == (0.6, 1.0)
    for transform in [*dataset.global_transforms, dataset.local_transform]:
        blur = [op.transforms[0] for op in transform.transforms
                if isinstance(op, T.RandomApply) and isinstance(op.transforms[0], T.GaussianBlur)]
        assert blur[0].sigma == (0.1, 0.8)


def test_koleo_matches_neighbour_distances_and_has_gradients():
    features = torch.tensor([[1., 0.], [0., 1.], [-1., 0.]], requires_grad=True)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        loss = koleo_loss(features)
    assert loss.dtype == torch.float32
    torch.testing.assert_close(loss, -torch.tensor(2.).sqrt().log())
    loss.backward()
    assert torch.isfinite(features.grad).all()
    assert features.grad.abs().sum() > 0
    assert torch.isfinite(koleo_loss(torch.ones(3, 4)))
    with pytest.raises(ValueError, match="at least two"):
        koleo_loss(torch.ones(1, 4))


def test_koleo_uses_existing_first_global_student_features_and_cleans_hook():
    old = torch.get_num_threads()
    torch.set_num_threads(2)
    try:
        model = ChallengeDINO(backbone="resnet18_small", n_prototypes=16, koleo_weight=0.05)
        logged = {}
        model.log = lambda key, value, **kw: logged.update({key: value})
        calls = []
        hook = model.backbone.student.register_forward_hook(lambda m, x, y: calls.append(y))
        batch = {"global_views": [torch.randn(2, 3, 96, 96) for _ in range(2)],
                 "local_views": [torch.randn(2, 3, 48, 48)]}
        result = model(batch)
        assert len(calls) == 2  # one global pass and one local pass, no extra pass
        expected = koleo_loss(calls[0][:2])
        torch.testing.assert_close(logged["train/koleo_loss"], expected)
        torch.testing.assert_close(result["loss"], logged["train/dino_loss"] + 0.05 * expected)
        assert torch.isfinite(result["loss"])
        result["loss"].backward()
        assert model.backbone.student.stem[0].weight.grad is not None
        assert len(model.backbone.student._forward_hooks) == 1
        assert expected.requires_grad
        assert all(parameter.grad is None for parameter in model.backbone.teacher.parameters())
        hook.remove()
    finally:
        torch.set_num_threads(old)


def test_old_checkpoint_defaults_new_settings_off(tmp_path):
    old = torch.get_num_threads()
    torch.set_num_threads(2)
    try:
        model = ChallengeDINO(backbone="resnet18_small", n_prototypes=16)
        params = dict(model.hparams)
        params.pop("koleo_weight")
        params["augmentation"] = {k: v for k, v in params["augmentation"].items()
                                  if k not in ("global_crop_min", "blur_sigma_max")}
        path = tmp_path / "old.ckpt"
        torch.save({"state_dict": model.state_dict(), "hyper_parameters": params,
                    "pytorch-lightning_version": pl.__version__}, path)
        restored = ChallengeDINO.load_from_checkpoint(path)
        assert restored.hparams.koleo_weight == 0
        assert restored.hparams.augmentation["global_crop_min"] == 0.4
        assert restored.hparams.augmentation["blur_sigma_max"] == 2
    finally:
        torch.set_num_threads(old)


@pytest.mark.parametrize("explicit", [False, True])
def test_resume_restores_zero_local_crops_and_rejects_conflicts(tmp_path, monkeypatch, explicit):
    import main as training
    old = torch.get_num_threads()
    torch.set_num_threads(2)
    try:
        model = ChallengeDINO(backbone="resnet18_small", n_prototypes=16,
                              epochs=2, schedule_epochs=2, local_crops=0)
        path = tmp_path / "zero-local.ckpt"
        torch.save({"state_dict": model.state_dict(), "hyper_parameters": dict(model.hparams),
                    "pytorch-lightning_version": pl.__version__}, path)
        options = ["--local-crops", "2"] if explicit else []
        monkeypatch.setattr(sys, "argv", ["main.py", "--resume", str(path), "--epochs", "2",
                                          "--output", str(tmp_path / "resumed"), *options])
        monkeypatch.setattr(training.spt, "set", lambda **kw: None)
        def stop_at_dataset(*args, **kwargs):
            assert args[1] == 0
            raise RuntimeError("restored zero local crops")
        monkeypatch.setattr(training, "DINOCrops", stop_at_dataset)
        if explicit:
            with pytest.raises(ValueError, match="local-crops must match"):
                training.main()
        else:
            with pytest.raises(RuntimeError, match="restored zero local crops"):
                training.main()
    finally:
        torch.set_num_threads(old)


@pytest.mark.parametrize("options", [
    ["--global-crop-min", "0"], ["--global-crop-min", "nan"],
    ["--blur-sigma-max", "0.05"], ["--blur-sigma-max", "inf"],
    ["--koleo-weight", "nan"], ["--koleo-weight", "-1"],
    ["--koleo-weight", "0.01", "--batch-size", "1"],
    ["--probe-low-shot", "5"], ["--probe-repeats", "0"],
])
def test_bad_followup_flags(options, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["main.py", *options])
    with pytest.raises(SystemExit):
        parse_args()
