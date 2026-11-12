from types import SimpleNamespace

import lightning as pl
import pytest
import torch
from stable_pretraining.methods.dino import DINO
from stable_pretraining.optim.utils import create_optimizer, is_bias_or_norm_param

from main import ChallengeDINO, NormalizedPrototypes
from submission import export_teacher


@pytest.fixture(params=["vit_tiny", "resnet18_small", "resnet18_wide"])
def model(request):
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    torch.manual_seed(17)
    model = ChallengeDINO(n_prototypes=16, normalize_prototypes=True,
                          freeze_last_layer_epochs=1, backbone=request.param)
    yield model
    torch.set_num_threads(previous)


def test_dino_student_updates_teacher_is_frozen_and_prototypes_unfreeze(model):
    model._trainer = SimpleNamespace(current_epoch=0)
    global_views = [torch.randn(2, 3, 96, 96) for _ in range(2)]
    local_views = [torch.randn(2, 3, 48, 48) for _ in range(2)]
    teacher_before = next(model.backbone.teacher.parameters()).clone()
    student_before = next(model.backbone.student.parameters()).detach().clone()
    prototype_before = model.projector.student[-1].weight.detach().clone()
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-3)
    output = DINO.forward(model, global_views=global_views, local_views=local_views)
    assert torch.isfinite(output.loss)
    assert output.student_logits.shape == (4, 2, 16)
    assert output.teacher_logits.shape == (2, 2, 16)
    output.loss.backward()
    assert next(model.backbone.student.parameters()).grad is not None
    assert all(p.grad is None for p in model.backbone.teacher.parameters())
    assert all(p.grad is None for p in model.projector.teacher.parameters())
    model.after_manual_backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    torch.testing.assert_close(model.projector.student[-1].weight, prototype_before)
    assert not torch.equal(next(model.backbone.student.parameters()), student_before)
    torch.testing.assert_close(next(model.backbone.teacher.parameters()), teacher_before)
    model.backbone.update_teacher()
    momentum = model.backbone.ema_coefficient
    torch.testing.assert_close(next(model.backbone.teacher.parameters()),
                               momentum * teacher_before + (1 - momentum) * next(model.backbone.student.parameters()))
    model.dino_loss.apply_center_update()
    expected_center = (1 - model.dino_loss.center_momentum) * output.teacher_logits.mean((0, 1), keepdim=False)
    torch.testing.assert_close(model.state_dict()["dino_loss.center"][0], expected_center)
    model._trainer.current_epoch = 1
    output = DINO.forward(model, global_views=global_views, local_views=local_views)
    output.loss.backward()
    model.after_manual_backward()
    assert model.projector.student[-1].weight.grad is not None
    optimizer.step()
    assert not torch.equal(model.projector.student[-1].weight, prototype_before)


def test_normalized_prototypes_ignore_weight_scale():
    head = NormalizedPrototypes(4, 8, bias=False)
    x = torch.randn(3, 4)
    before = head(x).detach()
    with torch.no_grad():
        head.weight.mul_(3)
    torch.testing.assert_close(head(x), before)


def test_optimizer_excludes_teacher_and_bias_norm_weight_decay(model):
    optimizer = create_optimizer(list(model.parameters()), model.optim["optimizer"],
                                 named_params=list(model.named_parameters()))
    groups = {id(parameter): group["weight_decay"]
              for group in optimizer.param_groups for parameter in group["params"]}
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            assert id(parameter) in groups
            expected = 0 if is_bias_or_norm_param(name, parameter) else model.hparams.weight_decay
            assert groups[id(parameter)] == expected
        else:
            assert id(parameter) not in groups


def test_head_and_center_checkpoint_roundtrip_including_legacy(model, tmp_path):
    def save_and_load(hparams, name):
        path = tmp_path / name
        torch.save({"state_dict": model.state_dict(), "hyper_parameters": hparams,
                    "pytorch-lightning_version": pl.__version__}, path)
        return ChallengeDINO.load_from_checkpoint(path, map_location="cpu")

    restored = save_and_load(dict(model.hparams), "new.ckpt")
    assert isinstance(restored.projector.student[-1], NormalizedPrototypes)
    assert restored.hparams.freeze_last_layer_epochs == 1
    legacy = dict(model.hparams)
    legacy.pop("normalize_prototypes")
    legacy.pop("freeze_last_layer_epochs")
    if model.hparams.backbone == "vit_tiny":
        legacy.pop("backbone")  # checkpoints predating encoder selection
    restored = save_and_load(legacy, "legacy.ckpt")
    assert type(restored.projector.student[-1]) is torch.nn.Linear
    assert restored.hparams.freeze_last_layer_epochs == 0
    assert restored.hparams.backbone == model.hparams.backbone
    torch.testing.assert_close(restored.dino_loss.center, model.dino_loss.center)


def test_monitoring_export_does_not_change_training_rng_or_encoder(model, tmp_path, monkeypatch):
    before = {name: value.clone() for name, value in model.backbone.teacher.state_dict().items()}
    rng = torch.get_rng_state().clone()

    def fake_export(encoder, path):
        torch.randn(17)
        with open(path, "wb") as file:
            file.write(b"export check")

    monkeypatch.setattr("submission.export_onnx", fake_export)
    export_teacher(model, tmp_path / "check.onnx")
    assert torch.equal(torch.get_rng_state(), rng)
    if model.hparams.backbone == "vit_tiny":
        assert model.backbone.teacher.dynamic_img_size
    for name, value in model.backbone.teacher.state_dict().items():
        torch.testing.assert_close(value, before[name], rtol=0, atol=0)


def test_resnet_batchnorm_modes_updates_and_feature_shapes(model):
    if model.hparams.backbone == "vit_tiny":
        return
    dim = 320 if model.hparams.backbone == "resnet18_wide" else 256
    student = model.backbone.student
    teacher = model.backbone.teacher
    norms = [layer for layer in student.modules() if isinstance(layer, torch.nn.BatchNorm2d)]
    teacher_norms = [layer for layer in teacher.modules() if isinstance(layer, torch.nn.BatchNorm2d)]
    assert all(layer.training for layer in norms + teacher_norms)
    assert all(layer.num_batches_tracked == 0 for layer in norms + teacher_norms)
    assert model.embed_dim == dim
    assert student(torch.randn(2, 3, 96, 96)).shape == (2, dim)
    assert student(torch.randn(2, 3, 48, 48)).shape == (2, dim)
    model.backbone.update_teacher()
    assert all(layer.num_batches_tracked == 2 for layer in norms + teacher_norms)
    assert all(torch.isfinite(layer.running_var).all() for layer in norms + teacher_norms)
    assert any(not torch.equal(layer.running_var, torch.ones_like(layer.running_var)) for layer in norms)
    teacher.eval()
    with torch.no_grad():
        features = teacher(torch.randn(1, 3, 96, 96))
    assert features.shape == (1, dim)
    assert torch.isfinite(features).all()
    assert sum(p.numel() for p in teacher.parameters()) * 4 < 20 * 1024**2
