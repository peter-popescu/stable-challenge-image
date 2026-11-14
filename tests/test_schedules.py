from types import SimpleNamespace

import lightning as pl
import pytest
import stable_pretraining as spt
import torch

from main import ChallengeDINO, ScheduledTeacherStudentCallback, dino_schedule


def lr_trace(epochs, horizon):
    optimizer = torch.optim.SGD([torch.nn.Parameter(torch.ones(1))], lr=0.001)
    module = SimpleNamespace(
        hparams={"schedule_epochs": horizon},
        trainer=SimpleNamespace(estimated_stepping_batches=epochs * 3, max_epochs=epochs),
    )
    scheduler = dino_schedule(optimizer, module)
    trace = []
    for _ in range(epochs * 3):
        trace.append(optimizer.param_groups[0]["lr"])
        optimizer.step()
        scheduler.step()
    return trace


def test_short_run_has_exact_long_run_lr_prefix():
    short = lr_trace(300, 500)
    long = lr_trace(500, None)
    assert short == long[:len(short)]
    assert short[-1] > lr_trace(300, None)[-1]
    assert lr_trace(300, 300) == lr_trace(300, None)


class TinyTeacher(torch.nn.Module):
    def __init__(self, horizon):
        super().__init__()
        self.hparams = {"schedule_epochs": horizon}
        self.backbone = spt.TeacherStudentWrapper(torch.nn.Linear(1, 1))
        self.projector = spt.TeacherStudentWrapper(torch.nn.Linear(1, 1))


def test_short_run_ema_coefficients_and_updates_match_long_run():
    torch.manual_seed(42)
    short = TinyTeacher(500)
    long = TinyTeacher(None)
    long.load_state_dict(short.state_dict())
    callbacks = [ScheduledTeacherStudentCallback(verbose=False) for _ in range(2)]
    for callback in callbacks:
        callback._wrapper_found = True
    for epoch in range(300):
        for model, callback, duration in zip((short, long), callbacks, (300, 500)):
            with torch.no_grad():
                for wrapper in (model.backbone, model.projector):
                    next(wrapper.student.parameters()).add_(0.01)
            callback._update_all_wrappers(SimpleNamespace(current_epoch=epoch, max_epochs=duration), model)
        for name, state in short.state_dict().items():
            torch.testing.assert_close(state, long.state_dict()[name], rtol=0, atol=0)


def test_schedule_and_momentum_survive_checkpoint_reload(tmp_path):
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    try:
        model = ChallengeDINO(epochs=3, schedule_epochs=9, teacher_momentum=0.999,
                              backbone="resnet18_wide", n_prototypes=16)
        assert model.warmup_epochs_temperature_teacher == 3
        for wrapper in (model.backbone, model.projector):
            assert wrapper.base_ema_coefficient.item() == pytest.approx(0.999)
        path = tmp_path / "schedule.ckpt"
        torch.save({"state_dict": model.state_dict(), "hyper_parameters": dict(model.hparams),
                    "pytorch-lightning_version": pl.__version__}, path)
        restored = ChallengeDINO.load_from_checkpoint(path)
        assert restored.hparams.schedule_epochs == 9
        assert restored.hparams.teacher_momentum == 0.999
        for epoch in range(3):
            model._trainer = restored._trainer = SimpleNamespace(current_epoch=epoch)
            assert model._teacher_temperature() == restored._teacher_temperature()
    finally:
        torch.set_num_threads(previous)
