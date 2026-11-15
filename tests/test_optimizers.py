import sys

import lightning as pl
import pytest
import torch
from stable_pretraining.optim.utils import create_optimizer

from main import ChallengeDINO, parse_args


def test_sgd_momentum_state_and_recipe_survive_checkpoint(tmp_path):
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    try:
        model = ChallengeDINO(backbone="resnet18_wide", n_prototypes=16,
                              optimizer="sgd", lr=0.01, weight_decay=1e-4, sgd_momentum=0.9)
        opt = create_optimizer(list(model.parameters()), model.optim["optimizer"],
                               named_params=list(model.named_parameters()))
        assert isinstance(opt, torch.optim.SGD)
        assert opt.defaults["momentum"] == 0.9
        parameter = next(p for p in model.parameters() if p.requires_grad)
        parameter.grad = torch.ones_like(parameter)
        opt.step()
        path = tmp_path / "sgd.ckpt"
        torch.save({"state_dict": model.state_dict(), "hyper_parameters": dict(model.hparams),
                    "optimizer_states": [opt.state_dict()], "pytorch-lightning_version": pl.__version__}, path)
        restored = ChallengeDINO.load_from_checkpoint(path)
        restored_opt = create_optimizer(list(restored.parameters()), restored.optim["optimizer"],
                                        named_params=list(restored.named_parameters()))
        restored_opt.load_state_dict(torch.load(path, weights_only=False)["optimizer_states"][0])
        restored_parameter = next(p for p in restored.parameters() if p.requires_grad)
        assert restored.hparams.optimizer == "sgd"
        assert restored.hparams.lr == 0.01
        assert restored.hparams.weight_decay == 1e-4
        assert restored_opt.defaults["momentum"] == 0.9
        torch.testing.assert_close(restored_opt.state[restored_parameter]["momentum_buffer"],
                                   opt.state[parameter]["momentum_buffer"])
        assert {group["weight_decay"] for group in restored_opt.param_groups} == {0, 1e-4}
    finally:
        torch.set_num_threads(previous)


@pytest.mark.parametrize("options", [["--sgd-momentum", "0.9"],
    ["--optimizer", "sgd", "--sgd-momentum", "1"],
    ["--optimizer", "sgd", "--sgd-momentum", "nan"]])
def test_invalid_optimizer_options_fail(options, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["main.py", *options])
    with pytest.raises(SystemExit) as error:
        parse_args()
    assert error.value.code == 2
