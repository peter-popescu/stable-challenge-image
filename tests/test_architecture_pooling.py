"""Architecture/readout correctness; all neural computation stays in Slurm."""
import os
import pytest

pytestmark = pytest.mark.skipif(not os.environ.get('SLURM_JOB_ID'), reason='Compute requires Slurm')


def test_teammate_architecture_exact():
    import torch
    from experiments.lejepa.resnet18 import CompactResNet18
    from experiments.lejepa.references.dino_backbones import SmallResNet18
    torch.set_num_threads(1)
    torch.manual_seed(42)
    reference = SmallResNet18(width=40).eval()
    torch.manual_seed(42)
    encoder = CompactResNet18().eval()
    assert sum(p.numel() for p in encoder.parameters()) == 4365480
    assert reference.state_dict().keys() == encoder.state_dict().keys()
    for name, tensor in reference.state_dict().items():
        torch.testing.assert_close(tensor, encoder.state_dict()[name], rtol=0, atol=0)
    for size, spatial in [(96, 12), (48, 6)]:
        x = torch.randn(2, 3, size, size)
        with torch.no_grad():
            assert encoder.spatial(x).shape == (2,320,spatial,spatial)
            assert encoder(x).shape == (2,320)
            torch.testing.assert_close(encoder(x), reference(x), rtol=0, atol=0)


def test_resnet18_lejepa_backward_and_restore(tmp_path):
    import json
    from pathlib import Path
    import torch
    from experiments.lejepa.model import make_method, load_encoder
    from experiments.lejepa.export import export
    from experiments.lejepa.main import Experiment
    c = json.loads(Path('experiments/lejepa/architecture/resnet18-tuned.json').read_text())
    c['n_slices'] = 8
    model = make_method(c).train()
    assert model.projector[0].in_features == 320
    result = model(global_views=[torch.randn(2,3,96,96) for _ in range(2)],
                   local_views=[torch.randn(2,3,48,48) for _ in range(2)])
    result.loss.backward()
    assert model.backbone.stem[0].weight.grad.abs().sum() > 0
    torch.testing.assert_close(result.loss, result.inv_loss+c['lamb']*result.sigreg_loss)
    path = tmp_path/'random.ckpt'
    torch.save({'hyper_parameters':{'config':c},
                'state_dict':{'method.'+k:v for k,v in model.state_dict().items()}}, path)
    encoder = load_encoder(path)
    assert encoder.embed_dim == 320
    export(path, tmp_path/'encoder.onnx')


@pytest.mark.parametrize('kind,dim', [('max2',1024),('avg2',1024),('gap',256),('gmp',256),('avgmax',512),('avgmaxstd',768)])
def test_pooling_values_and_export(kind, dim, tmp_path):
    import numpy as np
    import torch
    from experiments.lejepa.model import Encoder
    from experiments.lejepa.readouts import SpatialReadout, FrozenReadout
    from experiments.lejepa.export import export_encoder
    torch.set_num_threads(1)
    torch.manual_seed(42)
    x = torch.randn(2,256,12,12)
    pool = SpatialReadout(kind)
    actual = pool(x)
    assert actual.shape == (2,dim)
    refs = dict(max2=torch.nn.functional.adaptive_max_pool2d(x,2).flatten(1),
                avg2=torch.nn.functional.adaptive_avg_pool2d(x,2).flatten(1),
                gap=x.mean((-1,-2)), gmp=x.flatten(2).amax(2))
    refs['avgmax'] = torch.cat([refs['gap'],refs['gmp']],1)
    refs['avgmaxstd'] = torch.cat([refs['gap'],refs['gmp'],x.std((-1,-2),unbiased=False)],1)
    torch.testing.assert_close(actual, refs[kind], rtol=0, atol=0)
    assert torch.isfinite(pool(torch.ones_like(x))).all()
    encoder = Encoder().eval()
    frozen = FrozenReadout(encoder, kind)
    before = {k:v.clone() for k,v in encoder.state_dict().items()}
    export_encoder(frozen, tmp_path/f'{kind}.onnx')
    assert all(not p.requires_grad for p in frozen.parameters())
    for key,value in encoder.state_dict().items():
        torch.testing.assert_close(value,before[key],rtol=0,atol=0)
