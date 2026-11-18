"""Compute tests must be invoked within a Slurm allocation on shared servers."""
import json
import os
from pathlib import Path
import pytest

pytestmark = pytest.mark.skipif(not os.environ.get('SLURM_JOB_ID'), reason='Compute requires Slurm')


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_encoder_and_loss(device):
    import torch
    from experiments.lejepa.model import Encoder, make_method
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA allocation required")
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)
    torch.manual_seed(42)
    x = torch.randn(2, 3, 96, 96, device=device)
    encoder = Encoder().to(device).eval()
    with torch.no_grad():
        assert encoder(x).shape == (2, 1024)
        assert encoder.spatial(x).shape == (2, 256, 12, 12)
        torch.testing.assert_close(encoder(x), encoder.conv(x).flatten(1))
    config = json.loads(Path('experiments/lejepa/config.json').read_text())
    config['n_slices'] = 8
    method = make_method(config).to(device).train()
    result = method(global_views=[x, x.flip(-1)], local_views=[x[:, :, :48, :48]])
    assert torch.isfinite(result.loss)
    torch.testing.assert_close(result.loss, result.inv_loss + config['lamb'] * result.sigreg_loss)
    result.loss.backward()
    assert method.backbone.conv[0].weight.grad.abs().sum() > 0


def test_export(tmp_path):
    import torch
    from experiments.lejepa.model import Encoder
    from experiments.lejepa.export import export
    encoder = Encoder()
    checkpoint = tmp_path / 'random.ckpt'
    torch.save({'state_dict': {'method.backbone.' + k: v for k, v in encoder.state_dict().items()}}, checkpoint)
    export(checkpoint, tmp_path / 'encoder.onnx')


@pytest.mark.parametrize("shape", [(12, 12), (6, 6), (7, 9)])
@pytest.mark.parametrize("ties", [False, True])
def test_pooling_equivalence(shape, ties):
    import torch
    from experiments.lejepa.model import DeterministicAdaptiveMaxPool2d
    if not torch.cuda.is_available():
        pytest.skip("CUDA allocation required")
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(42)
    reference = (torch.zeros(2, 3, *shape) if ties else torch.randn(2, 3, *shape))
    reference.requires_grad_()
    candidate = reference.detach().cuda().requires_grad_()
    expected = torch.nn.functional.adaptive_max_pool2d(reference, (2, 2))
    actual = DeterministicAdaptiveMaxPool2d()(candidate)
    weights = torch.randn_like(expected)
    (expected * weights).sum().backward()
    (actual * weights.cuda()).sum().backward()
    torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0)
    torch.testing.assert_close(candidate.grad.cpu(), reference.grad)


def test_full_screening_batch_fits_cuda():
    """Validate actual multiview activation memory before screening on smaller GPUs."""
    import torch
    from experiments.lejepa.model import make_method
    if not torch.cuda.is_available():
        pytest.skip('CUDA allocation required')
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(42)
    config_path = Path(os.environ.get('LEJEPA_MEMORY_CONFIG', 'experiments/lejepa/config.json'))
    config = json.loads(config_path.read_text())
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    model = make_method(config).cuda().train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
    global_views = [torch.randn(config['batch_size'], 3, 96, 96, device='cuda')
                    for _ in range(config['global_views'])]
    local_views = [torch.randn(config['batch_size'], 3, config['local_size'], config['local_size'], device='cuda')
                   for _ in range(config['local_views'])]
    result = model(global_views=global_views, local_views=local_views)
    assert torch.isfinite(result.loss)
    result.loss.backward()
    optimizer.step()
    torch.cuda.synchronize()
    assert model.backbone.conv[0].weight.grad.abs().sum() > 0
    print(f"Full batch {config['batch_size']} passed on {torch.cuda.get_device_name()}; "
          f"peak allocated {torch.cuda.max_memory_allocated()/1024**3:.2f} GiB, "
          f"reserved {torch.cuda.max_memory_reserved()/1024**3:.2f} GiB", flush=True)
    if os.environ.get('EXPERIMENT_STORAGE'):
        report = {'gpu': torch.cuda.get_device_name(), 'batch_size': config['batch_size'],
                  'config': str(config_path), 'global_views': config['global_views'],
                  'local_views': config['local_views'], 'local_size': config['local_size'],
                  'total_vram_gib': torch.cuda.get_device_properties(0).total_memory / 1024**3,
                  'peak_allocated_gib': torch.cuda.max_memory_allocated() / 1024**3,
                  'peak_reserved_gib': torch.cuda.max_memory_reserved() / 1024**3}
        (Path(os.environ['EXPERIMENT_STORAGE']) /
         f'gpu-memory-preflight-{os.environ["SLURM_JOB_ID"]}.json').write_text(json.dumps(report, indent=2))
