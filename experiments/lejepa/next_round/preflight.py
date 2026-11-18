"""Measure the exact backbone export and full SSL batch before training launch."""
import argparse
import json
import os
from pathlib import Path
import socket
import statistics
import time


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--config', type=Path, default=Path('experiments/lejepa/architecture/resnet18-tuned.json'))
    a = p.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):
        p.error('Run on a Slurm GPU allocation')
    from .storage_guard import require_free_space
    storage = require_free_space(a.output)
    import torch
    from ..model import make_encoder, make_method
    from ..export import export_encoder
    from ..sweep import atomic_json, digest, run_command
    import sys
    c = json.loads(a.config.read_text())
    a.output.mkdir(parents=True, exist_ok=True)
    report = dict(config=c, job_id=os.environ['SLURM_JOB_ID'], status='started', storage=storage,
                  hostname=socket.gethostname(),
                  artifact_paths={name:str((a.output/name).resolve()) for name in
                                  ['untrained.onnx','validator.log','preflight.json']})
    try:
        torch.set_num_threads(int(os.environ.get('SLURM_CPUS_PER_TASK','4')))
        torch.manual_seed(c['seed'])
        encoder = make_encoder(c).eval()
        report['parameters'] = sum(p.numel() for p in encoder.parameters())
        report['embedding_dim'] = encoder.embed_dim
        report['spatial_shapes'] = {}
        for size in [96,48]:
            with torch.no_grad():
                report['spatial_shapes'][str(size)] = list(encoder.spatial(torch.randn(1,3,size,size)).shape)
        model_path = a.output/'untrained.onnx'
        export_encoder(encoder,model_path)
        report.update(onnx_bytes=model_path.stat().st_size,onnx_sha256=digest(model_path))
        run_command([sys.executable,'-u','-m','experiments.lejepa.next_round.frozen_validate',model_path],a.output/'validator.log')
        del encoder
        torch.manual_seed(c['seed'])
        torch.use_deterministic_algorithms(True)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        method = make_method(c).cuda().train()
        assert method.projector[0].in_features == report['embedding_dim']
        report['projector'] = str(method.projector)
        from ..readouts import SpatialReadout
        for device in ['cpu', 'cuda']:
            for value in [0., 1., 1e-12]:
                constant = torch.full((2, 8, 6, 6), value, device=device, requires_grad=True)
                SpatialReadout('avgmaxstd')(constant).sum().backward()
                assert torch.isfinite(constant.grad).all()
        report['ssl_parameters_including_projector'] = sum(p.numel() for p in method.parameters())
        opt = torch.optim.AdamW(method.parameters(),lr=c['lr'],weight_decay=c['weight_decay'])
        gs = [torch.randn(c['batch_size'],3,c.get('global_size',96),c.get('global_size',96),device='cuda') for _ in range(c['global_views'])]
        ls = [torch.randn(c['batch_size'],3,c['local_size'],c['local_size'],device='cuda') for _ in range(c['local_views'])]
        times = []
        for step in range(8):
            torch.cuda.synchronize()
            start = time.perf_counter()
            opt.zero_grad(set_to_none=True)
            result = method(global_views=gs,local_views=ls)
            if not torch.isfinite(result.loss):
                raise ValueError('Nonfinite loss in full-batch preflight')
            result.loss.backward()
            assert next(method.backbone.parameters()).grad.abs().sum()>0
            assert all(p.grad is None or torch.isfinite(p.grad).all() for p in method.parameters())
            opt.step()
            torch.cuda.synchronize()
            elapsed = time.perf_counter()-start
            print(f'full batch step {step}: {elapsed:.3f}s',flush=True)
            if step>=3:
                times.append(elapsed)
            del result
        report.update(status='passed',gpu=torch.cuda.get_device_name(),
                      total_vram_gib=torch.cuda.get_device_properties(0).total_memory/1024**3,
                      peak_allocated_gib=torch.cuda.max_memory_allocated()/1024**3,
                      peak_reserved_gib=torch.cuda.max_memory_reserved()/1024**3,
                      median_synthetic_step_seconds=statistics.median(times),
                      measured_synthetic_steps=len(times),precision='float32',batch_size=c['batch_size'],global_views=c['global_views'],local_views=c['local_views'],sigreg_embedding_samples=c['batch_size']*(c['global_views']+c['local_views']))
    except Exception as error:
        report.update(status='failed',error=repr(error))
        raise
    finally:
        sources = [Path(__file__).parent.parent/f for f in ['main.py','model.py','data.py','augmentation_policy.py','resources.py','resnet18.py','readouts.py','export.py','next_round/train.py','next_round/preflight.py','next_round/preflight_all.py','next_round/frozen_validate.py','next_round/storage_guard.py']]
        report['source_hashes'] = {str(f):digest(f) for f in sources}
        atomic_json(a.output/'preflight.json',report)
        print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':
    main()
