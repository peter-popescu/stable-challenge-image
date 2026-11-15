"""Post-training export, official development probes and spatial diagnostics."""
import argparse
import json
import os
from pathlib import Path
import sys


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--data', required=True)
    p.add_argument('--output', required=True, type=Path)
    a = p.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):
        p.error('Run analysis inside a Slurm allocation')
    import torch
    import onnxruntime as ort
    from .export import export
    from .model import Encoder
    from export_onnx import export_onnx
    import evaluate
    import test_submission
    from . import diagnostics

    threads = int(os.environ.get('SLURM_CPUS_PER_TASK', '2'))
    torch.set_num_threads(threads)
    # The challenge utilities otherwise request all CPUs on a shared node.
    # Restrict session threading; embedding preprocessing and ridge are unchanged.
    original_session = ort.InferenceSession
    def bounded_session(path, sess_options=None, **kwargs):
        options = sess_options if sess_options is not None else ort.SessionOptions()
        options.intra_op_num_threads = threads
        options.inter_op_num_threads = 1
        return original_session(path, sess_options=options, **kwargs)
    ort.InferenceSession = bounded_session

    a.output.mkdir(parents=True, exist_ok=True)
    trained_path = a.output / 'model.onnx'
    export(a.checkpoint, trained_path)
    if test_submission.main([str(trained_path)]) != 0:
        raise RuntimeError('Official submission validation failed')
    trained = evaluate.evaluate(str(trained_path), a.data)
    (a.output / 'development_probe.json').write_text(json.dumps(trained, indent=2))
    print('TRAINED DEVELOPMENT PROBE', json.dumps(trained), flush=True)
    torch.manual_seed(42)
    untrained_path = a.output / 'untrained.onnx'
    export_onnx(Encoder().eval(), str(untrained_path), pad=False)
    untrained = evaluate.evaluate(str(untrained_path), a.data)
    (a.output / 'untrained_probe.json').write_text(json.dumps(untrained, indent=2))
    print('UNTRAINED DEVELOPMENT PROBE', json.dumps(untrained), flush=True)
    for shared in [False, True]:
        sys.argv = ['diagnostics', a.checkpoint, '--data', a.data,
                    '--output', str(a.output / ('shared' if shared else 'per_image')),
                    '--count', '64'] + (['--shared'] if shared else [])
        diagnostics.main()
    result = {'trained': trained, 'untrained': untrained,
              'accuracy_difference': trained['score'] - untrained['score'],
              'onnx_bytes': trained_path.stat().st_size,
              'note': 'Development scores on labeled training images, not held-out accuracy.'}
    (a.output / 'report.json').write_text(json.dumps(result, indent=2))
    print('REPORT', json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
