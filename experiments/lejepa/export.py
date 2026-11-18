"""Export just the encoder and validate dynamic batches and numerical parity."""
import argparse
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
import onnxruntime as ort
from export_onnx import export_onnx, PadTo
from .model import load_encoder


def export(checkpoint, output):
    torch.set_num_threads(1)
    encoder = load_encoder(checkpoint)
    return export_encoder(encoder, output)


def export_encoder(encoder, output):
    """Use the shared export utility, with zero padding only at export."""
    torch.set_num_threads(1)
    encoder = encoder.eval().cpu()
    original_session = ort.InferenceSession
    def bounded_session(path, sess_options=None, **kwargs):
        options = sess_options if sess_options is not None else ort.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        return original_session(path, sess_options=options, **kwargs)
    # The shared-node allocation may contain only a few of the host's CPUs.
    with patch.object(ort, 'InferenceSession', side_effect=bounded_session):
        export_onnx(encoder, str(output), pad=True)
    assert Path(output).stat().st_size < 20 * 1024**2
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    session = ort.InferenceSession(str(output), options, providers=['CPUExecutionProvider'])
    generator = torch.Generator().manual_seed(42)
    for batch in [1, 3, 4]:
        x = torch.randn(batch, 3, 96, 96, generator=generator)
        with torch.no_grad():
            expected = PadTo(encoder)(x).numpy()
        actual = session.run(None, {'image': x.numpy()})[0]
        assert actual.shape == (batch, 1024)
        np.testing.assert_allclose(actual, expected, rtol=1e-3, atol=1e-4)


if __name__ == '__main__':
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('checkpoint')
    p.add_argument('output')
    a = p.parse_args()
    export(a.checkpoint, a.output)
