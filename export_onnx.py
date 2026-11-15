"""Export an encoder to the submission format and check it with onnxruntime.

Submission contract:
    input  "image":     float32 [B, 3, 96, 96], ImageNet mean/std normalised
    output "embedding": float32 [B, 1024]
"""

import os

import numpy as np
import onnxruntime as ort
import torch
import torch.nn.functional as F

SIZE, DIM = 96, 1024
MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)


class PadTo(torch.nn.Module):
    """Zero-pad the encoder output to `dim` features (zero columns don't change a ridge probe)."""

    def __init__(self, encoder, dim=DIM):
        super().__init__()
        self.encoder, self.dim = encoder, dim

    def forward(self, x):
        z = self.encoder(x).flatten(1)
        return F.pad(z, (0, self.dim - z.shape[1]))


def export_onnx(encoder, path, pad=True):
    model = (PadTo(encoder) if pad else encoder).eval().cpu()
    x = torch.randn(2, 3, SIZE, SIZE)
    if pad:
        with torch.no_grad():
            n = model.encoder(x).flatten(1).shape[1]
        if n > DIM:
            raise ValueError(f"encoder outputs {n} features, more than the {DIM} allowed")
    torch.onnx.export(model, (x,), path, input_names=["image"], output_names=["embedding"],
                      dynamic_axes={"image": {0: "batch"}, "embedding": {0: "batch"}}, opset_version=17, dynamo=False)
    with torch.no_grad():
        ref = model(x).numpy()
    # Explicit threads avoid ORT's automatic affinity pinning outside Slurm's
    # CPU set. The comparison CLI already sets PyTorch's requested thread count.
    options = ort.SessionOptions()
    allocation = int(os.environ.get("SLURM_CPUS_PER_TASK", "2"))
    options.intra_op_num_threads = max(1, min(torch.get_num_threads(), allocation))
    options.inter_op_num_threads = 1
    out = ort.InferenceSession(path, sess_options=options,
                               providers=["CPUExecutionProvider"]).run(None, {"image": x.numpy()})[0]
    assert out.shape == (2, DIM), f"embedding shape {out.shape}, expected (B, {DIM})"
    np.testing.assert_allclose(out, ref, rtol=1e-3, atol=1e-4)
    print(f"exported {path}: output {out.shape}, matches PyTorch")
    return path
