"""Run unmodified official checks with allocation-bounded ONNX CPU threads."""
import os,sys
import onnxruntime as ort
if not os.environ.get('SLURM_JOB_ID'):raise SystemExit('Slurm required')
original=ort.InferenceSession
def bounded(path,sess_options=None,**kwargs):
    options=sess_options if sess_options is not None else ort.SessionOptions()
    options.intra_op_num_threads=1;options.inter_op_num_threads=1
    return original(path,sess_options=options,**kwargs)
ort.InferenceSession=bounded
from test_submission import main
raise SystemExit(main(sys.argv[1:]))
