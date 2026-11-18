"""Fresh-process CPU inference memory and latency; invoke from Slurm."""
import json,os,resource,sys,time
import numpy as np
import onnxruntime as ort
if not os.environ.get('SLURM_JOB_ID'):raise SystemExit('Slurm required')
before=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
options=ort.SessionOptions();options.intra_op_num_threads=1;options.inter_op_num_threads=1
session=ort.InferenceSession(sys.argv[1],options,providers=['CPUExecutionProvider'])
rows={}
for b in [1,4]:
    x=np.random.default_rng(42).standard_normal((b,3,96,96)).astype('float32')
    for _ in range(3):session.run(None,{'image':x})
    timings=[]
    for _ in range(20):
        start=time.perf_counter();session.run(None,{'image':x});timings.append(time.perf_counter()-start)
    rows[str(b)]=dict(median_batch_ms=1000*float(np.median(timings)),median_image_ms=1000*float(np.median(timings))/b)
print(json.dumps(dict(provider='CPUExecutionProvider',onnxruntime_version=ort.__version__,host=os.uname().nodename,threads=1,batches=rows,baseline_rss_kib=before,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)))
