"""CPU inference in a new process; Linux VmHWM avoids inherited ru_maxrss.

Only NumPy and ONNX Runtime are imported; no encoder/PyTorch dependencies.
"""
import hashlib,json,os,sys,time
from pathlib import Path
import numpy as np
import onnxruntime as ort


def memory():
    fields={}
    for line in Path('/proc/self/status').read_text().splitlines():
        if line.startswith(('VmRSS:','VmHWM:')):
            key,value,*_=line.split();fields[key[:-1]]=int(value)*1024
    if set(fields)!={'VmRSS','VmHWM'}:raise RuntimeError('Linux process memory counters unavailable')
    return fields


def main():
    if not os.environ.get('SLURM_JOB_ID'):raise RuntimeError('Slurm required')
    baseline=memory();options=ort.SessionOptions();options.intra_op_num_threads=1;options.inter_op_num_threads=1
    session=ort.InferenceSession(sys.argv[1],options,providers=['CPUExecutionProvider'])
    loaded=memory();rows={}
    for batch in (1,4):
        x=np.random.default_rng(42).standard_normal((batch,3,96,96)).astype('float32')
        for _ in range(3):session.run(None,{'image':x})
        times=[]
        for _ in range(20):
            start=time.perf_counter();result=session.run(None,{'image':x})[0];times.append(time.perf_counter()-start)
        if result.shape!=(batch,1024) or result.dtype!=np.float32 or not np.isfinite(result).all():raise ValueError('Invalid benchmark output')
        rows[str(batch)]=dict(median_batch_ms=1000*float(np.median(times)),median_image_ms=1000*float(np.median(times))/batch,**memory())
    final=memory();cpu=next((line.split(':',1)[1].strip() for line in Path('/proc/cpuinfo').read_text().splitlines() if line.startswith('model name')),'unknown')
    print(json.dumps(dict(source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),provider='CPUExecutionProvider',onnxruntime_version=ort.__version__,host=os.uname().nodename,cpu=cpu,threads=1,
        batches=rows,warmup_per_batch=3,repeats_per_batch=20,input='seed42 synthetic float32 standard normal image tensors',
        baseline_rss_bytes=baseline['VmRSS'],baseline_hwm_bytes=baseline['VmHWM'],loaded_rss_bytes=loaded['VmRSS'],
        peak_rss_bytes=final['VmHWM'],peak_rss_kib=final['VmHWM']//1024,baseline_rss_kib=baseline['VmRSS']//1024,final_rss_bytes=final['VmRSS'],peak_minus_postimport_baseline_bytes=max(0,final['VmHWM']-baseline['VmRSS']),
        memory_measurement='Linux /proc/self/status VmHWM for this exec process; absolute process peak includes ORT/runtime, weights and activations; delta subtracts postimport baseline and is not isolated model allocation')))
if __name__=='__main__':main()
