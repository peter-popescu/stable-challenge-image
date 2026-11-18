"""Complete exact feature-width metadata for old candidates without weight changes."""
import json,os,subprocess,sys
from pathlib import Path
import numpy as np
import onnx
import onnxruntime as ort
from ..sweep import atomic_json,digest
from .final_export import OUT


def session(path):
    options=ort.SessionOptions();options.intra_op_num_threads=1;options.inter_op_num_threads=1
    return ort.InferenceSession(str(path),options,providers=['CPUExecutionProvider'])


def main():
    if not os.environ.get('SLURM_JOB_ID'):raise RuntimeError('Slurm required')
    directory=OUT/'exports';manifest=directory/'manifest.json';rows=json.loads(manifest.read_text())
    if len(rows)!=7:raise ValueError('Require combined five historical and two new candidates')
    atomic_json(directory/'manifest-before-metadata-compliance.json',rows)
    completed=[];corrected=0
    status=OUT/'compliance-status.json'
    atomic_json(status,dict(job_id=os.environ['SLURM_JOB_ID'],phase='checking'))
    for row in rows:
        source=Path(row['path']);source_sha=digest(source)
        if source_sha!=row['sha256']:raise ValueError('Source artifact identity changed')
        graph=onnx.load(str(source),load_external_data=False)
        if any(t.data_location==onnx.TensorProto.EXTERNAL for t in graph.graph.initializer):raise ValueError('Source graph uses external tensor data')
        outputs=graph.graph.output
        if len(outputs)!=1:raise ValueError('Require one output')
        shape=outputs[0].type.tensor_type.shape.dim
        if len(shape)!=2 or shape[0].HasField('dim_value'):raise ValueError('Require dynamic batch and rank2 output')
        changed=not shape[1].HasField('dim_value') or shape[1].dim_value!=1024
        if shape[1].HasField('dim_value') and shape[1].dim_value!=1024:raise ValueError('Incorrect fixed feature width')
        result=dict(row)
        if changed:
            target=directory/('compliant-'+source.name)
            shape[1].ClearField('dim_param');shape[1].dim_value=1024
            onnx.save(graph,str(target));onnx.checker.check_model(str(target))
            if onnx.load(str(target)).SerializeToString()!=graph.SerializeToString():raise ValueError('Saved graph changed unexpectedly')
            # Source/new graphs differ only in output type's feature-width metadata;
            # all operators, weights and dynamic batching are retained.
            original=session(source);annotated=session(target)
            for batch in (1,3,4):
                x=np.random.default_rng(42).standard_normal((batch,3,96,96)).astype('float32')
                a=original.run(None,{'image':x})[0];b=annotated.run(None,{'image':x})[0]
                if b.shape!=(batch,1024) or b.dtype!=np.float32 or not np.isfinite(b).all():raise ValueError('Invalid corrected graph output')
                np.testing.assert_allclose(a,b,rtol=1e-6,atol=1e-6)
            log=target.with_suffix('.official.log')
            with log.open('w') as f:subprocess.run([sys.executable,'-m','experiments.lejepa.next_round.frozen_validate',str(target)],stdout=f,stderr=subprocess.STDOUT,check=True)
            if target.stat().st_size>=20*1024**2:raise ValueError('Candidate exceeds strict20MiB')
            result.update(source_onnx=dict(path=str(source),sha256=source_sha),
                annotation_only_change='Output feature-width type metadata fixed to1024; all operators/weights unchanged; batch remains dynamic',
                original_to_corrected_parity_batches=[1,3,4],cached_parity_provenance='Original artifact passed cached-embedding parity; corrected graph passed source-artifact parity. No encoder/head changes.',
                path=str(target),sha256=digest(target),size_bytes=target.stat().st_size,official_checks=True,official_log=str(log),exact_output_signature=True)
            result['pre_annotation_cpu_attempt']=row['fresh_process_cpu']
            result['fresh_process_cpu']=json.loads(subprocess.check_output([sys.executable,'-m','experiments.lejepa.next_round.final_cpu_benchmark',str(target)],text=True))
            corrected+=1
        else:
            result['exact_output_signature']=True
            result['cpu_benchmark_reuse']='Artifact SHA unchanged; retaining corrected /proc benchmark from final export job.'
        final=session(result['path']);signature=final.get_outputs()[0];inputs=final.get_inputs()
        if len(inputs)!=1 or inputs[0].name!='image' or inputs[0].type!='tensor(float)' or inputs[0].shape[1:]!=[3,96,96] or isinstance(inputs[0].shape[0],int):raise ValueError('Final input signature failed')
        if signature.shape[1:]!=[1024] or isinstance(signature.shape[0],int) or signature.type!='tensor(float)':raise ValueError('Exact final signature failed')
        if digest(result['path'])!=result['sha256']:raise ValueError('Final artifact changed')
        if digest(source)!=source_sha:raise ValueError('Original source artifact changed')
        result.update(signature_checked=True,self_contained=True,source_artifact_preserved=True)
        result['latency_ms_batch1']=result['fresh_process_cpu']['batches']['1']['median_batch_ms']
        result['latency_ms_batch4']=result['fresh_process_cpu']['batches']['4']['median_batch_ms']
        completed.append(result);atomic_json(directory/'manifest-compliance-progress.json',completed)
        print('validated',result['name'],'corrected_metadata',changed,flush=True)
    atomic_json(manifest,completed)
    atomic_json(status,dict(job_id=os.environ['SLURM_JOB_ID'],phase='complete',candidates=len(completed),metadata_corrected_copies=corrected,
                           old_artifacts_unchanged=True,all_exact_dynamic_batch_feature_width1024=True))
if __name__=='__main__':
    try:main()
    except Exception as error:
        atomic_json(OUT/'compliance-status.json',dict(job_id=os.environ.get('SLURM_JOB_ID'),phase='failed',error=repr(error)))
        raise
