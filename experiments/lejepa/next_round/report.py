"""Refresh the next-round task profile without selecting on external tests.

Run in a CPU Slurm allocation. Missing endpoints remain pending. Historical
spatial metrics are inherited only by frozen readouts of the same encoder.
"""
import argparse
import csv
import hashlib
import json
import os
import shutil
from pathlib import Path

ROOT = Path('experiment_outputs/next-round')
HISTORIC = Path('experiment_outputs/green-area/downstream')


def read(path, default=None):
    return json.loads(path.read_text()) if path.exists() else default


def write_table(lines, columns, rows):
    lines.append('| ' + ' | '.join(label for _, label in columns) + ' |')
    lines.append('|' + '|'.join('---' for _ in columns) + '|')
    for row in rows:
        values = []
        for key, _ in columns:
            value = row.get(key)
            values.append('pending' if value is None else f'{value:.4f}' if isinstance(value, float) else str(value))
        lines.append('| ' + ' | '.join(values) + ' |')


def refresh(root=ROOT):
    root.mkdir(parents=True, exist_ok=True)
    launch = read(root/'training/launch.json', {})
    free_gib = shutil.disk_usage(root).free/1024**3
    external_location = str(root/'external')
    if (root/'external').is_symlink():
        external_location = os.readlink(root/'external')
    lines = ['# Next-round representation experiments', '',
        'Incremental report: pending experiments are not results. No leaderboard uploads are authorized or performed. '
        'Challenge development scores do not establish hidden performance. External test partitions are evaluation endpoints; '
        'probe hyperparameters use training and validation only.', '',
        f'Current report refresh sees {free_gib:.2f}GiB available on the output filesystem. '
        f'Preflight job: {launch.get("preflight_job", "pending")}; recorded status: {launch.get("preflight_status", "pending")}. '
        'Consult [training launch metadata](training/launch.json) and Slurm for live job state. '
        f'External dataset/cache location: `{external_location}`. '
        'Completed Pet cache arrays and temporary test checkpoints have been pruned with preserved-artifact verification; '
        'archival milestone weights retain explicit old/new artifact mappings. '
        'See [cleanup record](../../experiments/lejepa/next_round/CLEANUP.md) and '
        '[storage verification](../../experiments/lejepa/next_round/STORAGE.md).', '',
        '## Questions and current evidence', '',
        '1. **Longer training:** the historical tuned400 checkpoint differs from tuned200 in seed and LR history. '
        'The new original400 and ResNet18/400 runs start from scratch with matched control seeds and planned400 schedules. '
        'Epoch200 of those schedules is an intermediate endpoint, not a completed200 schedule.',
        '2. **View diversity:** six48px local views retain the pinned objective, batch128 and two96px global views. '
        'This changes loss composition, BatchNorm updates and compute together. See the provenance objective audit and exact-batch preflight.',
        '3. **Multiscale:** direct concatenation of stage statistics tests extraction of existing information; encoder weights are frozen.',
        '4. **Complementarity:** compare concatenation with both512-dimensional components, strongest single readouts, ResNet18, '
        'independent-seed controls and duplicate embeddings. Oracle accuracy diagnoses potential and is not an achievable ensemble score.',
        '5. **Independent transfer:** DTD, CIFAR100 and CLEVR count/distance use documented partitions and fixed preprocessing. '
        'Missing datasets or endpoints remain pending.',
        '6. **Tradeoffs:** retain native task units; no overall representation score is computed. '
        'Historical statistical training improves some dense/color metrics while reducing some semantic metrics.', '',
        '## Checkpoint provenance', '',
        'Authoritative checkpoint metadata, saved resolved configurations, epochs, hashes and artifact paths: '
        '[manifest](provenance/MANIFEST.md), [machine-readable audit](provenance/checkpoints.json). '
        'Checkpoint paths and labels are never rewritten to make comparisons appear controlled.', '',
        '## Model-by-task profile', '']
    rows = read(HISTORIC/'performance_matrix.json', [])
    frozen = root/'frozen'
    frozen_results = list((frozen/'conditions').glob('*/results.json'))
    frozen_results += list((root/'frozen-new').glob('*/conditions/*/results.json'))
    for path in sorted(frozen_results):
        result = read(path, {})
        metrics = result.get('semantic', {})
        classification = metrics.get('classification', {})
        model, _, kind = result.get('name', path.parent.name).partition('__')
        inherited = next((r for r in rows if r['model'] == model), {})
        rows.append(dict(model=model, readout=kind, dimensions=result.get('dimensions'),
            fivefold=classification.get('accuracy'), macro=classification.get('macro_accuracy'),
            loo=classification.get('official_local_loo_accuracy'),
            fewshot5=metrics.get('fewshot', {}).get('5', {}).get('accuracy', {}).get('mean'),
            retrieval_map=metrics.get('retrieval_map'), knn5=metrics.get('knn', {}).get('5'),
            ari=metrics.get('clustering_ari'), nmi=metrics.get('clustering_nmi'),
            rgb_r2=result.get('appearance', {}).get('test', {}).get('r2'),
            **{k: inherited.get(k) for k in ('mid_pet_miou','final_pet_miou','mid_local_rgb_r2','final_local_rgb_r2')}))
    # Each result retains raw predictions/episodes. Accept the suite schema and
    # keep the historical profile when newer stages have not finished yet.
    for path in sorted(frozen.rglob('semantic.json')) if frozen.exists() else []:
        result = read(path, {})
        for kind, metrics in result.items():
            if not isinstance(metrics, dict) or 'classification' not in metrics:
                continue
            classification = metrics['classification']
            model = path.parent.name
            appearance = read(path.parent/'appearance.json', {})
            inherited = next((r for r in rows if r['model'] == model), {})
            rows.append(dict(model=model, readout=kind, dimensions=metrics.get('dimensions'),
                fivefold=classification.get('accuracy'), macro=classification.get('macro_accuracy'),
                loo=classification.get('official_local_loo_accuracy'),
                fewshot5=metrics.get('fewshot', {}).get('5', {}).get('accuracy', {}).get('mean'),
                retrieval_map=metrics.get('retrieval_map'), knn5=metrics.get('knn', {}).get('5'),
                ari=metrics.get('clustering_ari'), nmi=metrics.get('clustering_nmi'),
                rgb_r2=appearance.get(kind, {}).get('test', {}).get('r2'),
                **{k: inherited.get(k) for k in ('mid_pet_miou','final_pet_miou','mid_local_rgb_r2','final_local_rgb_r2')}))
    # Preserve task identity and avoid a repeated historical readout.
    rows = list({(r['model'], r['readout']): r for r in rows}.values())
    columns = [('model','Encoder'),('readout','Readout'),('dimensions','Dim'),('fivefold','5fold'),
        ('macro','Macro'),('loo','LOO'),('fewshot5','5shot'),('retrieval_map','mAP'),('knn5','kNN5'),
        ('ari','ARI'),('nmi','NMI'),('rgb_r2','RGB R²'),('final_pet_miou','Pet mIoU')]
    write_table(lines, columns, rows)
    stage_names = {'all_means','last_two_avgmax','all_means_final_max',
        'mid_mean_final_avgmax','mid_mean_final_avgmaxstd'}
    matched = []
    for model in sorted({r['model'] for r in rows if not r['model'].startswith('random-') and '++' not in r['model']}):
        stages = [r for r in rows if r['model']==model and r['readout'] in stage_names and r.get('fivefold') is not None]
        final = [r for r in rows if r['model']==model and r['readout'] in ('avgmax','avgmaxstd') and r.get('fivefold') is not None]
        if not stages or not final: continue
        a=max(stages,key=lambda r:r['fivefold']); b=max(final,key=lambda r:r['fivefold'])
        matched.append(dict(model=model,stage=a['readout'],final=b['readout'],
            stage_accuracy=a['fivefold'],final_accuracy=b['fivefold'],delta=a['fivefold']-b['fivefold']))
    lines += ['', 'Multiscale screening against final-layer statistics: each side is selected by five-fold '
        'development accuracy within the stated family. This is readout screening on reused development '
        'data, not an independent transfer test.', '']
    write_table(lines, [('model','Encoder'),('stage','Multiscale'),('stage_accuracy','5fold'),
        ('final','Final statistics'),('final_accuracy','5fold'),('delta','Difference')], matched)
    (root/'multiscale_comparisons.json').write_text(json.dumps(matched, indent=2)+'\n')
    (root/'performance_matrix.json').write_text(json.dumps(rows, indent=2)+'\n')
    if rows:
        keys = sorted(set().union(*(r.keys() for r in rows)))
        with (root/'performance_matrix.csv').open('w') as handle:
            writer = csv.DictWriter(handle, fieldnames=keys)
            writer.writeheader(); writer.writerows(rows)
    lines += ['', '## Transformation-specific spatial results', '',
        'Correspondence uses192px with receptive-field-valid supports; global and segmentation tasks use96px. '
        'Random encoders already score highly on some transforms. These results measure transformation-specific behavior, '
        'not uniform semantic quality. Frozen readouts share the same encoder maps.', '']
    spatial = read(HISTORIC/'spatial_matrix.json', [])
    write_table(lines, [('model','Encoder'),('layer','Layer'),('transform','Transform'),
        ('mean_error','Error px'),('oracle_error','Grid oracle px'),('pck8','PCK8')],
        [r for r in spatial if r.get('descriptor') == 'cnn'])
    (root/'spatial_matrix.json').write_text(json.dumps(spatial, indent=2)+'\n')
    dense = read(HISTORIC/'dense_matrix.json', [])
    (root/'dense_matrix.json').write_text(json.dumps(dense, indent=2)+'\n')
    lines += ['', 'Full intermediate/final color accessibility and segmentation results remain in '
        '[dense_matrix.json](dense_matrix.json) and the [previous report](../green-area/downstream/REPORT.md).', '',
        '## Training curves and compute', '',
        '| Run | Planned epochs | Completed epoch | Train wall hours | Peak allocated GiB | Latest phase |',
        '|---|---:|---:|---:|---:|---|']
    resources = []
    for directory in sorted(root.glob('training/*')):
        if not directory.is_dir(): continue
        planned_config = Path(__file__).parent/(directory.name+'.json')
        if not (directory/'config.json').exists() and not planned_config.exists(): continue
        config = read(directory/'config.json', read(planned_config, {}))
        progress = read(directory/'progress.json', {})
        resource = read(directory/'resources.json', {})
        epochs = resource.get('epochs', {})
        done = max(map(int, epochs), default=0)
        hours = resource.get('training_wall_seconds', 0)/3600
        lines.append(f'| {directory.name} | {config.get("epochs", "pending")} | {done} | {hours:.3f} | '
            f'{resource.get("peak_allocated_gib", "pending")} | {progress.get("phase", "pending")} |')
        resources.append(dict(run=directory.name, config=config, progress=progress, resources=resource))
        report = directory/'REPORT.md'
        if report.exists(): lines += ['', f'[{directory.name} learning curves, LR and loss histories](training/{directory.name}/REPORT.md)', '']
    (root/'training_resources.json').write_text(json.dumps(resources, indent=2)+'\n')
    lines += ['', 'Compute-quality curves must use cumulative measured training seconds alongside epoch; '
        'allocation elapsed includes startup, extraction, milestone evaluation and idle time. '
        'Neither quantity is interchangeable with GPU FLOPs.', '',
        '## Complementarity, external tasks and export evidence', '',
        'Detailed frozen comparisons, held-out error tables and selected ensemble exports: [frozen report](frozen/REPORT.md). '
        'External validation/test matrices and dataset definitions: [external report](external/REPORT.md). '
        'Tests never choose SSL recipes or exported readouts. If validation guides future development, those datasets '
        'will be designated development data explicitly.', '',
        'Identical outer-fold predictions define the following fractions. Oracle accuracy is diagnostic only.', '']
    write_table(lines, [('a','A'),('b','B'),('agreement','Agree'),('disagreement','Disagree'),
        ('both_correct','Both correct'),('a_only_correct','A only'),('b_only_correct','B only'),
        ('both_wrong','Both wrong'),('oracle_accuracy','Oracle'),('ensemble_accuracy','Concat'),
        ('gain_over_best_component','Gain')], read(frozen/'complementarity.json', []))
    lines += ['', 'External test columns below are final evaluations, separated from validation used for probe selection.', '']
    external_rows = []
    for matrix in sorted((root/'external').glob('*/matrix.json')):
        protocol = ('CLEVR-light:1000/500/2000;seed42' if matrix.parent.name=='clevr-light'
                    else matrix.parent.name+' documented partitions')
        external_rows.extend(dict(r, protocol=protocol) for r in read(matrix, []))
    for matrix in sorted((root/'external-new').glob('*/*/matrix.json')):
        protocol = ('CLEVR-light:1000/500/2000;seed42' if 'clevr-light' in matrix.parent.parent.name
                    else matrix.parent.name+' documented partitions')
        external_rows.extend(dict(r, protocol=protocol) for r in read(matrix, []))
    lines += ['', 'CLEVR uses a fixed seed42, label-independent lightweight subset: '
        '1000 training,500 validation,2000 test images, drawn within the documented parent partitions. '
        'Counting and distance share rows and encoder passes. This is a screening protocol using VTAB '
        'task labels, not an exact VTAB-1k benchmark or the aborted full85k protocol. '
        'No CLEVR probe scores were inspected before fixing this cost-driven subset.', '']
    write_table(lines, [('representation','Representation'),('task','Task'),('protocol','Protocol'),('split','Split'),
        ('validation_accuracy','Validation'),('test_accuracy','Test'),('test_macro_accuracy','Test macro')], external_rows)
    (root/'external_matrix.json').write_text(json.dumps(external_rows, indent=2)+'\n')
    aggregate = read(root/'external/matrix.json', [])
    external_pairs = [
        ('Complementary encoders', 'jitter-200-seed42__statpool-200-seed42',
         'jitter-200-seed42--avgmax', 'statpool-200-seed42--avgmax'),
        ('ResNet18 multiscale versus GAP', 'resnet18-tuned-200-seed42--all_means',
         'resnet18-tuned-200-seed42--gap', None)]
    matched_external = []
    for task in ('dtd','cifar100'):
        lookup = {r['representation']:r for r in aggregate if r['task']==task}
        for label,an,bn,cn in external_pairs:
            if any(n not in lookup for n in (an,bn,cn) if n): continue
            a=lookup[an]; b=lookup[bn]; c=lookup[cn] if cn else None
            matched_external.append(dict(task=task,comparison=label,
                a_validation=a['validation_accuracy'],a_test=a['test_accuracy'],
                b_validation=b['validation_accuracy'],b_test=b['test_accuracy'],
                c_validation=c['validation_accuracy'] if c else None,
                c_test=c['test_accuracy'] if c else None,
                test_difference=a['test_accuracy']-max(b['test_accuracy'],c['test_accuracy'] if c else b['test_accuracy'])))
    lines += ['', 'Prespecified external comparisons: DTD values average the three fixed official splits. '
        'A is the concatenation or all-stage mean readout, B/C its components or GAP control. '
        'Test differences are descriptive endpoints; they do not select a leaderboard candidate.', '']
    write_table(lines, [('task','Task'),('comparison','Comparison'),('a_validation','A val'),('a_test','A test'),
        ('b_validation','B val'),('b_test','B test'),('c_validation','C val'),('c_test','C test'),
        ('test_difference','A minus best control test')], matched_external)
    (root/'external_matched_comparisons.json').write_text(json.dumps(matched_external, indent=2)+'\n')
    lines += ['', '## Candidate artifacts', '',
        'Candidates below are exported local experiments, not evidence of hidden gains. '
        'Only artifacts with completed official validation may be shortlisted for a later user-directed submission.', '',
        '| Path | Bytes | MiB | SHA256 |', '|---|---:|---:|---|']
    artifacts = []
    for path in sorted(root.rglob('*.onnx')):
        if 'preflight' in path.parts: continue
        h = hashlib.sha256()
        with path.open('rb') as handle:
            for block in iter(lambda: handle.read(1024*1024), b''): h.update(block)
        size = path.stat().st_size
        artifacts.append(dict(path=str(path), bytes=size, sha256=h.hexdigest(), within_size_limit=size<20*1024**2,
            validation='See adjacent frozen export audit; size alone does not establish validation'))
        lines.append(f'| `{path}` | {size} | {size/1024**2:.3f} | `{h.hexdigest()}` |')
    (root/'artifacts.json').write_text(json.dumps(artifacts, indent=2)+'\n')
    final_exports = root/'final-frozen/exports/manifest.json'
    if read(root/'final-frozen/status.json',{}).get('phase')=='complete' and final_exports.exists():
        exports = [dict(r,latency_ms_batch1=r['fresh_process_cpu']['batches']['1']['median_batch_ms'],
                          latency_ms_batch4=r['fresh_process_cpu']['batches']['4']['median_batch_ms']) for r in read(final_exports, [])]
    else:
        exports = read(frozen/'exports/manifest.json', [])
    validated = [r for r in exports if r.get('official_checks') and r.get('cached_parity') and r.get('size_bytes',20*1024**2)<20*1024**2]
    (root/'candidate_shortlist.json').write_text(json.dumps(validated, indent=2)+'\n')
    lines += ['', 'Validated local candidate shortlist (pending until all checks finish):', '']
    write_table(lines, [('name','Candidate'),('parameters','Parameters'),('size_bytes','Bytes'),
        ('latency_ms_batch1','CPU batch1 ms'),('latency_ms_batch4','CPU batch4 ms'),('sha256','SHA256')], validated)
    lines += ['', 'Fresh-process CPU peak RSS and inference-only memory increments are in '
        '[final export manifest](final-frozen/exports/manifest.json). Timing is specific to the recorded CPU and thread count. '
        'Earlier ru_maxrss observations are retained as historical attempts; final memory uses fresh-process /proc RSS/high-water counters.']
    lines += ['', '## Monitoring and launch decisions', '',
        'See [launch record](../../experiments/lejepa/next_round/LAUNCH.md) for submitted IDs, resource limits, '
        'preflight results, monitoring commands and deviations. Refresh this report after jobs finish; missing files '
        'are not interpreted as failed scientific hypotheses.']
    (root/'REPORT.md').write_text('\n'.join(lines)+'\n')


def main():
    import fcntl
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--output', type=Path, default=ROOT)
    args = parser.parse_args()
    if not os.environ.get('SLURM_JOB_ID'): parser.error('Run reporting through Slurm')
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output/'report.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        refresh(args.output)


if __name__ == '__main__': main()
