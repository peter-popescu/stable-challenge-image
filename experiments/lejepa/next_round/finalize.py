"""Final read-only scientific synthesis; execute in a CPU Slurm allocation."""
import csv
import argparse
import json
import os
from pathlib import Path

ROOT=Path('experiment_outputs/next-round')
PAIRS=[('Original ResNet9 duration','original-200-seed42','resnet9-original-400-seed42','max2'),
       ('ResNet18 duration','resnet18-tuned-200-seed42','resnet18-tuned-400-seed42','gap'),
       ('Six local views','tuned-200-seed42','resnet9-more-views-200-seed42','max2')]


def read(path,default=None):
    return json.loads(path.read_text()) if path.exists() else default


def table(lines,labels,rows):
    lines.extend(['| '+' | '.join(labels)+' |','|'+'|'.join(['---']*len(labels))+'|'])
    for row in rows:lines.append('| '+' | '.join(str(x) for x in row)+' |')


def pct(x):return f'{100*x:.2f}%'
def delta(x):return f'{100*x:+.2f}pp'


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--allow-pending-exports',action='store_true')
    args=parser.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):raise RuntimeError('Final synthesis runs in Slurm')
    from .report import refresh
    import fcntl
    with (ROOT/'report.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX);refresh(ROOT)
    rows=read(ROOT/'frozen/matrix.json',[])
    for file in sorted((ROOT/'frozen-new').glob('*/matrix.json')):rows+=read(file,[])
    indexed={r['name']:r for r in rows}
    external=read(ROOT/'external/final_external_matrix.json',[])
    if not external:raise ValueError('Final external audit/matrix must complete before synthesis')
    by_external={(r['representation'],r['task']):r for r in external}
    paired=[]
    for label,a,b,kind in PAIRS:
        old=indexed[a+'__'+kind];new=indexed[b+'__'+kind]
        paired.append(dict(comparison=label,control=a,model=b,readout=kind,
            **{key:new[key]-old[key] for key in ['accuracy','macro_accuracy','loo','fewshot5','retrieval','appearance_r2']}))
    lines=['# Completed LeJEPA next-round experiments','',
        'All three SSL runs and their frozen/global/external evaluation chains completed successfully. '
        'The full85k CLEVR pass was stopped for cost; the completed results use a fixed lightweight3500-image protocol. '
        'No leaderboard submission was made, and no hidden-score improvement is inferred.','',
        'The round gives task-dependent gains rather than a universal winner. Longer training raises native challenge '
        'five-fold accuracy by1.47pp for both backbones, but new ResNet18 GAP declines on DTD, CIFAR100 and CLEVR distance. '
        'Six local views do not improve the task profile broadly. Multiscale readouts and complementary encoder '
        'concatenation remain useful, with different semantic, texture and appearance tradeoffs.','',
        '## 1. Does longer training help?','',
        'Both matched-seed native readouts improve local classification. These comparisons change the planned cosine horizon: '
        'they compare completed200 and completed400 schedules, not continuation at the same LR history. '
        'Historical tuned400/seed43 versus tuned200/seed42 additionally confounds initialization.','']
    table(lines,['Comparison / native readout','200ep five-fold','400ep five-fold','Change','200ep LOO','400ep LOO'],[
        [label+' / '+kind,pct(indexed[a+'__'+kind]['accuracy']),pct(indexed[b+'__'+kind]['accuracy']),
         delta(indexed[b+'__'+kind]['accuracy']-indexed[a+'__'+kind]['accuracy']),
         pct(indexed[a+'__'+kind]['loo']),pct(indexed[b+'__'+kind]['loo'])] for label,a,b,kind in PAIRS[:2]])
    lines+=['','ResNet18/400 also improves its GAP five-shot and retrieval endpoints: '
        f"{pct(indexed['resnet18-tuned-200-seed42__gap']['fewshot5'])}→{pct(indexed['resnet18-tuned-400-seed42__gap']['fewshot5'])} and "
        f"{indexed['resnet18-tuned-200-seed42__gap']['retrieval']:.4f}→{indexed['resnet18-tuned-400-seed42__gap']['retrieval']:.4f}. "
        'External comparisons below qualify whether these gains transfer.','',
        '## 2. Do six local views help?','']
    a=indexed['tuned-200-seed42__max2'];b=indexed['resnet9-more-views-200-seed42__max2']
    lines += [f"Native five-fold changes {pct(a['accuracy'])}→{pct(b['accuracy'])}; LOO {pct(a['loo'])}→{pct(b['loo'])}. "
        f"Five-shot changes {pct(a['fewshot5'])}→{pct(b['fewshot5'])}; retrieval mAP {a['retrieval']:.4f}→{b['retrieval']:.4f}. "
        'This is not a broad local improvement. Read external endpoints separately rather than treating tiny semantic changes as a universal gain.','',
        'The pinned objective, batch128, photometric policy and crop sizes were preserved. '
        'Invariance global/local coefficient totals change1/2:1/2→1/4:3/4; SIGReg samples512→1024, '
        'including its N-scaled statistic; BatchNorm sample composition changes. '
        'Corrected backbone input-pixel proxy is3.5/2.5=7/5=1.4×, not the earlier5/3 estimate. '
        'Projector/SIGReg sample count doubles. Measured times and memory supersede workload estimates.','',
        '## 3. Do multiscale features beat final statistics?','',
        'Effects depend on checkpoint and task. For ResNet18/400, final-two-stage mean+max gives '
        f"{pct(indexed['resnet18-tuned-400-seed42__last_two_avgmax']['accuracy'])} five-fold versus "
        f"{pct(indexed['resnet18-tuned-400-seed42__avgmaxstd']['accuracy'])} final mean+max+std; both LOO values are "
        f"{pct(indexed['resnet18-tuned-400-seed42__avgmaxstd']['loo'])}. "
        'The final-statistical readout has better retrieval and five-shot performance, so the higher-CV multiscale readout is not universally preferable.','',
        'All-stage means often reduce challenge semantics but can improve texture/object transfer. '
        'Direct concatenation used no learned dimension reduction. R9 stage widths64/128/256/256 yield704,768 or1024 dimensions; '
        'R18 stage widths40/80/160/320 yield600,920 or960 dimensions. Padding occurs only at the competition interface.','',
        '## 4. Are two encoders worth the parameter budget?','']
    comps=read(ROOT/'frozen/complementarity.json',[])
    table(lines,['Pair','A512 CV','B512 CV','Concat CV','Gain vs best512','Oracle'],[
        [r['a']+' + '+r['b'],pct(r['a_accuracy']),pct(r['b_accuracy']),pct(r['ensemble_accuracy']),delta(r['gain_over_best_component']),pct(r['oracle_accuracy'])] for r in comps])
    lines+=['','Oracle accuracy is a diagnostic of complementary errors, not a realizable score. '
        'Agreement/disagreement, both-correct/one-correct/both-wrong fractions and per-class/per-fold differences are retained in '
        '[complementarity.json](frozen/complementarity.json). Pairs share encoders and are not independent replicates. '
        'The observed disagreement/gain relationship does not justify treating disagreement alone as a reliable ensemble selector. '
        'The seed43/44 same-recipe pair changes initialization, shuffle/augmentation and SIGReg randomness together; '
        'it controls independent seeded training rather than isolating initialization alone.','',
        'Duplicate tuned200 mean+max embeddings reproduce classification, few-shot and retrieval scores. '
        'The global appearance probe uses a fixed absolute ridge grid and can change under duplication through effective regularization; '
        'that difference is not encoder complementarity. Classification uses the official feature standardization '
        'and relative-ridge selection with no block weighting.','']
    candidates=['jitter-200-seed42++tuned-400-seed43','jitter-200-seed42++statpool-200-seed42',
        'tuned-400-seed43__max2','resnet18-tuned-400-seed42__last_two_avgmax','resnet18-tuned-400-seed42__avgmaxstd']
    table(lines,['Matched-budget condition','Dimensions','Five-fold','Macro','LOO','5-shot','Retrieval mAP'],[
        [n,indexed[n]['dimensions'],pct(indexed[n]['accuracy']),pct(indexed[n]['macro_accuracy']),pct(indexed[n]['loo']),pct(indexed[n]['fewshot5']),f"{indexed[n]['retrieval']:.4f}"] for n in candidates])
    lines+=['','A full ResNet18 uses4.365M encoder parameters versus4.875M for twoResNet9 encoders. '
        'The best original ensemble CV is76.40%; new ResNet18 multiscale CV is76.27% (one image apart). '
        'Final-statistical ResNet18 has stronger retrieval/few-shot than the displayed ensembles. '
        'Hidden transfer remains unknown; repeated development selection limits interpretation of small differences.','',
        '## 5. Do changes transfer externally?','',
        'DTD averages official splits1–3; CIFAR100 uses official train/test with fixed validation and full50k refit. '
        'CLEVR count/distance use the same fixed seed42 subset:1000train/500validation/2000test, with1500-row final refit. '
        'CLEVR labels follow VTAB, but this is not official VTAB1k. No CLEVR probe scores were used to pick the subset. '
        'All external hyperparameters use training/validation; test values below are fixed evaluation endpoints.','']
    ext_deltas=[]
    for label,a,b,kind in PAIRS:
        for task in ['dtd','cifar100','count','distance']:
            x=by_external.get((a+'--'+kind,task));y=by_external.get((b+'--'+kind,task))
            if x is None or y is None:raise ValueError('Missing matched external endpoint')
            ext_deltas.append(dict(comparison=label,task=task,control=a,model=b,readout=kind,
                control_validation=x['validation_accuracy'],validation=y['validation_accuracy'],
                control_test=x['test_accuracy'],test=y['test_accuracy'],test_macro=y['test_macro_accuracy'],
                test_delta=y['test_accuracy']-x['test_accuracy']))
    table(lines,['Native comparison','Task','Control val','New val','Control test','New test','Test difference'],[
        [r['comparison'],r['task'],pct(r['control_validation']),pct(r['validation']),pct(r['control_test']),pct(r['test']),delta(r['test_delta'])] for r in ext_deltas])
    lines += ['', 'Prespecified complementary-encoder comparison (512 mean+max features per component):', '']
    ext_table=[]
    for task in ['dtd','cifar100','count','distance']:
        a=by_external['jitter-200-seed42__statpool-200-seed42',task]
        b=by_external['jitter-200-seed42--avgmax',task]
        c=by_external['statpool-200-seed42--avgmax',task]
        ext_table.append([task,pct(a['validation_accuracy']),pct(a['test_accuracy']),pct(b['test_accuracy']),pct(c['test_accuracy']),
            delta(a['test_accuracy']-max(b['test_accuracy'],c['test_accuracy']))])
    table(lines,['Task','Concat val','Concat test','Jitter test','Statistical-training test','Difference vs best component'],ext_table)
    lines+=['','Full180-condition validation/test/macro matrix, DTD split variability and CIFAR few-shot results: '
        '[final external report](external/FINAL_EXTERNAL_REPORT.md), [machine-readable matrix](external/final_external_matrix.json). '
        'Native, statistical and all-stage readouts, random controls and both prespecified ensembles are included. '
        'These test results did not select the exported candidates. If used to guide future development, these datasets '
        'must explicitly be treated as development evidence rather than pristine held-out tests.','',
        '## 6. Broad gains and tradeoffs','',
        'Longer training improves matched native challenge semantics, while view diversity does not improve every task. '
        'Statistical and multiscale readouts expose different semantic, texture and appearance information; accessible color is not itself semantic utility. '
        'Complementary encoders can help independent tasks but do not dominate a similarly sized single ResNet18. '
        'Use the raw task profiles rather than an arbitrary scalar quality score.','',
        'Existing segmentation, local color accessibility and transformation-specific correspondence remain in '
        '[dense_matrix.json](dense_matrix.json), [spatial_matrix.json](spatial_matrix.json), and the '
        '[previous dense report](../green-area/downstream/REPORT.md). Frozen readouts share their encoder maps. '
        'The three new encoders have global/external endpoints; new dense probes were not run. '
        'Random encoders already perform strongly on some transforms; aggregate PCK is not a uniform learned-semantic score.','',
        '## Provenance, learning curves and resources','',
        'Final14-checkpoint provenance: [final-checkpoints.json](provenance/final-checkpoints.json). '
        'The tuned400 historical seed is43; new runs match control seed42. All planned milestones were retained and evaluated. '
        'Checkpoint archival mappings preserve old hashes; optimizer-free archives are not resume checkpoints.','',
        '[Final training audit](provenance/FINAL-TRAINING-AUDIT.md) contains actual optimizer LR/loss checks, '
        'checkpoint epochs, GPU allocations and memory. Each training directory contains epoch-based and cumulative-time curves. '
        'Training seconds, allocated GPU-hours and kernel work are separate quantities; hardware differs from some historical controls.','',
        '![New runs: quality against epochs and measured compute](final_learning_curves.png)','',
        '## Validated candidate artifacts','',
        'Candidates were selected on challenge development comparisons and scientific readout coverage, not external tests. '
        'Nothing was uploaded. The table links exact artifacts; [final export manifest](final-frozen/exports/manifest.json) '
        'retains dynamic-batch/parity/official-check evidence and fresh-process CPU timing/memory scopes.','']
    compliant=(read(ROOT/'final-frozen/status.json',{}).get('phase')=='complete'
               and read(ROOT/'final-frozen/compliance-status.json',{}).get('phase')=='complete')
    if not compliant and not args.allow_pending_exports:
        raise ValueError('Final ONNX signature/resource compliance audit has not completed')
    exports=read(ROOT/'final-frozen/exports/manifest.json',[]) if compliant else []
    if compliant and len(exports)!=7:raise ValueError('Expected five previous and two new validated artifacts')
    if not compliant:
        lines += ['**Artifact packaging still in progress:** scientific results above are complete and audited. '
            'The final signature/resource check206616 is running or queued. This candidate table will be populated '
            'automatically after it succeeds; no submission readiness is claimed for pending artifacts.','']
    for r in exports:
        if (not r.get('official_checks') or not r.get('cached_parity') or not r.get('signature_checked')
                or not r.get('self_contained') or r['size_bytes']>=20*1024**2):
            raise ValueError('Unvalidated candidate in final manifest')
    table(lines,['Candidate','Parameters','MiB','CPU batch1 ms','CPU batch4 ms','Peak process MiB','SHA256'],[
        [f"[{r['name']}]({os.path.relpath(r['path'],ROOT)})",r['parameters'],f"{r['size_bytes']/1024**2:.3f}",
         f"{r['fresh_process_cpu']['batches']['1']['median_batch_ms']:.2f}",
         f"{r['fresh_process_cpu']['batches']['4']['median_batch_ms']:.2f}",
         f"{r['fresh_process_cpu']['peak_rss_bytes']/1024**2:.1f}",f"`{r['sha256']}`"] for r in exports])
    lines += ['', 'CPU timings use one ORT thread, three warmups and20 timed repeats for batches1/4. '
        'Peak memory is absolute process VmHWM after those batch sizes, including runtime, weights and activations. '
        'The manifest records host/CPU/version and baseline/delta; it is not an isolated weight allocation or a batch256 memory estimate.']
    accounting=read(ROOT/'final-round-accounting.json',{})
    lines += ['', 'All task-owned allocations, including interrupted full CLEVR, failed/cancelled attempts and storage recovery: '
        '[final-round-accounting.json](final-round-accounting.json). This lists parent allocations only and excludes the unrelated interactive allocation. '
        'CPU allocation-hours include CPUs reserved alongside GPU jobs and are distinct from measured CPU utilization.']
    lines+=['','NYU remains a prepared masked linear-probe implementation with synthetic checks; no real depth dataset/run was launched. '
        'The stopped full CLEVR attempt and recovered preflight/migration attempts remain in launch/accounting history. '
        'Storage and cleanup preserved metrics, probes, visualizations and final weights.','',
        'Complete challenge profiles: [REPORT.md](REPORT.md), [performance_matrix.csv](performance_matrix.csv). '
        'Exact protocol, operational deviations and monitoring history: '
        '[LAUNCH.md](../../experiments/lejepa/next_round/LAUNCH.md).']
    (ROOT/'final_paired_changes.json').write_text(json.dumps(dict(challenge=paired,external=ext_deltas),indent=2)+'\n')
    with (ROOT/'final_external_paired_changes.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(ext_deltas[0]));w.writeheader();w.writerows(ext_deltas)
    (ROOT/'FINAL_REPORT.md').write_text('\n'.join(lines)+'\n')
    plot()
    print(json.dumps(dict(report=str(ROOT/'FINAL_REPORT.md'),challenge_conditions=len(rows),external_conditions=len(external),validated_exports=len(exports))))


def plot():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for name in ['resnet18-tuned-400','resnet9-original-400','resnet9-more-views-200']:
        rows=read(ROOT/'training'/name/'comparison.json')['milestones']
        for ax,key,label in [(axes[0],'epoch','Completed epochs'),(axes[1],'cumulative_training_seconds','Measured training hours')]:
            x=[r[key]/3600 if key.endswith('seconds') else r[key] for r in rows]
            ax.plot(x,[100*r['accuracy'] for r in rows],marker='o',label=name)
            ax.set(xlabel=label,ylabel='Native readout five-fold accuracy (%)')
    axes[0].legend(fontsize=7);fig.suptitle('New runs on RTX A6000; epoch seconds exclude milestone evaluation')
    fig.tight_layout();fig.savefig(ROOT/'final_learning_curves.png',dpi=160);fig.savefig(ROOT/'final_learning_curves.svg');plt.close(fig)


if __name__=='__main__':main()
