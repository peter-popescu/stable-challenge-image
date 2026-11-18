"""Refresh external endpoint tables without inventing pending measurements."""
import json
import fcntl
from pathlib import Path
import numpy as np
from ..sweep import atomic_json
from .external_run import ROOT


def report(root=ROOT):
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    with (root/'.report.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        _report(root)


def _report(root=ROOT):
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    rows=[];lines=['# External evaluation endpoints','',
        'Evaluation only; validation selects probe regularization. Test endpoints do not select encoders or readouts.',
        'Missing tasks remain pending data acquisition or execution. See EXTERNAL.md for protocol and deviations.','',
        '| Task | Representation | Splits | Validation accuracy | Test accuracy | Test macro |',
        '|---|---|---:|---:|---:|---:|']
    for dataset in ['dtd','cifar100','clevr']:
        files=list((root/dataset/'results').glob('*.json'))
        grouped={}
        for f in files:
            r=json.loads(f.read_text());grouped.setdefault((r['task'],r['representation']),[]).append(r)
        if not grouped:lines.append(f'|{dataset}|pending|—|—|—|—|')
        for (task,name),rr in grouped.items():
            row=dict(task=task,representation=name,splits=len(rr),
                     validation_accuracy=float(np.mean([r['validation']['accuracy'] for r in rr])),
                     test_accuracy=float(np.mean([r['test']['accuracy'] for r in rr])),
                     test_macro_accuracy=float(np.mean([r['test']['macro_accuracy'] for r in rr])))
            if len(rr)>1:row['test_accuracy_split_sd']=float(np.std([r['test']['accuracy'] for r in rr],ddof=1))
            rows.append(row)
            lines.append(f"|{task}|{name}|{len(rr)}|{row['validation_accuracy']:.2%}|{row['test_accuracy']:.2%}|{row['test_macro_accuracy']:.2%}|")
    lines += ['','DTD means cover fixed official splits1/2/3; split SD is descriptive, not independent training uncertainty.',
              'CIFAR full endpoint refits all50k training images after selection. Few-shot probe train counts exclude validation.','',
              '| CIFAR representation | Shots/class | Repeats | Validation mean | Test mean ± SD | Test macro mean |',
              '|---|---:|---:|---:|---:|---:|']
    few=[]
    for f in (root/'cifar100/results').glob('*.json'):
        r=json.loads(f.read_text())
        for shots in [1,5,20]:
            rr=[x for x in r.get('fewshot',[]) if x['shots']==shots]
            if not rr:continue
            row=dict(representation=r['representation'],shots=shots,repeats=len(rr),
                     validation_mean=float(np.mean([x['validation']['accuracy'] for x in rr])),
                     test_mean=float(np.mean([x['test']['accuracy'] for x in rr])),
                     test_sd=float(np.std([x['test']['accuracy'] for x in rr],ddof=1)),
                     test_macro_mean=float(np.mean([x['test']['macro_accuracy'] for x in rr])))
            few.append(row)
            lines.append(f"|{row['representation']}|{shots}|{len(rr)}|{row['validation_mean']:.2%}|{row['test_mean']:.2%} ± {row['test_sd']:.2%}|{row['test_macro_mean']:.2%}|")
    atomic_json(root/'matrix.json',rows);atomic_json(root/'fewshot.json',few)
    (root/'REPORT.md').write_text('\n'.join(lines)+'\n')


if __name__=='__main__':report()
