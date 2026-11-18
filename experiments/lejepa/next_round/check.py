"""Light CPU integration check for orchestration/report modules, in Slurm."""
import ast
import json
import os
from pathlib import Path
import tempfile
from .report import refresh


def main():
    if not os.environ.get('SLURM_JOB_ID'): raise RuntimeError('Slurm required')
    for path in Path(__file__).parent.glob('*.py'):
        ast.parse(path.read_text(), filename=str(path))
    with tempfile.TemporaryDirectory(dir='experiment_outputs/next-round/reporting') as tmp:
        root = Path(tmp)
        p = root/'frozen/conditions/a++b'
        p.mkdir(parents=True)
        result = dict(name='a++b',dimensions=1024,semantic=dict(classification=dict(accuracy=.75),
            fewshot={'5':{'accuracy':{'mean':.5}}}),appearance={'test':{'r2':.2}})
        (p/'results.json').write_text(json.dumps(result))
        refresh(root)
        rows = json.loads((root/'performance_matrix.json').read_text())
        ensemble = next(r for r in rows if r['model']=='a++b')
        assert ensemble['final_pet_miou'] is None
        assert ensemble['fivefold']==.75 and ensemble['dimensions']==1024
        assert json.loads((root/'candidate_shortlist.json').read_text())==[]
    print('Syntax and report integration checks passed: ensembles inherit no spatial scores; no unchecked shortlist entries')


if __name__=='__main__': main()
