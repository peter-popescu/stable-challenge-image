"""Completed Pet endpoints must remain usable after disposable cache pruning."""
from unittest.mock import patch
import json
from pathlib import Path
from ..sweep import digest
from ..downstream.common import models
from ..downstream import segmentation


def main():
    count=0
    with patch.object(segmentation,'load',side_effect=AssertionError('Completed evaluation must not load/extract an encoder')):
        with patch.object(segmentation,'pet_cache',side_effect=AssertionError('Completed evaluation must not regenerate maps')):
            for spec in models(True):
                out=segmentation.OUTPUT/spec['name']
                if (out/'results.json').exists():
                    result=segmentation.run(spec,device='cpu')
                    assert all('test' in result[layer] for layer in ['mid','final'])
                    count+=1
    assert count==9
    plan=json.loads((segmentation.OUTPUT/'cache-prune-plan.json').read_text())
    preserved={}
    for row in plan['files']:
        assert not Path(row['path']).exists()
        preserved.update(row['preserved_artifacts_sha256'])
    for path,sha in preserved.items():assert digest(path)==sha, path
    print(f'{count} completed Pet evaluations reused; encoder load/cache regeneration forbidden; all preserved artifact hashes unchanged')


if __name__=='__main__':main()
