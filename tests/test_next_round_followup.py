import json
from concurrent.futures import ThreadPoolExecutor
import pytest
from experiments.lejepa.next_round.followup import publish_manifest


def test_concurrent_identical_manifest_publication(tmp_path):
    spec = dict(name='encoder', backbone='resnet9', checkpoint='checkpoint.ckpt')
    with ThreadPoolExecutor(max_workers=8) as pool:
        paths = list(pool.map(lambda _: publish_manifest(tmp_path, spec), range(32)))
    assert len(set(paths)) == 1
    assert json.loads(paths[0].read_text()) == [spec]
    assert not list(tmp_path.glob('*.tmp'))
    with pytest.raises(ValueError, match='Manifest changed'):
        publish_manifest(tmp_path, dict(spec, checkpoint='changed.ckpt'))
