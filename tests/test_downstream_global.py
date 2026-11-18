"""Global-only caches must never persist or require full spatial tensors."""
import json,os
from pathlib import Path
import numpy as np
import pytest
pytestmark=pytest.mark.skipif(not os.environ.get('SLURM_JOB_ID'),reason='Compute in Slurm')

@pytest.mark.parametrize('backbone',['resnet9','resnet18'])
def test_global_cache_native_parity_and_no_maps(tmp_path,monkeypatch,backbone):
    import torch
    from PIL import Image
    from experiments.lejepa.downstream import common
    spec=dict(name='global-only-fixture-'+backbone,backbone=backbone,seed=42)
    ds=[dict(image=Image.fromarray(np.random.default_rng(i).integers(0,256,(19,23,3),dtype=np.uint8))) for i in range(2)]
    ids=np.array([0,1]);model=common.load(spec)
    original={k:v.clone() for k,v in model.state_dict().items()}
    with torch.inference_mode():native=model(common.normalize(common.rgb_images(ds,ids))).numpy()
    monkeypatch.setattr(common,'load',lambda *_:model)
    monkeypatch.setattr(common,'spatial',lambda *_args,**_kw:pytest.fail('Global extraction requested dense adapter'))
    common.cache_global_features(spec,ds,ids,'fixture-data',tmp_path,device='cpu',batch_size=2)
    assert {p.name for p in tmp_path.iterdir()}=={'cache.json','embeddings.npz'}
    z=np.load(tmp_path/'embeddings.npz');np.testing.assert_array_equal(z['row_ids'],ids)
    np.testing.assert_allclose(z['max2' if backbone=='resnet9' else 'gap'],native,rtol=0,atol=0)
    for k,v in model.state_dict().items():torch.testing.assert_close(v,original[k],rtol=0,atol=0)
    meta=common.validate_global_cache(spec,ids,'fixture-data',tmp_path)
    assert meta['spatial_maps_persisted'] is False
    monkeypatch.setattr(common,'load',lambda *_:pytest.fail('Verified cache should skip encoder inference'))
    common.cache_global_features(spec,ds,ids,'fixture-data',tmp_path,device='cpu')
    with pytest.raises(ValueError,match='provenance'):common.validate_global_cache(spec,ids[::-1],'fixture-data',tmp_path)
    with (tmp_path/'embeddings.npz').open('ab') as f:f.write(b'changed')
    with pytest.raises(ValueError,match='content'):common.validate_global_cache(spec,ids,'fixture-data',tmp_path)


def test_cpu_global_uses_only_pooled_cache(tmp_path,monkeypatch):
    from PIL import Image
    import torch
    from experiments.lejepa.downstream import common,run
    from experiments.lejepa import cross_validate
    from experiments.lejepa.readouts import SpatialReadout
    class Tiny(torch.nn.Module):
        def __init__(self):super().__init__();self.conv=torch.nn.Conv2d(3,3,1)
        def spatial(self,x):return torch.nn.functional.adaptive_avg_pool2d(self.conv(x),4)
    model=Tiny().eval().requires_grad_(False)
    monkeypatch.setattr(common,'load',lambda *_:model)
    monkeypatch.setattr(common,'layer_info',lambda *_:pytest.fail('Global CPU stage requested spatial layers'))
    monkeypatch.setattr(run,'digest',lambda *_:'fixture-data')
    monkeypatch.setattr(cross_validate,'fixed_plan',lambda _path,ids,labels,_hash:cross_validate.make_folds(ids,labels))
    # Nonconstant random RGB with all classes supporting every few-shot budget.
    ds=[dict(image=Image.fromarray(np.random.default_rng(i).integers(0,256,(12,15,3),dtype=np.uint8))) for i in range(60)]
    ids=np.arange(60);labels=np.repeat(np.arange(3),20)
    spec=dict(name='global-cpu-fixture',backbone='resnet9',seed=42)
    directory=tmp_path/spec['name'];cache=directory/'global-cache'
    common.cache_global_features(spec,ds,ids,'fixture-data',cache,device='cpu')
    run.run_one('cpu-global',spec,ds,ids,labels,tmp_path)
    semantic=json.loads((directory/'semantic-global.json').read_text())
    appearance=json.loads((directory/'appearance-global.json').read_text())
    assert 'avgmax' in semantic and 'classification' in semantic['avgmax']
    assert all(not k.startswith('spatial-') for k in appearance)
    assert not list(tmp_path.rglob('*.npy'))
    assert {p.name for p in cache.iterdir()}=={'cache.json','embeddings.npz'}
