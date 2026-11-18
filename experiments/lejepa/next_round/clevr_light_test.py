"""Synthetic row-selection/provenance fixtures; no encoder inference/dataset IO."""
import json
from pathlib import Path
import tempfile
from unittest.mock import patch
import numpy as np
from . import clevr_light as light
from .external_data import Dataset
from ..sweep import atomic_json,digest


class Base:
    def __init__(self):
        self.ids=['id-'+str(i*7%101) for i in range(90)]
        self.labels=np.random.default_rng(5).integers(0,6,(90,2))
        self.splits=dict(train=np.arange(30),val=np.arange(30,50),test=np.arange(50,90))
    def image(self,i):return ('original-index',i)


def main():
    base=Base();sizes=dict(train=10,val=5,test=20);ds=light.Subset(base,sizes)
    first=ds.selected.copy();ids=list(ds.ids)
    base.labels=base.labels[::-1].copy()
    other=light.Subset(base,sizes)
    np.testing.assert_array_equal(first,other.selected);assert ids==other.ids
    for name in ['train','val','test']:
        assert set(ds.selected[ds.splits[name]])<=set(base.splits[name])
    assert not set(ds.splits['train'])&set(ds.splits['val'])
    assert ds.image(3)==base.image(int(ds.selected[3]))
    np.testing.assert_array_equal(other.labels,base.labels[other.selected])
    reorder=[base.ids[i] for i in np.random.default_rng(8).permutation(90)]
    ix=light.rows_by_id(reorder,ds.ids)
    assert [reorder[i] for i in ix]==ds.ids
    actual=Dataset.__new__(Dataset);actual.name='clevr';actual.ids=Base().ids
    actual.labels=Base().labels;actual.splits=Base().splits;actual.source_indices=None;actual.paths=list(range(90))
    original_ids=list(actual.ids);original_labels=actual.labels.copy()
    actual.exclude([0,12,33,75])
    post=light.Subset(actual,sizes)
    class FakeImage:
        def __init__(self,path):self.path=path
        def __enter__(self):return self
        def __exit__(self,*args):pass
        def convert(self,mode):return ('original-index',self.path)
    with patch('PIL.Image.open',side_effect=FakeImage):
        for i in range(len(post)):
            original=int(actual.source_indices[post.selected[i]])
            assert post.image(i)==('original-index',original)
            assert post.ids[i]==original_ids[original]
            np.testing.assert_array_equal(post.labels[i],original_labels[original])
    assert set(post.split_sets)=={'clevr-light'}
    spec=dict(name='random-resnet9',backbone='resnet9',seed=42)
    with tempfile.TemporaryDirectory(dir='experiment_outputs/next-round') as tmp:
        parent=Path(tmp)/'full';root=Path(tmp)/'light';source=parent/'features'/spec['name']
        source.mkdir(parents=True);root.mkdir()
        atomic_json(parent/'dataset.json',dict(row_ids=reorder));atomic_json(root/'dataset.json',dict(row_ids=ds.ids))
        arrays={}
        for k in light.external_run.kinds(spec):
            arrays[k+'.npy']=np.arange(90*3,dtype=np.float32).reshape(90,3)
            np.save(source/(k+'.npy'),arrays[k+'.npy'])
        with patch.object(light,'PARENT',parent):
            assert light.slice_complete_cache(spec,ds,root) is False # Partial arrays have no trusted manifest.
            saved=dict(plan=light.expected_cache(spec,parent),files={n:digest(source/n) for n in arrays})
            atomic_json(source/'cache.json',saved)
            # Hash rejection happens before any slice publication.
            broken=dict(saved);broken['files']=dict(saved['files']);broken['files']['max2.npy']='bad'
            atomic_json(source/'cache.json',broken)
            try:light.slice_complete_cache(spec,ds,root)
            except ValueError:pass
            else:raise AssertionError('Corrupt full cache accepted')
            assert not (root/'features'/spec['name']/'cache.json').exists()
            atomic_json(source/'cache.json',saved)
            assert light.slice_complete_cache(spec,ds,root)
            for name,x in arrays.items():np.testing.assert_array_equal(np.load(root/'features'/spec['name']/name),x[ix])
    print('CLEVR-light: label-independent deterministic splits, partition roles, image/label mapping, exact row-ID cache slicing, partial-cache exclusion and hash rejection passed')


if __name__=='__main__':main()
