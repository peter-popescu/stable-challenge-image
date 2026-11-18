"""Small independent fixtures for external label boundaries and ridge leakage."""
from pathlib import Path
import tempfile
import numpy as np
from .external_data import clevr_labels,stratified_validation,Dataset
from .external_probe import probe,fit_candidates,predict


def main():
    for n in [3,10]:
        for z,expected in [(7.9,0),(8.,0),(8.1,1),(8.5,1),(8.6,2),(9.,2),(9.1,3),(9.5,3),(9.6,4),(10.,4),(10.1,5)]:
            assert clevr_labels([{'pixel_coords':[0,0,z]}]*n)==(n-3,expected)
    y=np.repeat(np.arange(3),12)
    train,val=stratified_validation(y,2)
    assert len(train)==30 and len(val)==6 and not set(train)&set(val)
    rng=np.random.default_rng(42)
    x=np.eye(3)[y]+rng.normal(0,.01,(len(y),3))
    splits=dict(train=np.r_[0:7,12:19,24:31],val=np.r_[7:9,19:21,31:33],test=np.r_[9:12,21:24,33:36])
    splits['trainval']=np.r_[splits['train'],splits['val']]
    with tempfile.TemporaryDirectory(dir='experiment_outputs/next-round/external') as d:
        a=probe(x,y,splits,Path(d)/'a.npz')
        y2=y.copy();y2[splits['test']]=(y2[splits['test']]+1)%3
        b=probe(x,y2,splits,Path(d)/'b.npz')
        assert a['validation']==b['validation']
        assert a['test']['accuracy']==1 and b['test']['accuracy']==0
    # Installed torchvision appends dtd/dtd to its root. Exercise all three
    # official partition adapters from a tiny repository-local fixture.
    from PIL import Image
    with tempfile.TemporaryDirectory(dir='experiment_outputs/next-round/external') as tmp:
        root=Path(tmp);base=root/'dtd/dtd';(base/'labels').mkdir(parents=True)
        (base/'images/striped').mkdir(parents=True)
        for i,split in enumerate(['train','val','test']):
            Image.new('RGB',(8,8),(i,0,0)).save(base/f'images/striped/{split}.png')
            for partition in [1,2,3]:
                (base/f'labels/{split}{partition}.txt').write_text(f'striped/{split}.png\n')
        ds=Dataset('dtd',root)
        assert len(ds)==3 and len(ds.split_sets)==3
        assert all(p.is_relative_to(base/'images') for p in ds.paths)
        assert ds.image(0).size==(8,8)
        assert len(ds.plan()['source_sha256'])==9
        assert all(len(v)==1 for ss in ds.split_sets.values() for k,v in ss.items() if k!='trainval')
    xx=rng.normal(size=(32,8)); yy=np.arange(32)%3; classes=np.arange(3)
    fit=fit_candidates(xx,yy,classes); scores,w=predict(fit,xx,.1)
    xn=(xx-fit[0])/fit[1]; target=(yy[:,None]==classes).astype(float)-fit[2]
    direct=xn.T@np.linalg.solve(xn@xn.T+np.eye(len(xx))*(.1*fit[-1]),target)
    np.testing.assert_allclose(w,direct,rtol=1e-10,atol=1e-10)
    print('External CLEVR boundaries, stratified splits, classification, test-label isolation, primal/dual parity, and installed torchvision DTD root fixture passed')


if __name__=='__main__':main()
