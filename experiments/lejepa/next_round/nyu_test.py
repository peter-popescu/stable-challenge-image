"""Synthetic validity and split-leakage fixtures; no NYU dataset accessed."""
from pathlib import Path
import tempfile
import numpy as np
from .nyu_probe import fit,metrics,validate_split_manifest


def main():
    ids=['a','b','c','d']
    manifest=dict(official_train_ids=['a','b','c'],official_test_ids=['d'],train_ids=['a','b'],
                  validation_ids=['c'],test_ids=['d'],official_split_source='synthetic fixture, not an official NYU split')
    rng=np.random.default_rng(42);x=rng.normal(size=(4,3,3,2));y=2+.1*x[...,0]+.2*x[...,1]
    mask=np.ones_like(y,dtype=bool);mask[:,0,0]=False
    y[:,0,0]=np.nan
    with tempfile.TemporaryDirectory(dir='experiment_outputs/next-round/external') as d:
        a=fit(x,y,mask,ids,manifest,Path(d)/'a')
        y2=y.copy();y2[3]*=5
        b=fit(x,y2,mask,ids,manifest,Path(d)/'b')
        assert a['alpha']==b['alpha'] and a['validation']==b['validation']
        assert a['test']['valid_pixels']==8 and a['test']['rmse_m']<.01
        assert b['test']['rmse_m']>1
        # Explicitly invalid target values cannot affect probe weights or scores.
        y3=y.copy();y3[:,0,0]=1e6
        c=fit(x,y3,mask,ids,manifest,Path(d)/'c')
        assert a['test']==c['test'] and a['validation']==c['validation']
    bad=dict(manifest,validation_ids=['d'])
    try:validate_split_manifest(bad,ids)
    except ValueError:pass
    else:raise AssertionError('Test/validation leakage accepted')
    m=metrics(np.asarray([1.,1.]),np.asarray([-1.,1.]))
    assert m['delta1']==.5 and m['nonpositive_prediction_fraction']==.5
    print('NYU prepared probe: valid masks, nonpositive outputs, official split consistency and test-label isolation fixtures passed; no dataset accessed')


if __name__=='__main__':main()
