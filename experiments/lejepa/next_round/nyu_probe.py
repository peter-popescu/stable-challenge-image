"""Prepared frozen linear metric-depth probe; no dataset acquisition or auto-launch.

Inputs are prealigned frozen feature maps [N,H,W,C], metric depths [N,H,W],
and explicit boolean validity masks. An external adapter must supply official
train/test image IDs and a training-only validation subset. Never interpolate
invalid depth values into valid locations while preparing these arrays.
"""
import json
from pathlib import Path
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from ..sweep import atomic_json


def validate_split_manifest(manifest, image_ids):
    ids=list(map(str,image_ids))
    if len(ids)!=len(set(ids)): raise ValueError('Duplicate input image IDs')
    required=['official_train_ids','official_test_ids','train_ids','validation_ids','test_ids','official_split_source']
    if any(k not in manifest for k in required): raise ValueError('Require documented official split and all partition IDs')
    if not manifest['official_split_source']: raise ValueError('Official split source must be documented')
    sets={}
    for key in required[:-1]:
        vals=list(map(str,manifest[key]))
        if len(vals)!=len(set(vals)):raise ValueError(f'Duplicate IDs in {key}')
        sets[key]=set(vals)
    official_train,official_test=sets['official_train_ids'],sets['official_test_ids']
    train,val,test=sets['train_ids'],sets['validation_ids'],sets['test_ids']
    if official_train & official_test or train & val or train & test or val & test:
        raise ValueError('Official/image probe splits must be disjoint')
    if not train or not val or not test: raise ValueError('All probe partitions must be nonempty')
    if train|val != official_train or test != official_test:
        raise ValueError('Train/validation must partition official training; test must equal official test')
    if set(ids) != official_train|official_test: raise ValueError('Input IDs must match official split IDs exactly')
    lookup={v:i for i,v in enumerate(ids)}
    return [np.asarray([lookup[str(v)] for v in manifest[key]],dtype=np.int64) for key in ['train_ids','validation_ids','test_ids']]


def metrics(target,prediction):
    target=np.asarray(target,dtype=np.float64);prediction=np.asarray(prediction,dtype=np.float64)
    if not len(target) or np.any(target<=0) or not np.isfinite(target).all() or not np.isfinite(prediction).all():
        raise ValueError('Require finite positive targets and finite predictions on valid sites')
    # Linear heads can predict nonpositive depth. Do not silently clamp outputs:
    # these predictions fail delta1 and remain visible in regression errors.
    ratio=np.full(target.shape,np.inf,dtype=np.float64)
    positive=prediction>0
    ratio[positive]=np.maximum(target[positive]/prediction[positive],prediction[positive]/target[positive])
    return dict(rmse_m=float(np.sqrt(np.mean((prediction-target)**2))),
                absolute_relative_error=float(np.mean(np.abs(prediction-target)/target)),
                delta1=float(np.mean(ratio<1.25)),valid_pixels=int(len(target)),
                nonpositive_prediction_fraction=float(np.mean(~positive)))


def fit(features,depth_m,valid_masks,image_ids,manifest,output,alphas=(.01,1.,100.)):
    """Select regularization using validation only, then score untouched test.

    No refit on validation: validation remains a distinct partition. Every valid
    site from training images contributes equally. Test metrics are pixel-weighted
    over valid sites, with per-image metrics retained separately.
    """
    x=np.asarray(features);y=np.asarray(depth_m);mask=np.asarray(valid_masks)
    if x.ndim!=4 or y.shape!=x.shape[:3] or mask.shape!=y.shape or mask.dtype!=np.bool_:
        raise ValueError('Require features NHWC, depth NHW and boolean masks NHW')
    if len(image_ids)!=len(x): raise ValueError('Image IDs must align arrays')
    if not np.isfinite(x).all(): raise ValueError('Nonfinite frozen feature maps')
    if not alphas or any(a<=0 or not np.isfinite(a) for a in alphas): raise ValueError('Use finite positive ridge alphas')
    splits=validate_split_manifest(manifest,image_ids)
    valid=mask & np.isfinite(y) & (y>0)
    xx=[];yy=[]
    for ids in splits:
        keep=valid[ids]
        if not keep.any():raise ValueError('Partition has no valid depth sites')
        xx.append(x[ids][keep].astype(np.float64));yy.append(y[ids][keep].astype(np.float64))
    scale=StandardScaler().fit(xx[0]);z=[scale.transform(v) for v in xx]
    best=None;candidates=[]
    for alpha in alphas:
        model=Ridge(alpha=float(alpha)).fit(z[0],yy[0])
        m=metrics(yy[1],model.predict(z[1]));candidates.append(dict(alpha=float(alpha),**m))
        if best is None or m['rmse_m']<best[0]:best=(m['rmse_m'],float(alpha),model,m)
    model=best[2];test_prediction=model.predict(z[2]);test=metrics(yy[2],test_prediction)
    per_image=[]
    for i in splits[2]:
        keep=valid[i]
        if not keep.any():
            per_image.append(dict(image_id=str(image_ids[i]),valid_pixels=0));continue
        pred=model.predict(scale.transform(x[i][keep].astype(np.float64)))
        per_image.append(dict(image_id=str(image_ids[i]),**metrics(y[i][keep],pred)))
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    np.savez(output/'linear-depth-probe.npz',mean=scale.mean_,scale=scale.scale_,weights=model.coef_,intercept=model.intercept_,
             test_predictions=test_prediction,test_targets=yy[2])
    result=dict(alpha=best[1],validation=best[3],test=test,candidates=candidates,per_image=per_image,
                split_manifest=manifest,training_images=len(splits[0]),validation_images=len(splits[1]),test_images=len(splits[2]),
                valid_sites=[len(v) for v in yy],mask_policy='explicit valid AND finite positive metric target; invalid depth never scored',
                selection='Training-only standardization and fitting; validation RMSE selects alpha; test untouched until selected fit',
                target_units='metres',head='Linear ridge with intercept; no nonlinear decoder or output clamp')
    atomic_json(output/'results.json',result)
    return result
