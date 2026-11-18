"""Frozen adapters and immutable, evaluation-only feature caches.

A teammate may register a checkpoint loader as module:function in the manifest;
the loader must return an eval-capable encoder with spatial/aggregate methods.
"""
import importlib
import json
from pathlib import Path
import numpy as np
import torch
from ..sweep import atomic_json, digest

ROOT = Path('experiment_outputs/green-area')


def models(include_new=False):
    from ..pooling_study import cases
    rows = [dict(c, backbone='resnet9') for c in cases()]
    aug = Path('experiment_outputs/augmentation-study/seed42-200-diversity-retry/local-min10')
    rows += [dict(name='crop-200-seed42',checkpoint=str(aug/'checkpoints/last.ckpt'),backbone='resnet9')]
    rows += [dict(c,backbone='resnet18') for c in cases('resnet18')]
    rows += [dict(name='random-'+b,backbone=b,seed=42) for b in ['resnet9','resnet18']]
    if include_new:
        rows += [dict(name='statpool-200-seed42',backbone='resnet9',checkpoint=str(ROOT/'statpool/checkpoints/last.ckpt'))]
    return rows


def load(spec, device='cpu'):
    from ..model import load_encoder, make_encoder
    torch.manual_seed(spec.get('seed',42))
    if spec.get('loader'):
        mod, fun = spec['loader'].split(':')
        model = getattr(importlib.import_module(mod),fun)(spec)
    elif spec.get('checkpoint'):
        model = load_encoder(spec['checkpoint'])
    else:
        model = make_encoder(dict(backbone=spec['backbone'],readout='gap' if spec['backbone']=='resnet18' else 'max2'))
    return model.to(device).eval().requires_grad_(False)


def load_matching_checkpoint(spec):
    """Explicit state-key/prefix adapter for teammate VISReg or other SSL."""
    from ..model import make_encoder
    checkpoint=torch.load(spec['checkpoint'],map_location='cpu',weights_only=False)
    state=checkpoint[spec['state_key']] if spec.get('state_key') else checkpoint
    prefix=spec.get('prefix','')
    state={k[len(prefix):]:v for k,v in state.items() if k.startswith(prefix)}
    encoder=make_encoder(spec)
    encoder.load_state_dict(state,strict=True)
    return encoder


def layer_info(spec):
    # Pixel edge coordinates: first pixel center=.5. RF is maximum theoretical
    # support along residual paths, not empirical effective receptive field.
    if spec.get('layer_info'): return spec['layer_info']
    if spec.get('layer_info'):return spec['layer_info']
    if spec['backbone']=='resnet9':
        return dict(mid=dict(stride=4,center=2.,rf=20),final=dict(stride=8,center=4.,rf=64))
    return dict(mid=dict(stride=4,center=.5,rf=53),final=dict(stride=8,center=.5,rf=109))


def spatial(model, x, spec, mid_only=False):
    if spec.get('spatial_adapter'):
        mod, fun = spec['spatial_adapter'].split(':')
        return getattr(importlib.import_module(mod),fun)(model,x)
    if spec['backbone']=='resnet9':
        output={}
        for i, layer in enumerate(list(model.conv.children())[:-1]):
            x=layer(x)
            if i==11:
                output['mid']=x
                if mid_only: return output
        output['final']=x
        return output
    x=model.stem(x)
    output={}
    for i,layer in enumerate(model.stages):
        x=layer(x)
        if i==2:
            output['mid']=x
            if mid_only: return output
    output['final']=x
    return output


def normalize(x):
    from evaluate import MEAN, STD
    return (x-torch.as_tensor(MEAN,device=x.device).reshape(1,3,1,1))/torch.as_tensor(STD,device=x.device).reshape(1,3,1,1)


def rgb_images(ds, ids, size=96):
    from PIL import Image
    return torch.from_numpy(np.stack([np.asarray(ds[int(i)]['image'].convert('RGB').resize((size,size),Image.Resampling.BILINEAR),dtype=np.float32)/255 for i in ids])).permute(0,3,1,2)


def identity(spec):
    return dict(spec=spec,checkpoint_sha256=digest(spec['checkpoint']) if spec.get('checkpoint') else None)


def cache_features(spec, ds, ids, data_hash, output, device='cuda'):
    """One pass for both layers. Float32 maps and native global readouts."""
    from ..readouts import SpatialReadout
    output=Path(output); output.mkdir(parents=True,exist_ok=True)
    expected=dict(identity(spec),row_ids=np.asarray(ids).tolist(),data_sha256=data_hash,
                  preprocessing='RGB/255; challenge mean/std;96px',layers=layer_info(spec),schema=1,
                  adapter_hash=digest(__file__))
    meta=output/'cache.json'
    if meta.exists():
        saved=json.loads(meta.read_text())
        if saved['plan']!=expected: raise ValueError('Cache provenance changed')
        for file,sha in saved['files'].items():
            if digest(output/file)!=sha: raise ValueError('Cache content changed')
        return
    model=load(spec,device)
    before={k:v.cpu().clone() for k,v in model.state_dict().items()}
    arrays={}; zs={}
    prior=Path('experiment_outputs/backbone-pooling/frozen-readouts')/spec['name']/'cache'
    if spec['backbone']=='resnet18': prior=ROOT/'resnet18-pooling'/spec['name']/'cache'
    cached_final=None
    if (prior/'cache.json').exists():
        metadata=json.loads((prior/'cache.json').read_text())
        if metadata['checkpoint_sha256']!=expected['checkpoint_sha256'] or metadata['data_sha256']!=data_hash:
            raise ValueError('Existing pooling cache identity differs')
        if metadata['maps_sha256']!=digest(prior/'maps.npy'): raise ValueError('Existing pooling cache damaged')
        old=np.load(prior/'maps.npy',mmap_mode='r'); lookup={v:i for i,v in enumerate(metadata['row_ids'])}
        cached_final=old[[lookup[int(i)] for i in ids]]
    kinds=['gap','gmp','avgmax','avgmaxstd']
    if spec['backbone']=='resnet9': kinds=['max2']+kinds
    with torch.inference_mode():
        for start in range(0,len(ids),16):
            maps=spatial(model,normalize(rgb_images(ds,ids[start:start+16]).to(device)),spec,mid_only=cached_final is not None)
            if cached_final is not None: maps['final']=torch.from_numpy(cached_final[start:start+16])
            for layer,v in maps.items():
                v=v.cpu()
                if layer not in arrays:
                    arrays[layer]=np.lib.format.open_memmap(output/(layer+'.npy'),mode='w+',dtype='float32',shape=(len(ids),*v.shape[1:]))
                arrays[layer][start:start+len(v)]=v.numpy()
            for kind in kinds:
                zs.setdefault(kind,[]).append(SpatialReadout(kind)(maps['final']).cpu().numpy())
    for arr in arrays.values(): arr.flush()
    np.savez(output/'embeddings.npz',row_ids=ids,**{k:np.concatenate(v) for k,v in zs.items()})
    for k,v in model.state_dict().items(): torch.testing.assert_close(v.cpu(),before[k],rtol=0,atol=0)
    files=['mid.npy','final.npy','embeddings.npz']
    atomic_json(meta,dict(plan=expected,shapes={k:list(v.shape) for k,v in arrays.items()},files={f:digest(output/f) for f in files}))


def cache_global_features(spec, ds, ids, data_hash, output, device='cuda', batch_size=16):
    """Persist pooled vectors only; spatial activations exist only for one batch.

    This cache has its own schema and must use a separate directory from the
    opt-in dense ``cache_features`` cache. No intermediate maps, correspondence
    outputs, or local RGB probes are required by global evaluation.
    """
    from ..readouts import SpatialReadout
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    if batch_size<1:raise ValueError('batch_size must be positive')
    expected=dict(identity(spec),row_ids=np.asarray(ids).tolist(),data_sha256=data_hash,
                  preprocessing='RGB/255;bilinear resize96;challenge mean/std',
                  schema='global-pooled-v1',adapter_hash=digest(__file__))
    meta=output/'cache.json'
    if meta.exists():
        saved=json.loads(meta.read_text())
        if saved['plan']!=expected:raise ValueError('Global cache provenance changed')
        if set(saved['files'])!={'embeddings.npz'}:raise ValueError('Global cache contains unexpected artifacts')
        if digest(output/'embeddings.npz')!=saved['files']['embeddings.npz']:raise ValueError('Global cache content changed')
        return
    if (output/'mid.npy').exists() or (output/'final.npy').exists():
        raise ValueError('Use a separate directory for global pooled caches')
    model=load(spec,device)
    before={k:v.cpu().clone() for k,v in model.state_dict().items()}
    kinds=['gap','gmp','avgmax','avgmaxstd']
    if spec['backbone']=='resnet9':kinds=['max2']+kinds
    pooled={k:[] for k in kinds}
    with torch.inference_mode():
        for start in range(0,len(ids),batch_size):
            x=normalize(rgb_images(ds,ids[start:start+batch_size]).to(device))
            features=model.spatial(x)
            for kind in kinds:pooled[kind].append(SpatialReadout(kind)(features).cpu().numpy())
    for k,v in model.state_dict().items():torch.testing.assert_close(v.cpu(),before[k],rtol=0,atol=0)
    np.savez(output/'embeddings.npz',row_ids=ids,**{k:np.concatenate(v) for k,v in pooled.items()})
    atomic_json(meta,dict(plan=expected,files={'embeddings.npz':digest(output/'embeddings.npz')},
                         dimensions={k:int(v[0].shape[1]) for k,v in pooled.items()},spatial_maps_persisted=False))


def validate_global_cache(spec, ids, data_hash, output):
    """Validate pooled-cache identity/content before any CPU interpretation."""
    output=Path(output);saved=json.loads((output/'cache.json').read_text())
    expected=dict(identity(spec),row_ids=np.asarray(ids).tolist(),data_sha256=data_hash,
                  preprocessing='RGB/255;bilinear resize96;challenge mean/std',
                  schema='global-pooled-v1',adapter_hash=digest(__file__))
    if saved['plan']!=expected:raise ValueError('Global cache provenance changed')
    if set(saved['files'])!={'embeddings.npz'}:raise ValueError('Global cache contains unexpected artifacts')
    if digest(output/'embeddings.npz')!=saved['files']['embeddings.npz']:raise ValueError('Global cache content changed')
    return saved
