"""Dataset-independent frozen-map 1x1 segmentation probes; no encoder optimizer."""
import json
import os
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from PIL import Image
from .common import load, spatial, normalize, identity, layer_info
from .pet_data import ROOT as PET_ROOT
from ..sweep import atomic_json,digest

OUTPUT=Path('experiment_outputs/external-eval/probes')


def trimap(mask):
    # Oxford: 1=pet,2=background,3=boundary. Keep boundary as third class.
    values=np.asarray(mask.resize((96,96),Image.Resampling.NEAREST),dtype=np.int64)
    if not np.isin(values,[1,2,3]).all(): raise ValueError('Unknown trimap labels')
    return values-1


def scores(confusion):
    cm=np.asarray(confusion,dtype=np.float64)
    intersection=np.diag(cm); union=cm.sum(0)+cm.sum(1)-intersection
    iou=np.divide(intersection,union,out=np.full(3,np.nan),where=union>0)
    return dict(miou=float(np.nanmean(iou)),foreground_iou=float(iou[0]),per_class_iou=iou.tolist(),
                pixel_accuracy=float(intersection.sum()/max(cm.sum(),1)),confusion=cm.astype('int64').tolist())


def logits(probe,x):
    return F.interpolate(probe(x),size=(96,96),mode='bilinear',align_corners=False)


def validate(probe,maps,masks,ids,device):
    confusion=np.zeros((3,3),dtype=np.int64)
    with torch.no_grad():
        for start in range(0,len(ids),64):
            ix=ids[start:start+64]
            x=torch.from_numpy(np.array(maps[ix],dtype='float32')).to(device)
            pred=logits(probe,x).argmax(1).cpu().numpy()
            confusion+=np.bincount((masks[ix].astype('int64')*3+pred).ravel(),minlength=9).reshape(3,3)
    return scores(confusion)


def train_probe(maps,masks,train_ids,val_ids,output,device='cuda',epochs=20):
    """Hyperparameters and checkpoint selection use validation only."""
    output.mkdir(parents=True,exist_ok=True)
    best_score=-np.inf; best_state=None; history=[];best_choice=None
    for lr in [.003,.01]:
        torch.manual_seed(42)
        probe=nn.Conv2d(maps.shape[1],3,1).to(device)
        optimizer=torch.optim.AdamW(probe.parameters(),lr=lr,weight_decay=1e-4)
        for epoch in range(epochs):
            order=np.random.default_rng(42+epoch).permutation(train_ids)
            probe.train();total=0.
            for start in range(0,len(order),64):
                ix=order[start:start+64]
                x=torch.from_numpy(np.array(maps[ix],dtype='float32')).to(device)
                y=torch.from_numpy(masks[ix].astype('int64')).to(device)
                optimizer.zero_grad(set_to_none=True);loss=F.cross_entropy(logits(probe,x),y)
                if not torch.isfinite(loss): raise ValueError('Nonfinite probe loss')
                loss.backward();optimizer.step();total+=loss.item()*len(ix)
            probe.eval();m=validate(probe,maps,masks,val_ids,device)
            history.append(dict(lr=lr,epoch=epoch+1,loss=total/len(order),validation=m))
            if m['miou']>best_score:
                best_score=m['miou'];best_state={k:v.detach().cpu().clone() for k,v in probe.state_dict().items()}
                best_choice=dict(lr=lr,epoch=epoch+1,validation=m)
    torch.save(dict(probe_state_dict=best_state,selection=best_choice,external_evaluation_only=True),output/'probe.pt')
    atomic_json(output/'history.json',history)
    probe=nn.Conv2d(maps.shape[1],3,1).to(device);probe.load_state_dict(best_state);probe.eval()
    return probe,best_choice


def pet_cache(model,spec,output,device):
    """Adapter contract: images/masks/split IDs. Other datasets can reuse probes."""
    from torchvision.datasets import OxfordIIITPet
    output.mkdir(parents=True,exist_ok=True)
    plan=dict(identity(spec),source=json.loads((PET_ROOT/'source.json').read_text()),
              code_sha256=digest(__file__),adapter_sha256=digest(Path(__file__).with_name('common.py')),
              input_size=96,cache_dtype='float16',layers=layer_info(spec))
    meta=output/'cache.json'
    if meta.exists():
        saved=json.loads(meta.read_text())
        if saved['plan']!=plan: raise ValueError('Pet cache identity changed')
        for f,h in saved['files'].items():
            if digest(output/f)!=h: raise ValueError('Pet cache corrupted')
        return saved
    train=OxfordIIITPet(PET_ROOT,split='trainval',target_types='segmentation')
    test=OxfordIIITPet(PET_ROOT,split='test',target_types='segmentation')
    before={k:v.cpu().clone() for k,v in model.state_dict().items()}
    samples=[(ds,i) for ds in [train,test] for i in range(len(ds))]
    n=len(samples);maps={};masks=np.lib.format.open_memmap(output/'masks.npy',mode='w+',dtype='uint8',shape=(n,96,96))
    # Fixed breed-stratified 20% validation, no access to test labels for tuning.
    val=[];rng=np.random.default_rng(42)
    for c in np.unique(train._labels):
        ids=np.flatnonzero(np.asarray(train._labels)==c);val.extend(rng.permutation(ids)[:max(1,round(.2*len(ids)))].tolist())
    splits=dict(train=sorted(set(range(len(train)))-set(val)),validation=sorted(val),test=list(range(len(train),n)))
    # Exact decoded-pixel and resized-pixel overlap exclusion against SSL inputs.
    # This detects exact images, not all possible crops or derivatives.
    from training_data import load_training_data
    challenge=load_training_data('data/train.parquet')
    import hashlib
    def hashes(im):
        im=im.convert('RGB')
        return [hashlib.sha256(np.asarray(v).tobytes()).hexdigest() for v in [im,im.resize((96,96),Image.Resampling.BILINEAR)]]
    known=set(h for im in challenge['image'] for h in hashes(im));overlap=[]
    with torch.inference_mode():
        for start in range(0,n,32):
            batch=[ds[i] for ds,i in samples[start:start+32]]
            raw=[]
            for offset,(im,mask) in enumerate(batch):
                if any(h in known for h in hashes(im)): overlap.append(start+offset)
                raw.append(np.asarray(im.convert('RGB').resize((96,96),Image.Resampling.BILINEAR),dtype=np.float32)/255)
                masks[start+offset]=trimap(mask)
            x=torch.from_numpy(np.stack(raw)).permute(0,3,1,2).to(device)
            features=spatial(model,normalize(x),spec)
            for layer,v in features.items():
                if layer not in maps:
                    maps[layer]=np.lib.format.open_memmap(output/f'{layer}.npy',mode='w+',dtype='float16',shape=(n,*v.shape[1:]))
                maps[layer][start:start+len(batch)]=v.cpu().numpy()
            if start%1024==0: print(spec['name'],'Pet cache',start,n,flush=True)
    for key in splits: splits[key]=[i for i in splits[key] if i not in set(overlap)]
    for v in maps.values(): v.flush()
    masks.flush()
    files=['mid.npy','final.npy','masks.npy']
    saved=dict(plan=plan,splits=splits,excluded_exact_overlap=overlap,files={f:digest(output/f) for f in files},
               names=[str(ds._images[i].name) for ds,i in samples],shapes={k:list(v.shape) for k,v in maps.items()})
    atomic_json(meta,saved)
    return saved


def completed_result(spec, output):
    """Reuse completed endpoints without regenerating disposable feature maps."""
    output=Path(output)
    if not (output/'results.json').is_file():return None
    metadata=json.loads((output/'cache/cache.json').read_text())
    if any(metadata['plan'].get(k)!=v for k,v in identity(spec).items()):
        raise ValueError('Completed Pet endpoint checkpoint identity differs')
    if metadata['plan']['source']!=json.loads((PET_ROOT/'source.json').read_text()):
        raise ValueError('Completed Pet endpoint dataset source differs')
    result=json.loads((output/'results.json').read_text())
    for layer in ['mid','final']:
        for name in ['results.json','probe.pt','history.json','examples.png']:
            artifact=output/layer/name
            if not artifact.is_file() or artifact.stat().st_size==0:return None
        if result.get(layer)!=json.loads((output/layer/'results.json').read_text()):
            raise ValueError('Completed Pet endpoint layer/root metrics differ')
    return result


def run(spec,device='cuda'):
    output=OUTPUT/spec['name'];output.mkdir(parents=True,exist_ok=True)
    existing=completed_result(spec,output)
    if existing is not None:return existing
    if (output/'cache/pruned.json').exists():
        raise ValueError('Pruned Pet cache has incomplete endpoint; explicit regeneration/new output required')
    model=load(spec,device);before={k:v.cpu().clone() for k,v in model.state_dict().items()}
    metadata=pet_cache(model,spec,output/'cache',device)
    for k,v in model.state_dict().items(): torch.testing.assert_close(v.cpu(),before[k],rtol=0,atol=0)
    del model
    masks=np.load(output/'cache/masks.npy',mmap_mode='r');result={}
    for layer in ['mid','final']:
        directory=output/layer;directory.mkdir(exist_ok=True)
        if (directory/'results.json').exists():
            result[layer]=json.loads((directory/'results.json').read_text());continue
        maps=np.load(output/f'cache/{layer}.npy',mmap_mode='r')
        splits=metadata['splits']
        probe,choice=train_probe(maps,masks,np.array(splits['train']),np.array(splits['validation']),directory,device)
        # Test accessed exactly once after all hyperparameter/epoch selection.
        test=validate(probe,maps,masks,np.array(splits['test']),device)
        r=dict(selection=choice,test=test,spatial_shape=list(maps.shape[1:]),
               split_sizes={k:len(v) for k,v in splits.items()},boundary='Third class: pet=0,background=1,boundary=2. No ignored pixels.',
               precision='Frozen inference FP32; feature cache FP16; probe/logits FP32',seed=42,epochs_per_candidate=20)
        atomic_json(directory/'results.json',r);result[layer]=r
        # Fixed first six official test images, no qualitative selection.
        canvas=Image.new('RGB',(96*3,96*6),'white');palette=np.array([[240,130,40],[30,30,100],[220,220,220]],dtype='uint8')
        for row,i in enumerate(splits['test'][:6]):
            with torch.no_grad(): pred=logits(probe,torch.from_numpy(np.array(maps[i:i+1],dtype='float32')).to(device)).argmax(1)[0].cpu().numpy()
            im=Image.open(PET_ROOT/'oxford-iiit-pet/images'/metadata['names'][i]).convert('RGB').resize((96,96),Image.Resampling.BILINEAR)
            canvas.paste(im,(0,row*96));canvas.paste(Image.fromarray(palette[masks[i]]),(96,row*96));canvas.paste(Image.fromarray(palette[pred]),(192,row*96))
        canvas.save(directory/'examples.png')
    atomic_json(output/'results.json',result)
    return result
