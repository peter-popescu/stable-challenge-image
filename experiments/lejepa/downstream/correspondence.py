"""Synthetic matching with analytic coordinates and full RF-valid overlap.

192px diagnostic inputs allow the R18 final 109px RF to fit without padding.
No resizing feature maps to a common grid: report native grid quantization.
"""
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
from .common import spatial, normalize, layer_info

SIZE=192


def transforms():
    angle=np.deg2rad(15); c,s=np.cos(angle),np.sin(angle)
    rot=np.array([[c,-s],[s,c]])
    r=np.eye(3);r[:2,:2]=rot;r[:2,2]=SIZE/2-rot@np.full(2,SIZE/2)
    return dict(identity=np.eye(3),translation=np.array([[1,0,16],[0,1,8],[0,0,1.]]),rotation=r,
                flip=np.array([[-1,0,SIZE],[0,1,0],[0,0,1.]]),
                crop_resize=np.array([[1.25,0,-24],[0,1.25,-24],[0,0,1.]]),photometric=np.eye(3),
                translation_photo=np.array([[1,0,16],[0,1,8],[0,0,1.]]))


def apply(points,matrix):
    return np.c_[points,np.ones(len(points))]@matrix[:2].T


def warp(rgb,matrix):
    y,x=torch.meshgrid(torch.arange(SIZE)+.5,torch.arange(SIZE)+.5,indexing='ij')
    xy=np.stack([x.numpy().ravel(),y.numpy().ravel()],1)
    source=apply(xy,np.linalg.inv(matrix))
    grid=torch.tensor(source.reshape(SIZE,SIZE,2)*2/SIZE-1,dtype=rgb.dtype,device=rgb.device)[None]
    return F.grid_sample(rgb,grid,mode='bilinear',padding_mode='zeros',align_corners=False)


def centers(shape,info):
    y,x=np.meshgrid(np.arange(shape[0])*info['stride']+info['center'],np.arange(shape[1])*info['stride']+info['center'],indexing='ij')
    return np.stack([x.ravel(),y.ravel()],1)


def valid(points, radius, matrix=None):
    """All square RF corners must remain inside image before/after mapping."""
    corners=(points[:,None,:]+np.array([[-radius,-radius],[-radius,radius],[radius,-radius],[radius,radius]]))
    if matrix is not None: corners=apply(corners.reshape(-1,2),matrix).reshape(-1,4,2)
    return ((corners>=0)&(corners<=SIZE)).all((1,2))


def match(a,b,info,matrix):
    pa=centers(a.shape[-2:],info);pb=centers(b.shape[-2:],info);radius=info['rf']/2
    qa=apply(pa,matrix)
    # Source RF, expected destination RF, destination inverse RF all visible.
    source_valid=valid(pa,radius)&valid(pa,radius,matrix)&valid(qa,radius)&valid(qa,radius,np.linalg.inv(matrix))
    dest_valid=valid(pb,radius)&valid(pb,radius,np.linalg.inv(matrix))
    ia=np.flatnonzero(source_valid);ib=np.flatnonzero(dest_valid)
    if not len(ia) or not len(ib): return None
    fa=F.normalize(a.flatten(1).T[ia],dim=1);fb=F.normalize(b.flatten(1).T[ib],dim=1)
    predicted=pb[ib[(fa@fb.T).argmax(1).cpu().numpy()]]
    error=np.linalg.norm(predicted-qa[ia],axis=1)
    # Best attainable error on the native target grid quantifies resolution.
    oracle=np.linalg.norm(qa[ia,None,:]-pb[ib][None,:,:],axis=-1).min(1)
    return dict(error=error,oracle=oracle,source=pa[ia],expected=qa[ia],predicted=predicted)


def rgb_descriptor(rgb,stride):
    # 9x9 raw RGB patches sampled on exactly the CNN grid (including offset).
    patches=F.unfold(rgb,9,padding=4).reshape(1,243,SIZE,SIZE)
    return patches


def evaluate(model,spec,images,output):
    output.mkdir(parents=True,exist_ok=True)
    infos=layer_info(spec); records=[]
    with torch.inference_mode():
        for index,rgb in enumerate(images):
            rgb=rgb[None].to(next(model.parameters()).device)
            base=spatial(model,normalize(rgb),spec)
            for transform,matrix in transforms().items():
                altered=warp(rgb,matrix)
                if 'photo' in transform: altered=(altered*.65+.15).clamp(0,1)
                current=spatial(model,normalize(altered),spec)
                for layer,info in infos.items():
                    for descriptor in ['cnn','rgb_patch']:
                        if descriptor=='cnn': a,b=base[layer][0],current[layer][0]; metadata=info
                        else:
                            # Raw descriptors at the SAME RF centers and stride.
                            grid=centers(base[layer].shape[-2:],info)
                            sample=torch.tensor(grid*2/SIZE-1,dtype=rgb.dtype,device=rgb.device).reshape(1,*base[layer].shape[-2:],2)
                            a=F.grid_sample(rgb_descriptor(rgb,info['stride']),sample,align_corners=False)[0]
                            b=F.grid_sample(rgb_descriptor(altered,info['stride']),sample,align_corners=False)[0]
                            metadata=info # same conservative RF-valid sites for paired comparisons
                        result=match(a,b,metadata,matrix)
                        if result is None:
                            records.append(dict(image=index,transform=transform,layer=layer,descriptor=descriptor,status='no_full_rf_overlap'))
                            continue
                        e=result['error'];o=result['oracle']
                        records.append(dict(image=index,transform=transform,layer=layer,descriptor=descriptor,status='ok',
                            points=len(e),mean_error=float(e.mean()),oracle_error=float(o.mean()),
                            pck4=float(np.mean(e<=4)),pck8=float(np.mean(e<=8)),pck16=float(np.mean(e<=16)),stride=info['stride'],rf=info['rf']))
                        if index==0 and descriptor=='cnn':
                            source=Image.fromarray(np.uint8(rgb[0].cpu().permute(1,2,0).numpy()*255))
                            target=Image.fromarray(np.uint8(altered[0].cpu().permute(1,2,0).numpy()*255))
                            canvas=Image.new('RGB',(SIZE*2,SIZE));canvas.paste(source);canvas.paste(target,(SIZE,0));draw=ImageDraw.Draw(canvas)
                            for j in np.linspace(0,len(e)-1,min(12,len(e))).astype(int):
                                p=tuple(result['source'][j]);q=tuple(result['predicted'][j]+[SIZE,0]);t=tuple(result['expected'][j]+[SIZE,0])
                                draw.line([p,q],fill='yellow',width=1);draw.ellipse([t[0]-2,t[1]-2,t[0]+2,t[1]+2],fill='lime')
                            canvas.save(output/f'{transform}-{layer}.png')
    summary=[]
    for transform in transforms():
        for layer in infos:
            for descriptor in ['cnn','rgb_patch']:
                rows=[r for r in records if r['transform']==transform and r['layer']==layer and r['descriptor']==descriptor and r['status']=='ok']
                summary.append(dict(transform=transform,layer=layer,descriptor=descriptor,images=len(rows),
                    **{key:float(np.mean([r[key] for r in rows])) if rows else None for key in ['mean_error','oracle_error','pck4','pck8','pck16']}))
    return dict(input_size=SIZE,thresholds_pixels=[4,8,16],rf_policy='Full theoretical RF corners valid in both views; no padded descriptors scored',
                coordinates='pixel-edge centers; grid_sample bilinear align_corners=False',summary=summary,per_image=records)
