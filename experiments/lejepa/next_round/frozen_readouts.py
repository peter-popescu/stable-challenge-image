"""Direct frozen stage pooling; no fitted reduction or feature alignment."""
import torch
from torch import nn
from ..readouts import SpatialReadout


def stage_maps(model, x, backbone):
    maps={}
    if backbone=='resnet18':
        x=model.stem(x)
        for i,layer in enumerate(model.stages):
            x=layer(x);maps['stage'+str(i+1)]=x
    else:
        for i,layer in enumerate(list(model.conv.children())[:-1]):
            x=layer(x)
            if i in (2,7,11,16):maps['stage'+str((2,7,11,16).index(i)+1)]=x
    return maps


def readout_arrays(maps, backbone):
    """Accept torch maps; return native dimensions <=1024. Population std."""
    means=[maps['stage'+str(i)].mean((-2,-1)) for i in range(1,5)]
    last=maps['stage4'];mid=maps['stage3']
    out={k:SpatialReadout(k)(last) for k in ('gap','avgmax','avgmaxstd')}
    if backbone=='resnet9':out['max2']=SpatialReadout('max2')(last)
    out['all_means']=torch.cat(means,1)
    out['last_two_avgmax']=torch.cat([SpatialReadout('avgmax')(mid),out['avgmax']],1)
    if backbone=='resnet18':out['all_means_final_max']=torch.cat(means+[SpatialReadout('gmp')(last)],1)
    else:
        out['mid_mean_final_avgmax']=torch.cat([means[2],out['avgmax']],1)
        out['mid_mean_final_avgmaxstd']=torch.cat([means[2],out['avgmaxstd']],1)
    if any(z.shape[1]>1024 for z in out.values()):raise ValueError('Embedding exceeds competition budget')
    return out


class MultiReadout(nn.Module):
    def __init__(self,model,backbone,kind):
        super().__init__();self.encoder=model.eval();self.backbone=backbone;self.kind=kind
        self.requires_grad_(False)
    def forward(self,x):return readout_arrays(stage_maps(self.encoder,x,self.backbone),self.backbone)[self.kind]


class EnsembleReadout(nn.Module):
    def __init__(self,a,b):
        super().__init__();self.a=a.eval();self.b=b.eval();self.pool=SpatialReadout('avgmax');self.embed_dim=1024
        self.requires_grad_(False)
    def forward(self,x):return torch.cat([self.pool(self.a.spatial(x)),self.pool(self.b.spatial(x))],1)
