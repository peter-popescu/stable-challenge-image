"""Protocol fixtures and frozen-probe smoke tests, executed in Slurm."""
import os
import numpy as np
import pytest
pytestmark=pytest.mark.skipif(not os.environ.get('SLURM_JOB_ID'),reason='Compute in Slurm')


def test_episode_isolation_and_perfect_semantics():
    from experiments.lejepa.downstream.semantic import episodes,evaluate
    labels=np.repeat(np.arange(3),20);z=np.eye(3)[labels]
    splits=episodes(labels,repeats=2)
    assert splits==episodes(labels,repeats=2)
    for s in splits:
        assert not set(s['support'])&set(s['query'])
        assert len(s['support'])==s['shots']*3
    result=evaluate(z,labels,splits,n_clusters=3)
    assert result['knn']['1']==1
    assert result['retrieval_map']==1
    assert result['clustering_ari']==1
    assert result['fewshot']['15']['accuracy']['mean']==1


def test_no_self_match():
    from experiments.lejepa.downstream.semantic import evaluate
    # Unique identity features give identical off-diagonal similarity. A
    # singleton class cannot retrieve itself and must have AP zero.
    labels=np.array([0,1,1]);z=np.eye(3)
    splits=[dict(shots=s,repeat=r,classes=[1],support=[1],query=[2]) for s in [1,5,10,15] for r in [0,1]]
    result=evaluate(z,labels,splits,n_clusters=3)
    assert result['retrieval_ap'][0]==0
    assert result['knn']['1']==0


def test_population_std_backward():
    import torch
    from experiments.lejepa.readouts import SpatialReadout
    for value in [0.,1.,1e-12]:
        x=torch.full((2,3,6,6),value,requires_grad=True)
        z=SpatialReadout('avgmaxstd')(x)
        torch.testing.assert_close(z[:,-3:],torch.zeros(2,3),atol=1e-18,rtol=0)
        z.sum().backward();assert torch.isfinite(x.grad).all()
    x=torch.randn(2,3,7,9,dtype=torch.float64,requires_grad=True)
    assert torch.autograd.gradcheck(SpatialReadout('avgmaxstd'),(x,))


def test_geometry_identity_translation_and_overlap():
    import torch
    from experiments.lejepa.downstream.correspondence import match,transforms,warp,apply,valid,SIZE
    info=dict(stride=8,center=4.,rf=16)
    # Each spatial site has a unique orthogonal descriptor.
    x=torch.eye(24*24).T.reshape(24*24,24,24)
    identity=match(x,x,info,np.eye(3));assert np.max(identity['error'])==0
    shift=transforms()['translation']
    y=torch.zeros_like(x);y[:,1:,2:]=x[:,:-1,:-2]
    result=match(x,y,info,shift);assert np.max(result['error'])==0
    p=np.array([[4.,4.],[96.,96.]])
    assert valid(p,8).tolist()==[False,True]
    image=torch.rand(1,3,SIZE,SIZE)
    torch.testing.assert_close(warp(image,np.eye(3)),image,atol=2e-5,rtol=0)
    for matrix in transforms().values():np.testing.assert_allclose(apply(apply(p,matrix),np.linalg.inv(matrix)),p,atol=1e-12)


def test_layer_geometry_and_frozen_encoder():
    import torch
    from experiments.lejepa.downstream.common import load,spatial,layer_info
    for b in ['resnet9','resnet18']:
        spec=dict(backbone=b);model=load(spec);before={k:v.clone() for k,v in model.state_dict().items()}
        with torch.inference_mode():maps=spatial(model,torch.randn(2,3,96,96),spec)
        assert maps['mid'].shape[-2:]==(24,24)
        assert maps['final'].shape[-2:]==(12,12)
        assert not any(p.requires_grad for p in model.parameters())
        for k,v in model.state_dict().items():torch.testing.assert_close(v,before[k],rtol=0,atol=0)


def test_appearance_image_splits_and_probe():
    from experiments.lejepa.downstream.appearance import split_images,probe,targets
    import torch
    parts=split_images(60)
    assert len(set(parts[0])&set(parts[1]))==0
    assert len(set(parts[0])&set(parts[2]))==0
    rng=np.random.default_rng(1);x=rng.normal(size=(60,4,3));y=x@np.ones((3,2))
    result=probe(x,y,parts)
    assert result['test']['r2']>.999
    assert result['constant']['r2']<.1
    rgb=torch.ones(2,3,96,96)*.25
    np.testing.assert_allclose(targets(rgb)[:,:3],.25)
    np.testing.assert_allclose(targets(rgb)[:,3:],0)


def test_segmentation_metrics_trimap_and_linear_smoke(tmp_path):
    from experiments.lejepa.downstream.segmentation import trimap,scores,train_probe,validate
    from PIL import Image
    import torch
    mask=Image.fromarray(np.tile(np.array([1,2,3],dtype='uint8'),(6,2)))
    assert set(np.unique(trimap(mask)))=={0,1,2}
    perfect=scores(np.eye(3,dtype=int)*10)
    assert perfect['miou']==1 and perfect['foreground_iou']==1
    rng=np.random.default_rng(0);maps=rng.normal(size=(12,4,12,12)).astype('float32')
    masks=np.zeros((12,96,96),dtype='uint8');masks[:,:,32:64]=1;masks[:,:,64:]=2
    model,choice=train_probe(maps,masks,np.arange(8),np.arange(8,10),tmp_path,device='cpu',epochs=1)
    assert model.weight.shape==(3,4,1,1)
    assert choice['epoch']==1
    assert np.isfinite(validate(model,maps,masks,np.arange(10,12),'cpu')['miou'])
