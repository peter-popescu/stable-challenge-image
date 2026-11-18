import numpy as np
import pytest
import torch
from experiments.lejepa.next_round.frozen_readouts import stage_maps,readout_arrays,MultiReadout,EnsembleReadout
from experiments.lejepa.next_round.frozen import complementarity
from experiments.lejepa.model import make_encoder

@pytest.mark.parametrize('backbone,dimensions',[('resnet9',{'all_means':704,'last_two_avgmax':1024,'mid_mean_final_avgmax':768,'mid_mean_final_avgmaxstd':1024}),('resnet18',{'all_means':600,'last_two_avgmax':960,'all_means_final_max':920,'avgmaxstd':960})])
def test_stage_readout_dimensions_and_native_parity(backbone,dimensions):
    torch.manual_seed(42);model=make_encoder({'backbone':backbone,'readout':'gap' if backbone=='resnet18' else 'max2'}).eval()
    x=torch.randn(2,3,96,96)
    with torch.inference_mode():
        maps=stage_maps(model,x,backbone);zs=readout_arrays(maps,backbone)
        torch.testing.assert_close(maps['stage4'],model.spatial(x),rtol=0,atol=0)
        torch.testing.assert_close(zs['gap' if backbone=='resnet18' else 'max2'],model(x),rtol=0,atol=0)
        for k,d in dimensions.items():assert zs[k].shape==(2,d)
        for z in zs.values():assert z.shape[1]<=1024 and torch.isfinite(z).all()
        k='all_means';torch.testing.assert_close(MultiReadout(model,backbone,k)(x),zs[k])
        if backbone=='resnet9':
            duplicate=EnsembleReadout(model,model)(x)
            torch.testing.assert_close(duplicate[:,:512],duplicate[:,512:],rtol=0,atol=0)

def test_complementarity_partitions_and_oracle():
    y=np.array([0,0,1,1]);a=np.array([0,0,0,0]);b=np.array([0,1,1,0])
    r=complementarity(y,a,b)
    assert r['oracle_accuracy']==.75
    assert r['both_correct']==r['a_only_correct']==r['b_only_correct']==r['both_wrong']==.25
    assert r['agreement']+r['disagreement']==1
