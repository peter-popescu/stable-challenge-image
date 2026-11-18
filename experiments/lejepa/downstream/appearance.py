"""Train-only standardization; validation ridge selection; image-disjoint tests."""
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score


def split_images(n):
    order=np.random.default_rng(42).permutation(n)
    return [order[:int(.6*n)],order[int(.6*n):int(.8*n)],order[int(.8*n):]]


def metrics(y,p):
    return dict(mse=float(mean_squared_error(y,p)),r2=float(r2_score(y,p,multioutput='uniform_average')),
                mse_per_channel=np.mean((y-p)**2,0).tolist(),r2_per_channel=r2_score(y,p,multioutput='raw_values').tolist())


def probe(x,y,splits):
    # x,y first axis is IMAGE, never pool sites before splitting.
    xx=[x[i].reshape(-1,x.shape[-1]).astype(np.float64) for i in splits]
    yy=[y[i].reshape(-1,y.shape[-1]).astype(np.float64) for i in splits]
    scale=StandardScaler().fit(xx[0]); xx=[scale.transform(v) for v in xx]
    best=None
    for alpha in [.01,1.,100.]:
        model=Ridge(alpha=alpha).fit(xx[0],yy[0])
        loss=mean_squared_error(yy[1],model.predict(xx[1]))
        if best is None or loss<best[0]: best=(loss,alpha,model)
    return dict(solver_dtype='float64',alpha=best[1],validation_mse=float(best[0]),test=metrics(yy[2],best[2].predict(xx[2])),
                constant=metrics(yy[2],np.broadcast_to(yy[0].mean(0),yy[2].shape)))


def targets(rgb, info=None):
    """Raw [0,1] RGB means/std. Local 8x8 windows at RF centers."""
    import torch
    if info is None:
        return torch.cat([rgb.mean((-1,-2)),rgb.std((-1,-2),unbiased=False)],1).numpy()
    h=96//info['stride']; centers=info['center']+np.arange(h)*info['stride']
    result=[]
    for cy in centers:
        row=[]
        for cx in centers:
            # Boundary windows clipped; document their smaller support.
            tile=rgb[:,:,max(0,round(cy)-4):min(96,round(cy)+4),max(0,round(cx)-4):min(96,round(cx)+4)]
            row.append(torch.cat([tile.mean((-1,-2)),tile.std((-1,-2),unbiased=False)],1).numpy())
        result.append(np.stack(row,1))
    return np.stack(result,1)
