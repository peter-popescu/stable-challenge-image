"""Scalable primal ridge classifier: train normalization, held-out val selection."""
import numpy as np
from evaluate import LAMBDAS
from ..error_analysis import classification_summary


def fit_candidates(x,y,classes):
    x=np.asarray(x,dtype=np.float64)
    mu=x.mean(0);sd=x.std(0)+1e-6;x=(x-mu)/sd
    y1=(y[:,None]==classes).astype(np.float64);bar=y1.mean(0)
    # Primal rather than n-by-n dual: required for 50k/70k external examples.
    gram=x.T@x;e,u=np.linalg.eigh(gram);e=np.maximum(e,0)
    rhs=u.T@(x.T@(y1-bar))
    scale=float(e.sum()/len(x))
    return mu,sd,bar,e,u,rhs,scale


def predict(fit,x,relative):
    mu,sd,bar,e,u,rhs,scale=fit
    w=u@(rhs/np.maximum(e+relative*scale,1e-30)[:,None])
    return ((np.asarray(x,dtype=np.float64)-mu)/sd)@w+bar,w


def probe(x,y,splits,output,refit=True):
    classes=np.unique(y[splits['train']]);fit=fit_candidates(x[splits['train']],y[splits['train']],classes)
    candidates=[];best=None
    for relative in LAMBDAS:
        scores,_=predict(fit,x[splits['val']],relative)
        pred=classes[scores.argmax(1)];metrics,_=classification_summary(y[splits['val']],pred)
        row=dict(relative_lambda=float(relative),accuracy=metrics['accuracy'],macro_accuracy=metrics['macro_accuracy'])
        candidates.append(row)
        if best is None or row['accuracy']>best['accuracy']: best=row
    if refit: fit=fit_candidates(x[splits['trainval']],y[splits['trainval']],classes)
    scores,w=predict(fit,x[splits['test']],best['relative_lambda'])
    pred=classes[scores.argmax(1)];metrics,confusion=classification_summary(y[splits['test']],pred)
    np.savez(output,mu=fit[0],sd=fit[1],intercept=fit[2],weights=w,classes=classes,predictions=pred,confusion=confusion)
    return dict(validation=best,candidates=candidates,test=metrics,refit_trainval=refit,
                lambda_absolute=float(best['relative_lambda']*fit[-1]),
                standardization='training-only population std +1e-6; recomputed on trainval for final fit',
                selection='Highest validation micro accuracy; smaller lambda wins ties; test scored once after selection')


def fewshot(x,y,splits,output):
    results=[]
    for seed in [42,43,44]:
        rng=np.random.default_rng(seed)
        orders={c:rng.permutation(splits['train'][y[splits['train']]==c]) for c in np.unique(y[splits['train']])}
        for budget in [1,5,20]:
            support=np.concatenate([ids[:budget] for ids in orders.values()])
            ss=dict(splits,train=support,trainval=support)
            r=probe(x,y,ss,output/f'fewshot-{budget}-{seed}.npz',refit=False)
            results.append(dict(shots=budget,seed=seed,**r))
    return results
