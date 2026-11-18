"""Shared label-efficiency episodes, cosine neighbors/retrieval, and clustering."""
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score


def episodes(labels, seed=42, repeats=20):
    rows=[]
    # Nested support sets per episode; all eligible classes have >=1 query.
    for repeat in range(repeats):
        rng=np.random.default_rng(seed+repeat)
        orders={int(c):rng.permutation(np.flatnonzero(labels==c)) for c in np.unique(labels)}
        for shots in [1,5,10,15]:
            eligible=[c for c,ids in orders.items() if len(ids)>shots]
            support=np.concatenate([orders[c][:shots] for c in eligible])
            query=np.concatenate([orders[c][shots:] for c in eligible])
            rows.append(dict(repeat=repeat,shots=shots,classes=eligible,support=support.tolist(),query=query.tolist()))
    return rows


def unit(x):
    return x/np.maximum(np.linalg.norm(x,axis=1,keepdims=True),1e-12)


def evaluate(z, labels, splits, n_clusters=30):
    x=unit(z.astype('float64'))
    similarity=x@x.T; np.fill_diagonal(similarity,-np.inf)
    # Stable tie ordering is row ID order, shared across every model.
    ranked=np.argsort(-similarity,axis=1,kind='stable')[:,:len(labels)-1]
    knn={}
    for k in [1,5,10]:
        pred=[np.bincount(labels[row[:k]]).argmax() for row in ranked]
        knn[str(k)]=float(np.mean(np.asarray(pred)==labels))
    average_precision=[]
    for i,row in enumerate(ranked):
        relevant=labels[row]==labels[i]
        precision=np.cumsum(relevant)/np.arange(1,len(row)+1)
        average_precision.append(float((precision*relevant).sum()/max(relevant.sum(),1)))
    clusters=[]
    for seed in [42,43,44,45,46]:
        assignments=KMeans(n_clusters=n_clusters,random_state=seed,n_init=10).fit_predict(x)
        clusters.append(dict(seed=seed,ari=float(adjusted_rand_score(labels,assignments)),nmi=float(normalized_mutual_info_score(labels,assignments))))
    fewshot=[]
    for split in splits:
        support=np.asarray(split['support']);query=np.asarray(split['query']);classes=np.asarray(split['classes'])
        # No dataset-wide fitted normalization; prototype uses support labels only.
        prototypes=unit(np.stack([x[support[labels[support]==c]].mean(0) for c in classes]))
        predicted=classes[(x[query]@prototypes.T).argmax(1)]
        accuracy=float(np.mean(predicted==labels[query]))
        macro=float(np.mean([np.mean(predicted[labels[query]==c]==c) for c in classes]))
        fewshot.append(dict(shots=split['shots'],repeat=split['repeat'],accuracy=accuracy,macro_accuracy=macro,classes=len(classes),queries=len(query)))
    summary={}
    for shots in [1,5,10,15]:
        rows=[r for r in fewshot if r['shots']==shots]
        summary[str(shots)]={key:dict(mean=float(np.mean([r[key] for r in rows])),sd=float(np.std([r[key] for r in rows],ddof=1))) for key in ['accuracy','macro_accuracy']}
        summary[str(shots)]['classes']=rows[0]['classes']
    return dict(dimensions=z.shape[1],normalization='Per-image L2, no fitted centering/scaling',distance='cosine',
        knn=knn,knn_protocol='Leave-one-out labeled reference bank; uniform voting, lowest class ID breaks ties; fixed k, no selection',
        retrieval_map=float(np.mean(average_precision)),retrieval_ap=average_precision,
        retrieval_macro_map=float(np.mean([np.mean(np.asarray(average_precision)[labels==c]) for c in np.unique(labels)])),
        clustering_k=n_clusters,clustering_k_source='Fixed30 from documented challenge class IDs0..29, not inferred from labels',clustering=clusters,clustering_ari=float(np.mean([r['ari'] for r in clusters])),clustering_nmi=float(np.mean([r['nmi'] for r in clusters])),
        fewshot=summary,episodes=fewshot,fewshot_probe='Cosine nearest support-class centroid;20 fixed repeats;eligible classes require shots+1 images')
