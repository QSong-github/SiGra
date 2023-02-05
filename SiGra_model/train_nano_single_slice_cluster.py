import scanpy as sc
import numpy as np
import pandas as pd
import os
import sys
import matplotlib.pyplot as plt
import random
from sklearn.metrics.cluster import adjusted_rand_score


# Load data
def gen_cluster(root, umap_save_path='../results/nano/Lung5-1/figures',seed=1234):
    adata = sc.read(root)
    print(adata.obsm['pred'].shape)

    nclusters = len(set(adata.obs['merge_cell_type']))

    sc.pp.neighbors(adata, use_rep='pred')
    print('find neighbors')

    sc.tl.umap(adata)

    plt.rcParams['figure.figsize'] = (3, 3)

    if not os.path.exists(umap_save_path):
        os.makedirs(umap_save_path)
    fig_save_path = umap_save_path
    sc.settings.figdir = fig_save_path
    print('save umap')
    ax = sc.pl.umap(adata, color=['merge_cell_type'], show=False, title='combined latent variables')
    plt.savefig(os.path.join(fig_save_path, 'umap_final.pdf'), bbox_inches='tight')


    # adata.obsm['imgs'] = adata.obsm['imgs'].to_numpy()
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)

    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

    print('find resolution for leiden')
    # sc.pp.neighbors(adata, nclusters, use_rep='pred', random_state=seed)
            
    def res_search(adata_pred, ncluster, seed, iter=200):
        start = 0; end =3
        i = 0
        while(start < end):
            if i >= iter: return res
            i += 1
            res = (start + end) / 2
            print(res)
            random.seed(seed)
            os.environ['PYTHONHASHSEED'] = str(seed)
            np.random.seed(seed)
            os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
            sc.tl.leiden(adata_pred, random_state=seed, resolution=res)
            obs_df = adata.obs.dropna()
            ARI = adjusted_rand_score(obs_df['leiden'], obs_df['merge_cell_type'])
            count = len(set(adata_pred.obs['leiden']))
            print('ARI, count: %.2f, %d'%(ARI, count))

            if count == ncluster:
                print('find', res)
                return res
            if count > ncluster:
                end = res
            else:
                start = res
        raise NotImplementedError()

    res = res_search(adata, nclusters, seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    sc.tl.leiden(adata, random_state=seed, resolution=res)
    obs_df = adata.obs.dropna()
    ARI = adjusted_rand_score(obs_df['leiden'], obs_df['merge_cell_type'])

    print('ARI: %.2f'%ARI)

    df = pd.DataFrame(adata.obs['leiden'], index=adata.obs.index)
    df.to_csv(os.path.join(umap_save_path, 'leiden_cluster.csv'))


if __name__ == '__main__':
    root = '../checkpoint/sigra_nano_lung5-1/processed_data_600.h5ad'
    umap_save_path='../results/nano/Lung5-1/figures'
    gen_cluster(root, umap_save_path)