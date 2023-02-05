import os
# from statistics import covariance
import scanpy as sc
import pandas as pd
import numpy as np
import torch
from sklearn.metrics.cluster import adjusted_rand_score
import SpaGCNcode as spg
import cv2
import random
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
import matplotlib.pyplot as plt
import argparse
import anndata

from scipy.optimize import linear_sum_assignment
def _hungarian_match(flat_preds, flat_target, preds_k, target_k):
    num_samples = flat_target.shape[0]
    num_k = preds_k
    num_correct = np.zeros((num_k, num_k))
    for c1 in range(num_k):
        for c2 in range(num_k):
            votes = int(((flat_preds==c1)*(flat_target==c2)).sum())
            num_correct[c1, c2] = votes
    
    match = linear_sum_assignment(num_samples-num_correct)
    match = np.array(list(zip(*match)))
    res = []
    for out_c, gt_c in match:
        res.append((out_c, gt_c))
    return res

def show_scanpy(root, id, ncluster, resolution=0.1):
    adata = sc.read(os.path.join(root, id, 'sampledata.h5ad'))
    adata.var_names_make_unique()
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    if 'highly_variable' in adata.var.columns:
        adata =  adata[:, adata.var['highly_variable']]
    else:
        adata = adata   
    
    sc.pp.neighbors(adata, n_neighbors=10, n_pcs=40)
    sc.tl.pca(adata, svd_solver='arpack')
    # t = np.arange(ncluster)
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
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            sc.tl.leiden(adata_pred, random_state=seed, resolution=res)
            count = len(set(adata_pred.obs['leiden']))
            # print(count)
            if count == ncluster:
                print('find', res)
                return res
            if count > ncluster:
                end = res
            else:
                start = res
        raise NotImplementedError()
    res = res_search(adata, ncluster, 1234)
    sc.tl.leiden(adata, resolution=res, random_state=1234)
    print(len(set(list(adata.obs['leiden']))))

    obs_df = adata.obs.dropna()
    ARI = adjusted_rand_score(obs_df['leiden'], obs_df['merge_cell_type'])
    print(ARI)
    return ARI, adata


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--root', type=str, default='../dataset/nanostring_Lung5_Rep1')
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--id', type=str, default='lung5-1')
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--save_path', type=str, default='../checkpoint/sigra_nano_lung5-1')
    parser.add_argument('--ncluster', type=int, default=8)
    parser.add_argument('--use_gray', type=float, default=0)
    parser.add_argument('--test_only', type=int, default=0)
    parser.add_argument('--pretrain', type=str, default='final.pth')
    parser.add_argument('--cluster_method', type=str, default='leiden')
    parser.add_argument('--num_fov', type=int, default=30)
    opt = parser.parse_args()
    
    root = opt.root
    save_path = opt.save_path
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    num_fov = opt.num_fov

    ids = ['fov%d'%i for i in range(1, num_fov+1)]
    # resolution = [1.65]
    ncluster = opt.ncluster

    # repeat_time = 1
    keep_record = dict()
    # root = 'dataset/nanostring'

    display_results = {}
    adatas = []

    for it, id in enumerate(ids):
        # ncluster = 8
        # adata = sc.read(os.path.join(root, id, 'sampledata.h5ad'))
        display_results[id] = []
        ARI, adata = show_scanpy(root, id, ncluster)
        display_results[id].append([ARI])
        adatas.append(adata)

        df = pd.DataFrame(index=adata.obs.index)
        df['scanpy'] = adata.obs['leiden']
        df['merge_cell_type'] = adata.obs['merge_cell_type']
        df.to_csv(os.path.join(save_path, '%s.csv'%id))

    
    arrays = []
    for k,v in display_results.items():
        arrays.append(v[0])
    
    arr = np.array(arrays)
    print(arr.shape)
    df = pd.DataFrame(arr, columns=['ari'], index=ids)
    df.to_csv(os.path.join(save_path, 'scanpy.csv'))

    adata_pred = anndata.concat(adatas)
    adata_pred.write(os.path.join(save_path, 'scanpy.h5ad'))