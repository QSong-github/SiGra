import argparse
import pandas as pd
import os
import scanpy as sc
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
import numpy as np
import cv2
import torchvision.transforms as transforms
from utils import Cal_Spatial_Net, Stats_Spatial_Net, mclust_R, seed_everything
from train_transformer import train_img, test_img, train_10x_all, test_10x_all
from sklearn.decomposition import PCA
import torch
from scipy.optimize import linear_sum_assignment
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
import random

os.environ['R_HOME'] = '/opt/R/4.0.2/lib/R'
os.environ['R_USER'] = '~/anaconda3/lib/python3.8/site-packages/rpy2'
os.environ['LD_LIBRARY_PATH'] = '/opt/R/4.0.2/lib/R/lib'
os.environ['PYTHONHASHSEED'] = '1234'
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

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

def split_adata(adata, ratio, seed=1234):
    cells = adata.shape[0]
    X = np.arange(cells)
    X_train, X_test = train_test_split(X, test_size=ratio, random_state=seed)
    # adata_train = adata[X_train, :]
    # adata_test = adata[X_test, :]
    return X_train, X_test

def load_adata(opt, id):
    seed_everything(opt.seed)
    adata = sc.read(os.path.join(opt.root, id, 'sampledata.h5ad'))
    adata.var_names_make_unique()
    sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=3000, check_values=False)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    Ann_df = pd.read_csv('%s/%s/annotation.txt'%(opt.root, id), sep='\t', header=None, index_col=0)
    Ann_df.columns = ['Ground Truth']
    adata.obs['Ground Truth'] = Ann_df.loc[adata.obs_names, 'Ground Truth']
    img = cv2.imread(os.path.join(opt.root,id, 'spatial/%s'%(opt.img_name)))
    if opt.use_gray:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    transform = transforms.ToTensor()
    img = transform(img)
    patchs = []
    for coor in adata.obsm['spatial']:
        py, px = coor
        img_p = img[:, px-25:px+25, py-25:py+25].flatten()
        patchs.append(img_p)
    patchs = np.stack(patchs)
    df = pd.DataFrame(patchs, index=adata.obs.index)
    adata.obsm['imgs'] = df
    X_train, X_test = split_adata(adata, opt.ratio, opt.seed)
    X_train_idx = np.zeros(adata.shape[0])
    # X_test_idx = np.zeros(adat.shape[0])
    X_train_idx[X_train] = 1
    # X_test_idx[X_test] = 1

    adata.obs['X_train'] = X_train_idx
    # adata.obs['X_test'] = X_test_idx
    return adata

@torch.no_grad()
def infer(opt, result_path='../results/10x_final/'):
    seed_everything(opt.seed)
    adata = sc.read(os.path.join(opt.root, opt.id, 'sampledata.h5ad'))
    adata.var_names_make_unique()
    sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=3000, check_values=False)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    Ann_df = pd.read_csv('%s/%s/annotation.txt'%(opt.root, opt.id), sep='\t', header=None, index_col=0)
    Ann_df.columns = ['Ground Truth']
    adata.obs['Ground Truth'] = Ann_df.loc[adata.obs_names, 'Ground Truth']

    sc.tl.rank_genes_groups(adata, "Ground Truth", method="wilcoxon")

    img = cv2.imread(os.path.join(opt.root,opt.id, 'spatial/%s'%(opt.img_name)))
    if opt.use_gray:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    transform = transforms.ToTensor()
    img = transform(img)

    patchs = []
    for coor in adata.obsm['spatial']:
        py, px = coor
        img_p = img[:, px-25:px+25, py-25:py+25].flatten()
        patchs.append(img_p)
    patchs = np.stack(patchs)
    df = pd.DataFrame(patchs, index=adata.obs.index)
    adata.obsm['imgs'] = df

    Cal_Spatial_Net(adata, rad_cutoff=150)

    model_path = os.path.join(os.path.join(opt.save_path, opt.id, opt.pretrain))
    adata_pred, val_loss = test_img(adata, model_path, hidden_dims=[512, 30])
    if opt.cluster_method == 'mclust':
        pred = mclust_R(adata_pred, used_obsm='pred', num_cluster=opt.ncluster)
        obs_df = pred.obs.dropna()
        ARI_all = adjusted_rand_score(obs_df['Ground Truth'], obs_df['mclust'])

        # X_test_idx = (adata_pred.obs['X_train'] == 0)
        # pred = mclust_R(adata_pred[X_test_idx, :], used_obsm='pred', num_cluster=opt.ncluster)
        # obs_df = pred.obs.dropna()
        # ARI_val = adjusted_rand_score(obs_df['Ground Truth'], obs_df['mclust'])

        # silhouette_score, calinski_harabasz_score, davies_bouldin_score
        ss = silhouette_score(adata_pred.obsm['pred'], pred.obs['mclust'])
        chs = calinski_harabasz_score(adata_pred.obsm['pred'], pred.obs['mclust'])
        dbs = davies_bouldin_score(adata_pred.obsm['pred'], pred.obs['mclust'])

        print('ari_all and val_loss are %.4f, %.4f'%(ARI_all, val_loss))
        print('silhouette_score, calinski_harabasz_score, davies_bouldin_score are %.4f, %.4f, %.4f'%(ss, chs, dbs))
        return ARI_all, val_loss, ss, chs, dbs
    else:
        res = res_search(adata_pred, opt.ncluster, opt.seed, iter=200)
        sc.tl.leiden(adata_pred, resolution=res)
        obs_df = adata_pred.obs.dropna()
        ARI_all = adjusted_rand_score(obs_df['Ground Truth'], obs_df['leiden'])

        ss = silhouette_score(adata_pred.obsm['pred'], pred.obs['leiden'])
        chs = calinski_harabasz_score(adata_pred.obsm['pred'], pred.obs['leiden'])
        dbs = davies_bouldin_score(adata_pred.obsm['pred'], pred.obs['leiden'])

        # X_test_idx = (adata_pred.obs['X_train'] == 0)
        # adata_test = adata_pred[X_test_idx, :]
        # res = res_search(adata_test, opt.ncluster, opt.seed, iter=200)
        # sc.tl.leiden(adata_test, resolution=res)
        # obs_df = adata_test.obs.dropna()
        # ARI_val = adjusted_rand_score(obs_df['Ground Truth'], obs_df['leiden'])
        print('ari_all and val_loss are %.4f, %.4f'%(ARI_all, val_loss))
        print('silhouette_score, calinski_harabasz_score, davies_bouldin_score are %.4f, %.4f, %.4f'%(ss, chs, dbs))
        return ARI_all, val_loss, ss, chs, dbs

def infer_all(opt):
    adatas = []
    n_clusters = [7,7,7,7,
                  5,5,5,5,
                  7,7,7,7,]
    # for id in ['151507', '151508', '151509', '151510',
    #         '151669', '151670', '151671', '151672',
    #         '151673', '151674', '151675', '151676']:


    for id in ['151507', '151508']:
        adata = load_adata(opt, id)

        Cal_Spatial_Net(adata, rad_cutoff=150)
        Stats_Spatial_Net(adata)
        adatas.append(adata)

    sp = os.path.join(opt.save_path, opt.id, str(opt.ratio))
    if not os.path.exists(sp):
        os.makedirs(sp)
    adata_lists = test_10x_all(adatas, hidden_dims=[512, 30],  n_epochs=opt.epochs, save_loss=True, 
            lr=opt.lr, random_seed=opt.seed, save_path=sp, ncluster=opt.ncluster, 
            repeat=1, use_combine=0, use_img_loss=1,
            lambda_1 = opt.g_weight, lambda_2 = opt.i_weight, lambda_3 = opt.c_weight)
    ARIS = []
    # use mclust to cluster
    if opt.cluster_method == 'mclust':
        for slice_i, adata in enumerate(adata_lists):
            X_test_idx = (adata.obs['X_train'] == 0)
            adata_test = adata[X_test_idx, :]

            adata_test = mclust_R(adata_test, used_obsm='pred', num_cluster=n_clusters[slice_i])
            obs_df = adata_test.obs.dropna()
            ARI = adjusted_rand_score(obs_df['Ground Truth'], obs_df['mclust'])

            print('ari is %.4f'%(ARI))
            ARIS.append(ARI)
        return ARIS

    elif opt.cluster_metho == 'leiden':
        for slice_i, adata in enumerate(adata_lists):
            X_test_idx = (adata.obs['X_train'] == 0)
            adata_test = adata[X_test_idx, :]
            res = res_search(adata_test, n_clusters[slice_i], opt.seed, iter=200)
            sc.tl.leiden(adata_test, resolution=res)
            obs_df = adata_test.obs.dropna()
            ARI = adjusted_rand_score(obs_df['Ground Truth'], obs_df['leiden'])
            ARIS.append(ARI)
            print('ari is %.4f'%(ARI))
        return ARIs

def train_all(opt):
    adatas = []
    n_clusters = [7,7,7,7,
                  5,5,5,5,
                  7,7,7,7,]
    # for id in ['151507', '151508', '151509', '151510',
    #         '151669', '151670', '151671', '151672',
    #         '151673', '151674', '151675', '151676']:


    for id in ['151507', '151508']:
        adata = load_adata(opt, id)

        Cal_Spatial_Net(adata, rad_cutoff=150)
        Stats_Spatial_Net(adata)
        adatas.append(adata)

    sp = os.path.join(opt.save_path, opt.id, str(opt.ratio))
    if not os.path.exists(sp):
        os.makedirs(sp)
    
    adata_lists = train_10x_all(adatas, hidden_dims=[512, 30],  n_epochs=opt.epochs, save_loss=True, 
                lr=opt.lr, random_seed=opt.seed, save_path=sp, ncluster=opt.ncluster, 
                repeat=1, use_combine=0, use_img_loss=1,
                lambda_1 = opt.g_weight, lambda_2 = opt.i_weight, lambda_3 = opt.c_weight)

    ARIS = []
    # use mclust to cluster
    if opt.cluster_method == 'mclust':
        for slice_i, adata in enumerate(adata_lists):
            X_test_idx = (adata.obs['X_train'] == 0)
            adata_test = adata[X_test_idx, :]

            adata_test = mclust_R(adata_test, used_obsm='pred', num_cluster=n_clusters[slice_i])
            obs_df = adata_test.obs.dropna()
            ARI = adjusted_rand_score(obs_df['Ground Truth'], obs_df['mclust'])

            print('ari is %.4f'%(ARI))
            ARIS.append(ARI)
        return ARIS

    elif opt.cluster_metho == 'leiden':
        for slice_i, adata in enumerate(adata_lists):
            X_test_idx = (adata.obs['X_train'] == 0)
            adata_test = adata[X_test_idx, :]
            res = res_search(adata_test, n_clusters[slice_i], opt.seed, iter=200)
            sc.tl.leiden(adata_test, resolution=res)
            obs_df = adata_test.obs.dropna()
            ARI = adjusted_rand_score(obs_df['Ground Truth'], obs_df['leiden'])
            ARIS.append(ARI)
            print('ari is %.4f'%(ARI))
        return ARIs

def res_search(adata, ncluster, seed, iter=200):
    # binary search to find the resolution
    start = 0; end = 3 
    i = 0 
    while(start < end):
        if i >= iter: return res
        i += 1
        res = (start + end) / 2
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        sc.tl.leiden(adata, random_state=seed, resolution=res)
        count = len(set(adata.obs['leiden']))

        if count == ncluster:
            print('find', res)
            return res
        if count > ncluster:
            end = res
        else:
            start = res
    raise NotImplementedError()

def train(opt):
    adata = load_adata(opt, opt.id)

    Cal_Spatial_Net(adata, rad_cutoff=150)
    Stats_Spatial_Net(adata)

    sp = os.path.join(opt.save_path, opt.id, str(opt.ratio))
    if not os.path.exists(sp):
        os.makedirs(sp)
    
    # adata_pred, val_loss = train_img(adata, hidden_dims=[512, 30],  n_epochs=opt.epochs, save_loss=True,
    #             lr=opt.lr, random_seed=opt.seed, save_path=sp, ncluster=opt.ncluster, 
    #             repeat=1, use_combine=1, use_img_loss=0,
    #             lambda_1 = opt.g_weight, lambda_2 = opt.i_weight, lambda_3 = opt.c_weight)
    adata_pred, val_loss = train_img(adata, hidden_dims=[512, 30],  n_epochs=opt.epochs, save_loss=True,
                lr=opt.lr, random_seed=opt.seed, save_path=sp, ncluster=opt.ncluster, 
                repeat=1, use_combine=opt.use_combine, use_img_loss=opt.use_img_loss,
                lambda_1 = opt.g_weight, lambda_2 = opt.i_weight, lambda_3 = opt.c_weight)


    if opt.cluster_method == 'mclust':
        pred = mclust_R(adata_pred, used_obsm='pred', num_cluster=opt.ncluster)
        obs_df = pred.obs.dropna()
        ARI_all = adjusted_rand_score(obs_df['Ground Truth'], obs_df['mclust'])

        # X_test_idx = (adata_pred.obs['X_train'] == 0)
        # pred = mclust_R(adata_pred[X_test_idx, :], used_obsm='pred', num_cluster=opt.ncluster)
        # obs_df = pred.obs.dropna()
        # ARI_val = adjusted_rand_score(obs_df['Ground Truth'], obs_df['mclust'])

        # silhouette_score, calinski_harabasz_score, davies_bouldin_score
        ss = silhouette_score(adata_pred.obsm['pred'], pred.obs['mclust'])
        chs = calinski_harabasz_score(adata_pred.obsm['pred'], pred.obs['mclust'])
        dbs = davies_bouldin_score(adata_pred.obsm['pred'], pred.obs['mclust'])

        print('ari_all and val_loss are %.4f, %.4f'%(ARI_all, val_loss))
        print('silhouette_score, calinski_harabasz_score, davies_bouldin_score are %.4f, %.4f, %.4f'%(ss, chs, dbs))
        return ARI_all, val_loss, ss, chs, dbs
    else:
        res = res_search(adata_pred, opt.ncluster, opt.seed, iter=200)
        sc.tl.leiden(adata_pred, resolution=res)
        obs_df = adata_pred.obs.dropna()
        ARI_all = adjusted_rand_score(obs_df['Ground Truth'], obs_df['leiden'])

        ss = silhouette_score(adata_pred.obsm['pred'], pred.obs['leiden'])
        chs = calinski_harabasz_score(adata_pred.obsm['pred'], pred.obs['leiden'])
        dbs = davies_bouldin_score(adata_pred.obsm['pred'], pred.obs['leiden'])

        # X_test_idx = (adata_pred.obs['X_train'] == 0)
        # adata_test = adata_pred[X_test_idx, :]
        # res = res_search(adata_test, opt.ncluster, opt.seed, iter=200)
        # sc.tl.leiden(adata_test, resolution=res)
        # obs_df = adata_test.obs.dropna()
        # ARI_val = adjusted_rand_score(obs_df['Ground Truth'], obs_df['leiden'])
        print('ari_all and val_loss are %.4f, %.4f'%(ARI_all, val_loss))
        print('silhouette_score, calinski_harabasz_score, davies_bouldin_score are %.4f, %.4f, %.4f'%(ss, chs, dbs))
        return ARI_all, val_loss, ss, chs, dbs


# ifmain
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--root', type=str, default='../dataset/DLPFC')
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--id', type=str, default='151673')
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--save_path', type=str, default='../checkpoint/sigra_final')
    parser.add_argument('--ncluster', type=int, default=7)
    parser.add_argument('--repeat', type=int, default=5)
    parser.add_argument('--use_gray', type=float, default=0)
    parser.add_argument('--test_only', type=int, default=0)
    parser.add_argument('--pretrain', type=str, default='final.pth')
    parser.add_argument('--ratio', type=float, default=0.1)
    parser.add_argument('--g_weight', type=float, default=0.1)
    parser.add_argument('--i_weight', type=float, default=0.1)
    parser.add_argument('--c_weight', type=float, default=1)
    parser.add_argument('--use_combine', type=int, default=1)
    parser.add_argument('--use_img_loss', type=int, default=0)
    parser.add_argument('--cluster_method', type=str, default='mclust')
    parser.add_argument('--img_name', type=str, default='lipofucsin_151676_mu10.tif')
    opt = parser.parse_args()

    logger = open('../result/pruning_10x_grid_search5.txt', 'a')
    logger.write('id: %s, ratio: %.2f, ncluster: %d, g_weight: %.2f, i_weight: %.2f, c_weight: %.2f, cluster_method: %s\n'%(opt.id, opt.ratio, opt.ncluster, opt.g_weight, opt.i_weight, opt.c_weight, opt.cluster_method))

    if opt.test_only:
        ari_all, val_loss, ss, chs, dbs  = infer(opt)
        # print(ari_all, ari_val)

    else:
        # create path
        if not os.path.exists(os.path.join(opt.save_path, opt.id)):
            os.makedirs(os.path.join(opt.save_path, opt.id))
        ari_all, val_loss, ss, chs, dbs = train(opt)
        # print(ari_all, val_loss)
    
    # logger.write('ari_all and val_loss are: %.4f, %.4f\n\n'%(ari_all, val_loss))
    logger.write('ari_all: %.4f, val_loss: %.4f, silhouette_score: %.4f, calinski_harabasz_score: %.4f, davies_bouldin_score: %.4f\n\n'%(ari_all, val_loss, ss, chs, dbs))
    # print('silhouette_score, calinski_harabasz_score, davies_bouldin_score are %.4f, %.4f, %.4f'%(ss, chs, dbs))
