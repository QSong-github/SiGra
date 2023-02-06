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
import anndata
import argparse



from scipy.optimize import linear_sum_assignment
def _hungarian_match(flat_preds, flat_target, preds_k, target_k):
    num_samples = flat_preds.shape[0]
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

def show_spagcn(root, id, imgname, ncluster):
    seed = 1234
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    dataroot = os.path.join(root, id, 'sampledata.h5ad')
    adata = sc.read(dataroot)
    adata.var_names_make_unique()
    sc.pp.normalize_per_cell(adata)

    imgroot = os.path.join(root, id, 'CellComposite_%s.jpg'%(imgname))
    img = cv2.imread(imgroot)
    # pos = os.path.join(root, id, 'spatial/tissue_positions_list.csv')
    # sp = pd.read_csv(pos, sep=',', header=None, na_filter=False, index_col=0)

    # adata.obs['x1'] = sp[1]
    adata.obs['x2'] = adata.obs['cy'].astype(np.int64)
    adata.obs['x3'] = adata.obs['cx'].astype(np.int64)
    adata.obs['x4'] = adata.obs['cy'].astype(np.int64)
    adata.obs['x5'] = adata.obs['cx'].astype(np.int64)
    # we don't have spots in single cell, so using pixel for both

    # Ann_df = pd.read_csv(os.path.join(root, id, 'annotation.txt'), sep='\t', header=None, index_col=0)
    # Ann_df.columns = ['Ground Truth']
    # adata.obs['Ground Truth'] = Ann_df.loc[adata.obs_names, 'Ground Truth']

    s=1
    b=49
    x_array = adata.obs['x2'].tolist()
    y_array = adata.obs['x3'].tolist()
    x_pixel = adata.obs['x4'].tolist()
    y_pixel = adata.obs['x5'].tolist()

    adj=spg.calculate_adj_matrix(x=x_array,y=y_array, x_pixel=x_pixel, y_pixel=y_pixel, image=img, beta=b, alpha=s, histology=True)
    # sc.pp.normalize_per_cell(adata)
    sc.pp.log1p(adata)

    p=0.5 
    #Find the l value given p
    l=spg.search_l(p, adj, start=0.1, end=1000, tol=0.01, max_run=100)

    r_seed=t_seed=n_seed=100
    res=spg.search_res(adata, adj, l, ncluster, start=1.6, step=0.1, tol=5e-4, lr=0.001, max_epochs=500, r_seed=r_seed, 
    t_seed=t_seed, n_seed=n_seed)
    # res = 0.7
    # l = 1.5

    clf=spg.SpaGCN()
    clf.set_l(l)

    clf.train(adata,adj,init_spa=True,init="louvain",res=res, tol=5e-3, lr=0.05, max_epochs=200)
    y_pred, prob, z =clf.predict()
    adata.obs["pred"]= y_pred
    adata.obs["pred"]=adata.obs["pred"].astype('category')

    adata.obsm['feat'] = z.detach().cpu().numpy()

    adj_2d=spg.calculate_adj_matrix(x=x_array,y=y_array, histology=False)
    refined_pred=spg.refine(sample_id=adata.obs.index.tolist(), pred=adata.obs["pred"].tolist(), dis=adj_2d, shape="hexagon")
    adata.obs["refined_pred"]=refined_pred
    adata.obs["refined_pred"]=adata.obs["refined_pred"].astype('category')


    # gt = adata.obs['Ground Truth']
    # gts = []
    # for cont in gt:
    #     if 'Layer' in str(cont):
    #         cont = int(cont.split('_')[-1])
    #     elif 'WM' in str(cont):
    #         cont = 0
    #     else:
    #         cont = -1
    #     gts.append(cont)
    # gt = np.stack(gts)

    # idx = (gt > -1)
    preds = adata.obs['refined_pred']
    # preds = pred[idx]
    gt = adata.obs['merge_cell_type'].astype('category').cat.codes
    cellid2name = {}
    for gt, name in zip(list(gt), adata.obs['merge_cell_type']):
        if not gt in cellid2name:
            cellid2name[gt] = name

    # match = _hungarian_match(preds.astype(np.int32), gt.astype(np.int32), ncluster, ncluster)
    # layers = []

    # pred2gt = {}
    # for outc, gtc in match:
    #     pred2gt[outc] = gtc
    # for pred in preds:
    #     layers.append(cellid2name[pred2gt[pred]])
    # adata.obs['refined_cell_names'] = layers


    obs_df = adata.obs.dropna()
    ARI = adjusted_rand_score(obs_df['refined_pred'], obs_df['merge_cell_type'])
    # ss = silhouette_score(adata.obsm['feat'], adata.obs['refined_pred'])
    # ch = calinski_harabasz_score(adata.obsm['feat'], adata.obs['refined_pred'])
    # db = davies_bouldin_score(adata.obsm['feat'], adata.obs['refined_pred'])
    # print('ari is %.2f, silhouette: %.2f, calinski: %.2f, db: %.2f'%(ARI, ss, ch, db))
    print('ari is %.2f'%(ARI))

    # save_path = 'results/spagcn_nano/%s'%(id)
    # if not os.path.exists(save_path):
    #     os.makedirs(save_path)
    # sc.pp.neighbors(adata, use_rep='feat')
    # sc.tl.umap(adata)
    # plt.rcParams["figure.figsize"] = (3, 3)
    # sc.settings.figdir = save_path
    # ax = sc.pl.umap(adata, color=['cell_type'], show=False, title='SpaGCN', na_in_legend=False, legend_loc='on data')
    # plt.savefig(os.path.join(save_path, 'umap_final.pdf'), bbox_inches='tight')
    # plt.close('all')
    # # save spatial
    # ax=sc.pl.spatial(adata, color=['refined_pred'], title=['spagcn (ARI=%.2f)'%(ARI)], show=False)
    # plt.savefig(os.path.join(save_path, 'spatial.pdf'), bbox_inches='tight')
    # plt.close('all')    

    # return ARI, ss, ch, db, adata
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


    # id = '151673'
    # root = 'dataset/DLPFC'
    # adata = sc.read(os.path.join(root, id, 'sampledata.h5ad'))
    # adata.var_names_make_unique()
    # show_stagate(adata)
    num_fovs = opt.num_fov
    ids = ['fov%d'%(i) for i in range(1, num_fovs+1)]
    imgnames = ['F0%02d'%(i) for i in range(1, num_fovs+1)]
    # ids = ['fov%d'%(i) for i in range(1, num_fovs+1)]
    # imgnames = ['F0%02d'%(i) for i in range(1, num_fovs+1)]

    repeat_time = 1
    keep_record = dict()
    # root = '../dataset/nanostring'
    # save_path = './spagcn/lung9-1'
    root = opt.root
    save_path = opt.save_path


    if not os.path.exists(save_path):
        os.makedirs(save_path)

    display_results = {}

    adatas = []

    for id, imname in zip(ids, imgnames):
        print(id, imname)
        ncluster=opt.ncluster
        # adata = sc.read(os.path.join(root, id, 'sampledata.h5ad'))
        display_results[id] = []
        ARI, adata = show_spagcn(root, id, imname, ncluster)
        display_results[id].append([ARI])
        refine_pred = adata.obs['refined_pred']
        merge_cell = adata.obs['merge_cell_type']
        df = pd.DataFrame(index=adata.obs.index)
        df['refined_pred'] = refine_pred
        df['merge_cell_type'] = merge_cell
        df.to_csv(os.path.join(save_path, '%s.csv'%(id)))
        adatas.append(adata)
    
    arrays = []
    for k,v in display_results.items():
        # print(k, v)
        arrays.append(v[0])
    
    arr = np.array(arrays)
    print(arr.shape)
    df = pd.DataFrame(arr, columns=['ari'], index=ids)
    df.to_csv(os.path.join(save_path, 'spagcn.csv'))

    adata_pred = anndata.concat(adatas)
    adata_pred.write(os.path.join(save_path, 'spagcn.h5ad'))