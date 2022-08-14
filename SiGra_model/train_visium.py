import argparse
import pandas as pd
import os
import scanpy as sc
from sklearn.metrics.cluster import adjusted_rand_score
# from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
import numpy as np
import cv2
import torchvision.transforms as transforms
from utils import Cal_Spatial_Net, Stats_Spatial_Net, mclust_R, seed_everything
from train_transformer import train_img, test_img
from sklearn.decomposition import PCA
import torch
from scipy.optimize import linear_sum_assignment
import matplotlib.pyplot as plt

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

    img = cv2.imread(os.path.join(opt.root,opt.id, 'spatial/full_image.tif'))
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
    adata = test_img(adata, model_path, hidden_dims=[512, 30])
    adata = mclust_R(adata, used_obsm='pred', num_cluster=opt.ncluster)
    obs_df = adata.obs.dropna()
    ARI = adjusted_rand_score(obs_df['mclust'], obs_df['Ground Truth'])

    print('ari is %.2f'%(ARI))

    # plt.rcParams["figure.figsize"] = (3, 3)
    save_path = os.path.join(result_path, opt.id)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    plt.rcParams["figure.figsize"] = (3, 3)
    sc.settings.figdir = save_path
    ax=sc.pl.spatial(adata, color=['mclust'], title=['scGIT (ARI=%.2f)'%(ARI)], show=False)
    plt.savefig(os.path.join(save_path, 'spatial.pdf'), bbox_inches='tight')

    sc.pp.neighbors(adata, n_neighbors=20, use_rep='pred')
    sc.tl.umap(adata)
    plt.rcParams["figure.figsize"] = (3, 3)
    sc.settings.figdir = save_path
    ax = sc.pl.umap(adata, color=['Ground Truth'], show=False, title='scGIT', legend_loc='on data')
    plt.savefig(os.path.join(save_path, 'umap_final.pdf'), bbox_inches='tight')

    # adata.write(os.path.join(save_path, 'results.h5ad'))
    if opt.id == '151507':
        # plot_genes = ['RELN', 'C1QL2', 'ADCYAP1', 'SYT2', 'PCP4', 'SEMA3E', 'MBP']
        plot_genes = ['RELN', 'C1QL2', 'ADCYAP1', 'SYT2', 'PCP4', 'SEMA3E']#, 'MBP']
    elif opt.id == '151676':
        plot_genes = ['MYH11', 'C1QL2', 'CUX2', 'SYT2', 'PCP4', 'SEMA3E']#, 'MBP']
    else:
        return ARI

    if not os.path.exists(os.path.join(save_path, 'selected_genes')):
        os.makedirs(os.path.join(save_path, 'selected_genes'))

    for plot_gene in plot_genes:
        fig, axs = plt.subplots(1, 2, figsize=(8, 4))
        sc.pl.spatial(adata, img_key="hires", color=plot_gene, show=False, title="Raw_"+plot_gene, vmax='p99', ax=axs[0])
        sc.pl.spatial(adata, img_key="hires", color=plot_gene, show=False, title="scGIT_"+plot_gene, layer='recon', vmax='p99', ax=axs[1])
        plt.savefig(os.path.join(save_path, 'selected_genes', '%s.png'%(plot_gene)))
        plt.savefig(os.path.join(save_path, 'selected_genes', '%s.pdf'%(plot_gene)))

        plt.close()

    fig, axs = plt.subplots(1,2, figsize=(8,4))
    sc.pl.stacked_violin(adata, plot_genes, groupby='mclust', ax=axs[0], title='scGIT_RAW', show=False, vmax=0.3)#, save='Violin_raw.pdf')
    sc.pl.stacked_violin(adata, plot_genes, groupby='mclust', ax=axs[1], title='scGIT_RECON', show=False, layer='recon', vmax=0.5)#, save='Violin_recon.pdf')
    plt.savefig(os.path.join(save_path, 'selected_genes', 'heatmap.png'))
    plt.close()

    return ARI

def train(opt, r):
    seed_everything(opt.seed)
    adata = sc.read(os.path.join(opt.root, opt.id, 'sampledata.h5ad'))
    adata.var_names_make_unique()
    sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=3000, check_values=False)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    Ann_df = pd.read_csv('%s/%s/annotation.txt'%(opt.root, opt.id), sep='\t', header=None, index_col=0)
    Ann_df.columns = ['Ground Truth']
    adata.obs['Ground Truth'] = Ann_df.loc[adata.obs_names, 'Ground Truth']

    img = cv2.imread(os.path.join(opt.root,opt.id, 'spatial/full_image.tif'))
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
    Stats_Spatial_Net(adata)

    sp = os.path.join(opt.save_path, opt.id)
    if not os.path.exists(sp):
        os.makedirs(sp)
    
    adata = train_img(adata, hidden_dims=[512, 30],  n_epochs=opt.epochs, save_loss=True, 
                lr=opt.lr, random_seed=opt.seed, save_path=sp, ncluster=opt.ncluster, repeat=r)

    # we use mclust for 10x dataset
    if opt.cluster_method == 'mclust':
        adata = mclust_R(adata, used_obsm='pred', num_cluster=opt.ncluster)
        obs_df = adata.obs.dropna()
        ARI = adjusted_rand_score(obs_df['mclust'], obs_df['Ground Truth'])

        print('ari is %.2f'%(ARI))
        return ARI

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--lr', type=float, default=1e-3)
#     parser.add_argument('--root', type=str, default='../dataset/DLPFC')
#     parser.add_argument('--epochs', type=int, default=1000)
#     parser.add_argument('--id', type=str, default='151673')
#     parser.add_argument('--seed', type=int, default=1234)
#     parser.add_argument('--save_path', type=str, default='../checkpoint/transformer_final')
#     parser.add_argument('--ncluster', type=int, default=7)
#     parser.add_argument('--repeat', type=int, default=5)
#     parser.add_argument('--use_gray', type=float, default=0)
#     parser.add_argument('--test_only', type=int, default=0)
#     parser.add_argument('--pretrain', type=str, default='final.pth')
#     opt = parser.parse_args()

def train_10x(opt):
    opt.cluster_method = 'mclust'
    if opt.test_only:
        ari = infer(opt)
    else:
        if not os.path.exists(os.path.join(opt.save_path, opt.id)):
            os.makedirs(os.path.join(opt.save_path, opt.id))
    
        logger = open(os.path.join(opt.save_path, opt.id, 'logger.txt'), 'w')
        for i in range(opt.repeat):
            ARI = train(opt, i)
            logger.write('%.2f\n'%(ARI))
            print('ARI is %.2f' %(ARI))
