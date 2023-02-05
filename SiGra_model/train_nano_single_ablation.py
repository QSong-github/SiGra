import argparse
import pandas as pd
import os
import scanpy as sc
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
import numpy as np
import cv2
import torchvision.transforms as transforms
from utils import Cal_Spatial_Net, Stats_Spatial_Net, _hungarian_match, seed_everything
from train_transformer import train_nano_fov, test_nano_fov, train_nano_fov_ablation, test_nano_fov_ablation
from sklearn.decomposition import PCA
import torch
import matplotlib.pyplot as plt
import random
from scipy.optimize import linear_sum_assignment
import anndata as ad

os.environ['R_HOME'] = '/opt/R/4.0.2/lib/R'
os.environ['R_USER'] = '~/anaconda3/lib/python3.8/site-packages/rpy2'
os.environ['LD_LIBRARY_PATH'] = '/opt/R/4.0.2/lib/R/lib'
os.environ['PYTHONHASHSEED'] = '1234'
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

def _hungarian_match(flat_preds, flat_target, preds_k, target_k):
    num_samples = flat_target.shape[0]
    num_correct = np.zeros((preds_k, target_k))
    for c1 in range(preds_k):
        for c2 in range(target_k):
            votes = int(((flat_preds==c1)*(flat_target==c2)).sum())
            num_correct[c1, c2] = votes
    metrics = num_samples-num_correct
    match = linear_sum_assignment(metrics)
    match = np.array(list(zip(*match)))
    res = []
    for out_c, gt_c in match:
        res.append((out_c, gt_c))
    return res

def generate_global(adatas, idxs, height, width):
    adata_alls = []
    for i in range(len(idxs)):
        for j in range(len(idxs[0])):
            idx = idxs[i][j]
            adata = adatas[idx]
            
            ind = adata.obs['cx'].isna()
            adata = adata[~ind]
            
            cx = adata.obs['cx']
            cy = adata.obs['cy']
            
            cx_g = cx + (j) * width
            cy_g = cy + (i) * height
            
            adata.obs['cx_g'] = cx_g
            adata.obs['cy_g'] = cy_g
            
            adata_alls.append(adata)
    return adata_alls


def match_leiden_to_cell(adata):
    leiden_number = list(set(adata.obs['leiden']))
    dicts = {}
    for ln in leiden_number:
        ind = (adata.obs['leiden'] == ln)
        temp = adata[ind]
        df = temp.obs['merge_cell_type'].value_counts()
        dicts[int(ln)] = df.index[0]
    return dicts
    

def gen_adatas(root, id, img_name, suffix=''):
    print(id)
    adata = sc.read(os.path.join(root, id, 'sampledata.h5ad'))
    adata.var_names_make_unique()

    ind = adata.obs['merge_cell_type'].isna()
    adata = adata[~ind]

    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    ncluster = len(set(adata.obs['merge_cell_type']))

    print(os.path.join(root, id, 'CellComposite_%s%s.jpg'%(img_name, suffix)))
    img = cv2.imread(os.path.join(root, id, 'CellComposite_%s%s.jpg'%(img_name, suffix)))
    height, width, c = img.shape
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # overlay = cv2.imread(os.path.join(root, id, 'CompartmentLabels_%s.tif'%(img_name)))
    # overlay = cv2.cvtColor(overlay, cv2.COLOR_BGR2GRAY)
    # print(overlay.shape)

    # if opt.use_gray:
    #     img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    transform = transforms.ToTensor()
    img = transform(img)
    # overlay = transform(overlay)

    patchs = []
    w, h = 60, 60
    # w, h = 50, 50


    df = pd.DataFrame(index=adata.obs.index)
    df['cx'] = adata.obs['cx']
    df['cy'] = adata.obs['cy']
    arr = df.to_numpy()
    adata.obsm['spatial'] = arr
    
    for coor in adata.obsm['spatial']:
        x, y = coor
        img_p = img[:, int(y-h):int(y+h), int(x-w): int(x+w)]
        # print(img_p.shape)
        patchs.append(img_p.flatten()) # 4 * h * w
    patchs = np.stack(patchs)


    df = pd.DataFrame(patchs, index=adata.obs.index)
    adata.obsm['imgs'] = df


    Cal_Spatial_Net(adata, rad_cutoff=80)
    Stats_Spatial_Net(adata)
    return adata
    
def calc_global_pos(processed_data_root, data_root, idxs, height=3648, width=5472, num_fovs=30):
    adatas = []
    for i in range(1, num_fovs+1):
        adata_root = os.path.join(data_root, 'fov%d'%(i), 'sampledata.h5ad')
        adata = sc.read(adata_root)
        adatas.append(adata)
    
    adata_global = generate_global(adatas, idxs, height, width)
    adata_global = ad.concat(adata_global)
    
    adata = sc.read(processed_data_root)
    adata.obs['cx_g'] = adata_global.obs.loc[adata.obs_names, 'cx_g']
    adata.obs['cy_g'] = adata_global.obs.loc[adata.obs_names, 'cy_g']
    
    return adata

# @torch.no_grad()
# def infer(opt, r=0):
#     seed = opt.seed
#     random.seed(seed)
#     os.environ['PYTHONHASHSEED'] = str(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed(seed)
#     os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
#     torch.backends.cudnn.deterministic = True
#     torch.backends.cudnn.benchmark = False
#     ids = [
#         'fov1', 'fov2', 'fov3', 'fov4', 'fov5',
#         'fov6', 'fov7', 'fov8', 'fov9', 'fov10',
#         'fov11', 'fov12', 'fov13', 'fov14', 'fov15',
#         'fov16', 'fov17', 'fov18', 'fov19', 'fov20'
#     ]
#     img_names = [
#         'F001', 'F002', 'F003', 'F004', 'F005',
#         'F006', 'F007', 'F008', 'F009', 'F010',
#         'F011', 'F012', 'F013', 'F014', 'F015',
#         'F016', 'F017', 'F018', 'F019', 'F020',
#     ]

#     adatas = list()
#     for id, name in zip(ids, img_names):
#         adata = gen_adatas(opt, opt.root, id, name)
#         adatas.append(adata)
    
#     sp = opt.save_path
#     adata = test_nano_fov(opt, adatas, hidden_dims=[512, 30],  n_epochs=opt.epochs, save_loss=True, 
#                 lr=opt.lr, random_seed=opt.seed, save_path=sp, ncluster=opt.ncluster, repeat=r)

#     print(adata.obsm['pred'].shape)

#     sc.pp.neighbors(adata, use_rep='pred')
#     sc.tl.umap(adata)
#     plt.rcParams["figure.figsize"] = (3, 3)

#     fig_save_path = '../results/nanostring/figures'
#     if not os.path.exists(fig_save_path):
#         os.makedirs(fig_save_path)
#     sc.settings.figdir = fig_save_path
#     ax = sc.pl.umap(adata, color=['merge_cell_type'], show=False, title='combined latent variables')
#     plt.savefig(os.path.join(fig_save_path, 'umap_final.pdf'), bbox_inches='tight')

#     adata.obsm['imgs'] = adata.obsm['imgs'].to_numpy()

#     ## we use leiden algorithm for nanostring dataset
#     if opt.cluster_method == 'leiden':
#         random.seed(seed)
#         os.environ['PYTHONHASHSEED'] = str(seed)
#         np.random.seed(seed)
#         torch.manual_seed(seed)
#         torch.cuda.manual_seed(seed)
#         os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
#         torch.backends.cudnn.deterministic = True
#         torch.backends.cudnn.benchmark = False
#         sc.pp.neighbors(adata, 8, use_rep='pred', random_state=seed)
            
#         def res_search(adata_pred, ncluster, seed, iter=200):
#             start = 0; end =3
#             i = 0
#             while(start < end):
#                 if i >= iter: return res
#                 i += 1
#                 res = (start + end) / 2
#                 print(res)
#                 # seed_everything(seed)
#                 random.seed(seed)
#                 os.environ['PYTHONHASHSEED'] = str(seed)
#                 np.random.seed(seed)
#                 torch.manual_seed(seed)
#                 torch.cuda.manual_seed(seed)
#                 os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
#                 torch.backends.cudnn.deterministic = True
#                 torch.backends.cudnn.benchmark = False
#                 sc.tl.leiden(adata_pred, random_state=seed, resolution=res)
#                 count = len(set(adata_pred.obs['leiden']))
#                 # print(count)
#                 if count == ncluster:
#                     print('find', res)
#                     return res
#                 if count > ncluster:
#                     end = res
#                 else:
#                     start = res
#             raise NotImplementedError()

#         res = res_search(adata, 8, seed)
    
#         random.seed(seed)
#         os.environ['PYTHONHASHSEED'] = str(seed)
#         np.random.seed(seed)
#         torch.manual_seed(seed)
#         torch.cuda.manual_seed(seed)
#         os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
#         torch.backends.cudnn.deterministic = True
#         torch.backends.cudnn.benchmark = False

#         sc.tl.leiden(adata, resolution=res, key_added='leiden', random_state=seed)

#         obs_df = adata.obs.dropna()
#         ARI = adjusted_rand_score(obs_df['leiden'], obs_df['merge_cell_type'])

#         print('ARI: %.2f'%ARI)

#         cell_type = list(set(adata.obs['merge_cell_type']))
#         ground_truth = [i for i in range(len(cell_type))]
#         gt = np.zeros(adata.obs['merge_cell_type'].shape)
#         for i in range(len(ground_truth)):
#             ct = cell_type[i]
#             idx = (adata.obs['merge_cell_type'] == ct)
#             gt[idx] = i
#         gt = gt.astype(np.int32)

#         pred = adata.obs['leiden'].to_numpy().astype(np.int32)
#         layers = []
#         cs = ['' for i in range(pred.shape[0])]
#         gt_cs = ['' for i in range(pred.shape[0])]
#         match = _hungarian_match(pred, gt, len(set(pred)), len(set(gt)))
#         colors = {'lymphocyte': '#E57272FF',
#             'Mcell': '#FFCA27FF',
#             'tumors': "#A6CEE3",
#             'epithelial': "#D3E057FF",
#             'mast': '#5B6BBFFF',
#             'endothelial': '#26C5D9FF',
#             'fibroblast': '#26A599FF',
#             'neutrophil': '#B967C7FF'
#             }
#         cs = ['' for i in range(pred.shape[0])]
#         gt_cs = ['' for i in range(pred.shape[0])]

#         for ind, j in enumerate(adata.obs['merge_cell_type'].tolist()):
#             gt_cs[ind] = colors[j]

#         for outc, gtc in match:
#             idx = (pred == outc)
#             for j in range(len(idx)):
#                 if idx[j]:
#                     cs[j] = colors[cell_type[gtc]]
#         adata.obs['cmap'] = cs
#         adata.obs['gtcmap'] = gt_cs
#         adata.write('../results/nanostring/process_data_%s.h5ad'%opt.pretrain[:-4])

@torch.no_grad()
def infer(opt, r=0, umap_save_path='../results/nano/Lung5-1/figures'):
    seed = opt.seed
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    gene_only = False
    img_only = False
    combine_only = False
    if opt.g_weight == 0 and opt.c_weight == 0:
        img_only = True
    elif opt.i_weight == 0 and opt.c_weight == 0:
        gene_only = True
    else:
        combine_only = True

    ids = ['fov%d'%(i) for i in range(1, opt.num_fov+1)]
    img_names = ['F0%02d'%(i) for i in range(1, opt.num_fov+1)]

    adatas = list()
    for id, name in zip(ids, img_names):
        adata = gen_adatas(opt.root, id, name, opt.suffix)
        adatas.append(adata)

    sp = os.path.join(opt.save_path, 'all')
    if not os.path.exists(sp):
        os.makedirs(sp)
    df = adatas[0].obs.dropna()
    nclusters = opt.ncluster
    print('nclusters', nclusters)

    adata, vloss = test_nano_fov_ablation(opt, adatas, hidden_dims=[opt.h_dim1, opt.h_dim2],  n_epochs=opt.epochs, save_loss=True,
                lr=opt.lr, random_seed=opt.seed, save_path=sp, ncluster=nclusters, repeat=r, model_name=opt.pretrain,
                gene_only=gene_only, img_only=img_only, combine_only=combine_only, use_img_loss=opt.use_img_loss)
    
    adata.obsm['imgs'] = adata.obsm['imgs'].to_numpy()
    if gene_only:
        rep = 'gene_pred'
    elif img_only:
        rep = 'img_pred'
    else:
        rep = 'pred'
    # adata.write('../checkpoint/sigra_nano_lung5-1/processed_data_600.h5ad')
    sc.pp.neighbors(adata, use_rep=rep)
    sc.tl.umap(adata)

    plt.rcParams['figure.figsize'] = (3, 3)

    if not os.path.exists(umap_save_path):
        os.makedirs(umap_save_path)
    fig_save_path = umap_save_path
    sc.settings.figdir = fig_save_path
    print('save umap')
    ax = sc.pl.umap(adata, color=['merge_cell_type'], show=False, title='combined latent variables')
    plt.savefig(os.path.join(fig_save_path, 'umap_%s.png'%(opt.pretrain[:-4])), bbox_inches='tight')
    seed = opt.seed

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
    ss = silhouette_score(adata.obsm['pred'], adata.obs['leiden'])
    chs = calinski_harabasz_score(adata.obsm['pred'], adata.obs['leiden'])
    dbs = davies_bouldin_score(adata.obsm['pred'], adata.obs['leiden'])
    # print('ARI: %.2f'%ARI)

    df = pd.DataFrame(adata.obs['leiden'], index=adata.obs.index)
    df.to_csv(os.path.join(umap_save_path, 'leiden_cluster.csv'))


    adata_save_path = '%s'%opt.save_path
    if not os.path.exists(adata_save_path):
        os.makedirs(adata_save_path)

    adata.write('%s/processed_data_%s.h5ad'%(opt.save_path, opt.pretrain[:-4]))
    return ARI, vloss, ss, chs, dbs

def train(opt, r=0):
    seed_everything(opt.seed)
    ids = ['fov%d'%(i) for i in range(1, opt.num_fov+1)]
    img_names = ['F0%02d'%(i) for i in range(1, opt.num_fov+1)]

    adatas = list()
    for id, name in zip(ids, img_names):
        adata = gen_adatas(opt.root, id, name, opt.suffix)
        adatas.append(adata)

    sp = os.path.join(opt.save_path, 'all')
    if not os.path.exists(sp):
        os.makedirs(sp)

    train_nano_fov_ablation(opt, adatas, hidden_dims=[opt.h_dim1, opt.h_dim2],  n_epochs=opt.epochs, save_loss=True, 
                lr=opt.lr, random_seed=opt.seed, save_path=sp, ncluster=opt.ncluster, repeat=r, use_img_loss=opt.use_img_loss,
                gene_weight=opt.g_weight, img_weight=opt.i_weight, combine_weight=opt.c_weight)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--root', type=str, default='../dataset/nanostring')
    parser.add_argument('--epochs', type=int, default=600)
    parser.add_argument('--id', type=str, default='lung9-1')
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--save_path', type=str, default='../checkpoint/nano_ablation')
    parser.add_argument('--ncluster', type=int, default=8)
    parser.add_argument('--repeat', type=int, default=0)
    parser.add_argument('--use_gray', type=float, default=0)
    parser.add_argument('--test_only', type=int, default=0)
    parser.add_argument('--suffix', type=str, default='')
    parser.add_argument('--pretrain', type=str, default='final_0.pth')
    parser.add_argument('--ratio', type=float, default=0.1)
    parser.add_argument('--g_weight', type=float, default=0)
    parser.add_argument('--i_weight', type=float, default=0)
    parser.add_argument('--c_weight', type=float, default=1)
    parser.add_argument('--use_combine', type=int, default=1)
    parser.add_argument('--use_img_loss', type=int, default=0)
    parser.add_argument('--h_dim1', type=int, default=64)
    parser.add_argument('--h_dim2', type=int, default=32)
    parser.add_argument('--cluster_method', type=str, default='leiden')
    parser.add_argument('--num_fov', type=int, default=20)
    opt = parser.parse_args()

    # logger = open('../result/pruning_nano_ablation.txt', 'a')
    # logger.write('id: %s, suffix: %s, '%(opt.id, opt.suffix))

    if opt.test_only:
        ari_all, val_loss, ss, chs, dbs = infer(opt, umap_save_path='../results/nano/%s/figures'%(opt.id))
    else:
        if not os.path.exists(os.path.join(opt.save_path, opt.id)):
            os.makedirs(os.path.join(opt.save_path, opt.id))
        train(opt)
        ari_all, val_loss, ss, chs, dbs = infer(opt, umap_save_path='../results/nano/%s/figures'%(opt.id))

    
    # logger.write('ari_all: %.4f, val_loss: %.4f, silhouette_score: %.4f, calinski_harabasz_score: %.4f, davies_bouldin_score: %.4f\n\n'%(ari_all, val_loss, ss, chs, dbs))

