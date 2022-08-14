import argparse
import pandas as pd
import os
import scanpy as sc
from sklearn.metrics.cluster import adjusted_rand_score
# from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
import numpy as np
import cv2
import torchvision.transforms as transforms
from utils import Cal_Spatial_Net, Stats_Spatial_Net, _hungarian_match, seed_everything
from train_transformer import train_nano_fov, test_nano_fov
from sklearn.decomposition import PCA
import torch
import matplotlib.pyplot as plt
import random
from scipy.optimize import linear_sum_assignment

os.environ['R_HOME'] = '/opt/R/4.0.2/lib/R'
os.environ['R_USER'] = '~/anaconda3/lib/python3.8/site-packages/rpy2'
os.environ['LD_LIBRARY_PATH'] = '/opt/R/4.0.2/lib/R/lib'
os.environ['PYTHONHASHSEED'] = '1234'
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

def gen_adatas(root, id, img_name):
    adata = sc.read(os.path.join(root, id, 'sampledata.h5ad'))
    adata.var_names_make_unique()
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    ncluster = len(set(adata.obs['merge_cell_type']))

    print(os.path.join(root, id, 'CellComposite_%s.jpg'%(img_name)))
    img = cv2.imread(os.path.join(root, id, 'CellComposite_%s.jpg'%(img_name)))
    height, width, c = img.shape
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    overlay = cv2.imread(os.path.join(root, id, 'CompartmentLabels_%s.tif'%(img_name)))
    overlay = cv2.cvtColor(overlay, cv2.COLOR_BGR2GRAY)
    print(overlay.shape)

    # if opt.use_gray:
    #     img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    transform = transforms.ToTensor()
    img = transform(img)
    overlay = transform(overlay)

    patchs = []
    w, h = 60, 60
    
    for coor in adata.obsm['spatial']:
        x, y = coor
        img_p = img[:, int(y-h):int(y+h), int(x-w): int(x+w)]

        patchs.append(img_p.flatten()) # 4 * h * w
    patchs = np.stack(patchs)


    df = pd.DataFrame(patchs, index=adata.obs.index)
    adata.obsm['imgs'] = df


    Cal_Spatial_Net(adata, rad_cutoff=80)
    Stats_Spatial_Net(adata)
    return adata

@torch.no_grad()
def infer(opt, r=0):
    seed = opt.seed
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    ids = [
        'fov1', 'fov2', 'fov3', 'fov4', 'fov5',
        'fov6', 'fov7', 'fov8', 'fov9', 'fov10',
        'fov11', 'fov12', 'fov13', 'fov14', 'fov15',
        'fov16', 'fov17', 'fov18', 'fov19', 'fov20'
    ]
    img_names = [
        'F001', 'F002', 'F003', 'F004', 'F005',
        'F006', 'F007', 'F008', 'F009', 'F010',
        'F011', 'F012', 'F013', 'F014', 'F015',
        'F016', 'F017', 'F018', 'F019', 'F020',
    ]

    adatas = list()
    for id, name in zip(ids, img_names):
        adata = gen_adatas(opt, opt.root, id, name)
        adatas.append(adata)
    
    sp = opt.save_path
    adata = test_nano_fov(opt, adatas, hidden_dims=[512, 30],  n_epochs=opt.epochs, save_loss=True, 
                lr=opt.lr, random_seed=opt.seed, save_path=sp, ncluster=opt.ncluster, repeat=r)

    print(adata.obsm['pred'].shape)

    sc.pp.neighbors(adata, use_rep='pred')
    sc.tl.umap(adata)
    plt.rcParams["figure.figsize"] = (3, 3)

    fig_save_path = '../results/nanostring/figures'
    if not os.path.exists(fig_save_path):
        os.makedirs(fig_save_path)
    sc.settings.figdir = fig_save_path
    ax = sc.pl.umap(adata, color=['merge_cell_type'], show=False, title='combined latent variables')
    plt.savefig(os.path.join(fig_save_path, 'umap_final.pdf'), bbox_inches='tight')

    adata.obsm['imgs'] = adata.obsm['imgs'].to_numpy()

    ## we use leiden algorithm for nanostring dataset
    if opt.cluster_method == 'leiden':
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        sc.pp.neighbors(adata, 8, use_rep='pred', random_state=seed)
            
        def res_search(adata_pred, ncluster, seed, iter=200):
            start = 0; end =3
            i = 0
            while(start < end):
                if i >= iter: return res
                i += 1
                res = (start + end) / 2
                print(res)
                # seed_everything(seed)
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

        res = res_search(adata, 8, seed)
    
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        sc.tl.leiden(adata, resolution=res, key_added='leiden', random_state=seed)

        obs_df = adata.obs.dropna()
        ARI = adjusted_rand_score(obs_df['leiden'], obs_df['merge_cell_type'])

        print('ARI: %.2f'%ARI)

        cell_type = list(set(adata.obs['merge_cell_type']))
        ground_truth = [i for i in range(len(cell_type))]
        gt = np.zeros(adata.obs['merge_cell_type'].shape)
        for i in range(len(ground_truth)):
            ct = cell_type[i]
            idx = (adata.obs['merge_cell_type'] == ct)
            gt[idx] = i
        gt = gt.astype(np.int32)

        pred = adata.obs['leiden'].to_numpy().astype(np.int32)
        layers = []
        cs = ['' for i in range(pred.shape[0])]
        gt_cs = ['' for i in range(pred.shape[0])]
        match = _hungarian_match(pred, gt, len(set(pred)), len(set(gt)))
        colors = {'lymphocyte': '#E57272FF',
            'Mcell': '#FFCA27FF',
            'tumors': "#A6CEE3",
            'epithelial': "#D3E057FF",
            'mast': '#5B6BBFFF',
            'endothelial': '#26C5D9FF',
            'fibroblast': '#26A599FF',
            'neutrophil': '#B967C7FF'
            }
        cs = ['' for i in range(pred.shape[0])]
        gt_cs = ['' for i in range(pred.shape[0])]

        for ind, j in enumerate(adata.obs['merge_cell_type'].tolist()):
            gt_cs[ind] = colors[j]

        for outc, gtc in match:
            idx = (pred == outc)
            for j in range(len(idx)):
                if idx[j]:
                    cs[j] = colors[cell_type[gtc]]
        adata.obs['cmap'] = cs
        adata.obs['gtcmap'] = gt_cs
        adata.write('../results/nanostring/process_data_%s.h5ad'%opt.pretrain[:-4])

def train(opt, r=0):
    seed_everything(opt.seed)
    ids = [
        'fov1', 'fov2', 'fov3', 'fov4', 'fov5',
        'fov6', 'fov7', 'fov8', 'fov9', 'fov10',
        'fov11', 'fov12', 'fov13', 'fov14', 'fov15',
        'fov16', 'fov17', 'fov18', 'fov19', 'fov20'
    ]
    img_names = [
        'F001', 'F002', 'F003', 'F004', 'F005',
        'F006', 'F007', 'F008', 'F009', 'F010',
        'F011', 'F012', 'F013', 'F014', 'F015',
        'F016', 'F017', 'F018', 'F019', 'F020',
    ]

    adatas = list()
    for id, name in zip(ids, img_names):
        adata = gen_adatas(opt.root, id, name)
        adatas.append(adata)

    sp = os.path.join(opt.save_path, 'all')
    if not os.path.exists(sp):
        os.makedirs(sp)

    train_nano_fov(opt, adatas, hidden_dims=[512, 30],  n_epochs=opt.epochs, save_loss=True, 
                lr=opt.lr, random_seed=opt.seed, save_path=sp, ncluster=opt.ncluster, repeat=r)


# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--lr', type=float, default=1e-3)
#     parser.add_argument('--root', type=str, default='../dataset/nanostring')
#     parser.add_argument('--epochs', type=int, default=2000)
#     parser.add_argument('--id', type=str, default='fov1')
#     parser.add_argument('--img_name', type=str, default='F001')
#     parser.add_argument('--seed', type=int, default=1234)
#     parser.add_argument('--save_path', type=str, default='../checkpoint/nanostring_final')
#     parser.add_argument('--ncluster', type=int, default=8)
#     parser.add_argument('--repeat', type=int, default=1)
#     parser.add_argument('--use_gray', type=float, default=0)
#     parser.add_argument('--test_only', type=int, default=0)
#     parser.add_argument('--pretrain', type=str, default='final.pth')
#     opt = parser.parse_args()

def train_nano(opt):
    if opt.test_only:
        infer(opt)
    else:
        train(opt, 0)
