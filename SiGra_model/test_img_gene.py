import scanpy as sc
import os
import random
import numpy as np
import torch
import pandas as pd
from sklearn.metrics.cluster import adjusted_rand_score
from scipy.optimize import linear_sum_assignment
import matplotlib.pyplot as plt
import cv2
from utils import _hungarian_match

def merge_fovs(root='pred_fov', save_name='pred.png'):
    n = ['fov1.png', 'fov2.png', 'fov3.png', 'fov4.png',
            'fov5.png', 'fov6.png', 'fov7.png', 'fov8.png',
            'fov9.png', 'fov10.png', 'fov11.png', 'fov12.png',
            'fov13.png', 'fov14.png', 'fov15.png', 'fov16.png',
            'fov17.png', 'fov18.png', 'fov19.png', 'fov20.png',
            ]

    
    im1 = cv2.imread(os.path.join(root, n[0]))
    im2 = cv2.imread(os.path.join(root, n[1]))
    im3 = cv2.imread(os.path.join(root, n[2]))
    im4 = cv2.imread(os.path.join(root, n[3]))
    imrow1 = cv2.hconcat([im1, im2, im3, im4])

    im5 = cv2.imread(os.path.join(root, n[4]))
    im6 = cv2.imread(os.path.join(root, n[5]))
    im7 = cv2.imread(os.path.join(root, n[6]))
    im8 = cv2.imread(os.path.join(root, n[7]))
    imrow2 = cv2.hconcat([im5, im6, im7, im8])

    im9 = cv2.imread(os.path.join(root, n[8]))
    im10 = cv2.imread(os.path.join(root, n[9]))
    im11 = cv2.imread(os.path.join(root, n[10]))
    im12 = cv2.imread(os.path.join(root, n[11]))
    imrow3 = cv2.hconcat([im9, im10, im11, im12])

    im13 = cv2.imread(os.path.join(root, n[12]))
    im14 = cv2.imread(os.path.join(root, n[13]))
    im15 = cv2.imread(os.path.join(root, n[14]))
    im16 = cv2.imread(os.path.join(root, n[15]))
    imrow4 = cv2.hconcat([im13, im14, im15, im16])

    im17 = cv2.imread(os.path.join(root, n[16]))
    im18 = cv2.imread(os.path.join(root, n[17]))
    im19 = cv2.imread(os.path.join(root, n[18]))
    im20 = cv2.imread(os.path.join(root, n[19]))
    imrow5 = cv2.hconcat([im17, im18, im19, im20])

    imall = cv2.vconcat([imrow5, imrow4, imrow3, imrow2, imrow1])
    cv2.imwrite(save_name, imall)

# all_h5ad = sc.read('pdata/final.h5ad')
all_h5ad = sc.read('pdata/process_data_final_900_0.h5ad')

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

# show ari with only genes
seed = 1234
all_h5ad.var_names_make_unique()
print(all_h5ad.obsm['img_pred'].shape)
sc.pp.neighbors(all_h5ad, 8, use_rep='img_pred', random_state=seed)

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
            sc.tl.leiden(adata_pred, random_state=seed, resolution=res, key_added='img_leiden')
            count = len(set(adata_pred.obs['img_leiden']))
            print(count)
            if count == ncluster:
                print('find', res)
                obs_df = adata_pred.obs.dropna()
                ARI = adjusted_rand_score(obs_df['img_leiden'], obs_df['merge_cell_type'])
                print('ARI: %.2f'%ARI)
                return res
            if count > ncluster:
                end = res
            else:
                start = res
        raise NotImplementedError()

res = res_search(all_h5ad, 8, seed)
os.environ['PYTHONHASHSEED'] = str(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
sc.tl.leiden(all_h5ad, random_state=seed, resolution=res, key_added='img_leiden')

cell_type = list(set(all_h5ad.obs['merge_cell_type']))
ground_truth = [i for i in range(len(cell_type))]
gt = np.zeros(all_h5ad.obs['merge_cell_type'].shape)
for i in range(len(ground_truth)):
    ct = cell_type[i]
    idx = (all_h5ad.obs['merge_cell_type'] == ct)
    gt[idx] = i
gt = gt.astype(np.int32)

pred = all_h5ad.obs['img_leiden'].to_numpy().astype(np.int32)
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

for ind, j in enumerate(all_h5ad.obs['merge_cell_type'].tolist()):
    gt_cs[ind] = colors[j]

for outc, gtc in match:
    idx = (pred == outc)
    for j in range(len(idx)):
        if idx[j]:
            cs[j] = colors[cell_type[gtc]]
all_h5ad.obs['imgcmap'] = cs
count = 0
start = 0
aris_gene = []
aris_img = []

genedf_g = sc.get.obs_df(
        all_h5ad,
        keys=["img_leiden"])

for id in ids:
    adata = sc.read(os.path.join('../dataset/nanostring', id, 'sampledata.h5ad'))
    start = count
    end = start + adata.shape[0]
    count = count + adata.shape[0]
    
    adata.var_names_make_unique()

    adata.obs['img_leiden'] = genedf_g.iloc[start:end]['img_leiden']
    obs_df = adata.obs.dropna()
    ari = adjusted_rand_score(obs_df['img_leiden'], obs_df['merge_cell_type'])
    ari = round(ari, 2)
    print(ari)
    aris_img.append(ari)

print(aris_img)