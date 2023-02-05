import scanpy as sc
import datetime
import os

import torch

import STAGATE_pyG as STAGATE

import pandas as pd
import numpy as np
import torch.nn.functional as F

import scipy.sparse as sp
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics.cluster import adjusted_rand_score
from torch_geometric.loader import DataLoader
from tqdm import tqdm
import random
import argparse

import torch.backends.cudnn as cudnn
cudnn.deterministic = True
cudnn.benchmark = True

os.environ['R_HOME'] = '/opt/R/4.0.2/lib/R'
os.environ['R_USER'] = '~/anaconda3/lib/python3.8/site-packages/rpy2'
os.environ['LD_LIBRARY_PATH'] = '/opt/R/4.0.2/lib/R/lib'
os.environ['PYTHONHASHSEED'] = '1234'
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

# def stagate_codes(adata, hidden_dims=[512, 30], n_epochs=1000, lr=1e-4,
#             gradient_clipping=5., weight_decay=0.0001, verbose=True, seed=0,
#             device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')): 
    
#     random.seed(seed)                                                            
#     torch.manual_seed(seed)                                                      
#     torch.cuda.manual_seed_all(seed)                                             
#     np.random.seed(seed)                                                         
#     os.environ['PYTHONHASHSEED'] = str(seed)  

def gen_adatas(root, id):
    print(id)
    adata = sc.read(os.path.join(root, id, 'sampledata.h5ad'))
    adata.var_names_make_unique()
    ind = adata.obs['merge_cell_type'].isna()
    adata = adata[~ind]
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    df = pd.DataFrame(index=adata.obs.index)
    df['cx'] = adata.obs['cx']
    df['cy'] = adata.obs['cy']
    arr = df.to_numpy()
    adata.obsm['spatial'] = arr

    STAGATE.Cal_Spatial_Net(adata, rad_cutoff=80)
    return adata 

def train(adata, id, hidden_dims=[512, 30], n_epochs=1000, lr=0.001, key_added='STAGATE',
        random_seed=0, device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'),
        weight_decay=0.0001, save_path='./STAGATE/lung9-2/'):
    seed = random_seed
    import random
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    data = STAGATE.utils.Transfer_pytorch_Data(adata)
    model = STAGATE.STAGATE(hidden_dims=[data.x.shape[1]] + hidden_dims).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    for epoch in tqdm(range(1, n_epochs + 1)):
        model.train()
        optimizer.zero_grad()
        data = data.to(device)
        z, out = model(data.x, data.edge_index)
        loss = F.mse_loss(out, data.x)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
        optimizer.step()
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    torch.save(model.state_dict(), os.path.join(save_path, 'STAGATE_%s.pth'%(id)))

    with torch.no_grad():
        model.eval()
        pred = None
        pred_out = None
        z, out = model(data.x, data.edge_index)
        pred = z
        pred = pred.cpu().detach().numpy()
        pred_out = out.cpu().detach().numpy().astype(np.float32)
        pred_out[pred_out < 0] = 0

        adata.obsm[key_added] = pred
        adata.obsm['recon'] = pred_out

    return adata


def train_fov(adatas, hidden_dims=[512, 30], n_epochs=1000, lr=0.001, key_added='STAGATE',
        random_seed=0, device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'),
        weight_decay=0.0001, save_path='./STAGATE/lung9-2/'):
    seed = random_seed
    import random
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    datas = []
    for adata in adatas:
        data = STAGATE.utils.Transfer_pytorch_Data(adata)
        datas.append(data)
    loader = DataLoader(datas, batch_size=1, shuffle=True)
    model = STAGATE.STAGATE(hidden_dims=[datas[0].x.shape[1]] + hidden_dims).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    for epoch in tqdm(range(1, n_epochs + 1)):
        for i, batch in enumerate(loader):
            model.train()
            batch = batch.to(device)
            optimizer.zero_grad()
            z, out = model(batch.x, batch.edge_index)
            loss = F.mse_loss(out, batch.x)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()
    
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    torch.save(model.state_dict(), os.path.join(save_path, 'STAGATE.pth'))
    
    import anndata
    adata_cat = anndata.concat(adatas)

    with torch.no_grad():
        model.eval()
        pred = None
        pred_out = None
        for batch in loader:
            batch = batch.to(device)
            z, out = model(batch.x, batch.edge_index)
            if pred is None:
                pred = z.detach().cpu()
                pred_out = out.detach().cpu()
            else:
                pred = torch.cat((pred, z.detach().cpu()), dim=0)
                pred_out = torch.cat((pred_out, out.detach().cpu()), dim=0)

        pred = pred.numpy()
        pred_out = pred_out.numpy().astype(np.float32)
        pred_out[pred_out < 0] = 0

        adata_cat.obsm[key_added] = pred
        adata_cat.obsm['recon'] = pred_out

    return adata_cat

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


    # root = '../dataset/nanostring/'
    root = opt.root
    save_path = opt.save_path
    n_epochs = 1000
    num_fov = opt.num_fov
    ids = ['fov%d'%i for i in range(1, num_fov+1)]
    adatas = list()
    for id in ids:
        adata = gen_adatas(root, id)
        adata = train(adata, id, save_path=save_path, n_epochs=n_epochs)
        adata = STAGATE.mclust_R(adata, num_cluster=opt.ncluster, used_obsm='STAGATE')
        obs_df = adata.obs.dropna()
        ARI = adjusted_rand_score(obs_df['mclust'], obs_df['merge_cell_type'])
        print(ARI)

        df = pd.DataFrame(index=adata.obs.index)
        df['mclust'] = adata.obs['mclust']
        df['merge_cell_type'] = adata.obs['merge_cell_type']
        df.to_csv(os.path.join(save_path, '%s.csv'%id))

        df = pd.DataFrame(adata.obsm['STAGATE'], index=adata.obs.index)
        df.to_csv(os.path.join(save_path, '%s_STAGATE.csv'%id))


    #     adatas.append(adata)
    # # save_path = './STAGATE/lung9-2/'
    # save_path = opt.save_path
    # # n_epochs = 1000
    # n_epochs = opt.epochs
    # adata = train_fov(adatas, save_path=save_path, n_epochs=n_epochs)

    # adata = STAGATE.mclust_R(adata, num_cluster=8, used_obsm='STAGATE')

    # obs_df = adata.obs.dropna()
    # ARI = adjusted_rand_score(obs_df['mclust'], obs_df['merge_cell_type'])
    # print('Adjusted rand index = %.2f' %ARI)

    # if not os.path.exists('./STAGATE/lung9-2/'):
    #     os.makedirs('./STAGATE/lung9-2/')
    # adata.write('./STAGATE/lung9-2/adata_lung9-2.h5ad')
    # plt.rcParams["figure.figsize"] = (3, 3)

    # sc.pp.neighbors(adata, use_rep='STAGATE')
    # sc.tl.umap(adata)

    # sc.pl.umap(adata, color=["mclust", "merge_cell_type"], title=['STAGATE (ARI=%.2f)'%ARI, "Ground Truth"])
    # plt.savefig(os.path.join(save_path, 'umap.png'))