import math
import pandas as pd
import numpy as np
import sklearn.neighbors
import scipy.sparse as sp
import seaborn as sns
import matplotlib.pyplot as plt
import random
import os
import torch
import scanpy as sc

from torch_geometric.data import Data
import torch.backends.cudnn as cudnn
cudnn.deterministic = True
cudnn.benchmark = False
from scipy.optimize import linear_sum_assignment


def seed_everything(seed: int):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def prefilter_genes(adata,min_counts=None,max_counts=None,min_cells=10,max_cells=None):
    if min_cells is None and min_counts is None and max_cells is None and max_counts is None:
        raise ValueError('Provide one of min_counts, min_genes, max_counts or max_genes.')
    id_tmp=np.asarray([True]*adata.shape[1],dtype=bool)
    id_tmp=np.logical_and(id_tmp,sc.pp.filter_genes(adata.X,min_cells=min_cells)[0]) if min_cells is not None  else id_tmp
    id_tmp=np.logical_and(id_tmp,sc.pp.filter_genes(adata.X,max_cells=max_cells)[0]) if max_cells is not None  else id_tmp
    id_tmp=np.logical_and(id_tmp,sc.pp.filter_genes(adata.X,min_counts=min_counts)[0]) if min_counts is not None  else id_tmp
    id_tmp=np.logical_and(id_tmp,sc.pp.filter_genes(adata.X,max_counts=max_counts)[0]) if max_counts is not None  else id_tmp
    adata._inplace_subset_var(id_tmp)
    return id_tmp, adata


def prefilter_specialgenes(adata,Gene1Pattern="ERCC",Gene2Pattern="MT-"):
    id_tmp1=np.asarray([not str(name).startswith(Gene1Pattern) for name in adata.var_names],dtype=bool)
    id_tmp2=np.asarray([not str(name).startswith(Gene2Pattern) for name in adata.var_names],dtype=bool)
    id_tmp=np.logical_and(id_tmp1,id_tmp2)
    adata._inplace_subset_var(id_tmp)
    return id_tmp, adata

class InstanceLoss(torch.nn.Module):
    def __init__(self, batch_size, temperature):
        super().__init__()
        self.batch_size = batch_size
        self.temperature = temperature

        self.mask = self.mask_correlated_samples(batch_size)
        self.criterion = torch.nn.CrossEntropyLoss(reduction='sum')
    
    def mask_correlated_samples(self, batch_size):
        N = 2 * batch_size
        mask = torch.ones((N, N))
        mask = mask.fill_diagonal_(0)
        for i in range(batch_size):
            mask[i, batch_size + i] = 0
            mask[batch_size + i, i] = 0
        mask = mask.bool()
        return mask
    
    def forward(self, zi, zj):
        N = 2 * self.batch_size
        z = torch.cat((zi, zj), dim=0)

        sim = torch.matmul(z, z.T) / self.temperature
        sim_ij = torch.diag(sim ,self.batch_size)
        sim_ji = torch.diag(sim, -self.batch_size)

        positive_samples = torch.cat((sim_ij, sim_ji), dim=0).reshape(N, 1)
        negative_samples = sim[self.mask].reshape(N, -1)

        labels = torch.zeros(N).to(positive_samples.device).long()
        logits = torch.cat((positive_samples, negative_samples), dim=1)

        loss = self.criterion(logits, labels)
        loss /= N

        return loss

class ClusterLoss(torch.nn.Module):
    def __init__(self, ncluster, temperature):
        super().__init__()
        self.ncluster = ncluster
        self.temperature = temperature

        self.mask = self.mask_correlated_clusters(ncluster)
        self.criterion = torch.nn.CrossEntropyLoss(reduction='sum')
        self.similarity_f = torch.nn.CosineSimilarity(dim=2)

    def mask_correlated_clusters(self, ncluster):
        N = 2 * ncluster
        mask = torch.ones((N, N))
        mask = mask.fill_diagonal_(0)
        for i in range(ncluster):
            mask[i, ncluster+1] = 0
            mask[ncluster+1, i] = 0
        mask = mask.bool()
        return mask
    
    def forward(self, ci, cj):
        pi = ci.sum(0).view(-1)
        pi /= pi.sum()
        ne_i = math.log(pi.size(0)) + (pi * torch.log(pi)).sum()
        pj = cj.sum(0).view(-1)
        pj /= pj.sum()
        ne_j = math.log(pj.size(0)) + (pj * torch.log(pj)).sum()

        ne_loss = ne_i + ne_j

        ci = ci.t()
        cj = cj.t()

        N = 2 * self.ncluster
        c = torch.cat((ci, cj), dim=0)
        
        sim = self.similarity_f(c.unsqueeze(1), c.unsqueeze(0)) / self.temperature
        # print(sim.shape, self.ncluster)
        sim_ij = torch.diag(sim, self.ncluster)
        sim_ji = torch.diag(sim, -self.ncluster)

        positive_clusters = torch.cat((sim_ij, sim_ji), dim=0).reshape(N,1)
        negative_clusters = sim[self.mask].reshape(N, -1)

        labels = torch.zeros(N).to(positive_clusters.device).long()
        logits = torch.cat((positive_clusters, negative_clusters), dim=1)
        loss = self.criterion(logits, labels)
        loss /= N

        return loss + ne_loss

def Transfer_img_Data(adata):
    G_df = adata.uns['Spatial_Net'].copy()
    cells = np.array(adata.obs_names)
    cells_id_tran = dict(zip(cells, range(cells.shape[0])))
    G_df['Cell1'] = G_df['Cell1'].map(cells_id_tran)
    G_df['Cell2'] = G_df['Cell2'].map(cells_id_tran)
    # print(G_df)
    # exit(0)
    e0 = G_df['Cell1'].to_numpy()
    e1 = G_df['Cell2'].to_numpy()
    edgeList = np.array((e0, e1))

    if type(adata.X) == np.ndarray:
        if 'X_train' in adata.obs.keys():
            X_train_idx = (adata.obs['X_train'].to_numpy() == 1)
            X_test_idx = (adata.obs['X_train'].to_numpy() == 0)
            print(X_train_idx) 
            data = Data(edge_index=torch.LongTensor(np.array(
                [edgeList[0], edgeList[1]])), x=torch.FloatTensor(adata.X), train_mask=list(X_train_idx), val_mask=list(X_test_idx))
            img = Data(edge_index=torch.LongTensor(np.array(
                [edgeList[0], edgeList[1]])), x=torch.FloatTensor(adata.obsm['imgs']), train_mask=list(X_train_idx), val_mask=list(X_test_idx))
        else:
            data = Data(edge_index=torch.LongTensor(np.array(
                [edgeList[0], edgeList[1]])), x=torch.FloatTensor(adata.X))  # .todense()
            img = Data(edge_index=torch.LongTensor(np.array(
                [edgeList[0], edgeList[1]])), x=torch.FloatTensor(adata.obsm['imgs']))
    else:
        if 'X_train' in adata.obs.keys():
            X_train_idx = (adata.obs['X_train'].to_numpy() == 1)
            X_test_idx = (adata.obs['X_train'].to_numpy() == 0)
            data = Data(edge_index=torch.LongTensor(np.array(
                [edgeList[0], edgeList[1]])), x=torch.FloatTensor(adata.X.todense()), train_mask=list(X_train_idx), val_mask=list(X_test_idx))
            img = Data(edge_index=torch.LongTensor(np.array(
                [edgeList[0], edgeList[1]])), x=torch.FloatTensor(adata.obsm['imgs'].to_numpy()), train_mask=list(X_train_idx), val_mask=list(X_test_idx))
        else:
            data = Data(edge_index=torch.LongTensor(np.array(
                [edgeList[0], edgeList[1]])), x=torch.FloatTensor(adata.X.todense()))  # .todense()
            img = Data(edge_index=torch.LongTensor(np.array(
                [edgeList[0], edgeList[1]])), x=torch.FloatTensor(adata.obsm['imgs'].to_numpy()))
    return data, img

def Batch_Data(adata, num_batch_x, num_batch_y, spatial_key=['X', 'Y'], plot_Stats=False):
    Sp_df = adata.obs.loc[:, spatial_key].copy()
    Sp_df = np.array(Sp_df)
    batch_x_coor = [np.percentile(Sp_df[:, 0], (1/num_batch_x)*x*100) for x in range(num_batch_x+1)]
    batch_y_coor = [np.percentile(Sp_df[:, 1], (1/num_batch_y)*x*100) for x in range(num_batch_y+1)]

    Batch_list = []
    for it_x in range(num_batch_x):
        for it_y in range(num_batch_y):
            min_x = batch_x_coor[it_x]
            max_x = batch_x_coor[it_x+1]
            min_y = batch_y_coor[it_y]
            max_y = batch_y_coor[it_y+1]
            temp_adata = adata.copy()
            temp_adata = temp_adata[temp_adata.obs[spatial_key[0]].map(lambda x: min_x <= x <= max_x)]
            temp_adata = temp_adata[temp_adata.obs[spatial_key[1]].map(lambda y: min_y <= y <= max_y)]
            Batch_list.append(temp_adata)
    if plot_Stats:
        f, ax = plt.subplots(figsize=(1, 3))
        plot_df = pd.DataFrame([x.shape[0] for x in Batch_list], columns=['#spot/batch'])
        sns.boxplot(y='#spot/batch', data=plot_df, ax=ax)
        sns.stripplot(y='#spot/batch', data=plot_df, ax=ax, color='red', size=5)
    return Batch_list


def Cal_Spatial_Net(adata, rad_cutoff=None, k_cutoff=None, model='Radius', verbose=True, use_global=False):
    if verbose:
        print('------Calculating spatial graph...')
    if use_global:
        coor = pd.DataFrame(adata.obsm['spatial_global'])
    else:
        coor = pd.DataFrame(adata.obsm['spatial'])
    coor.index = adata.obs.index
    coor.columns = ['imagerow', 'imagecol']

    if model == 'Radius':
        nbrs = sklearn.neighbors.NearestNeighbors(radius=rad_cutoff).fit(coor)
        distances, indices = nbrs.radius_neighbors(coor, return_distance=True)
        KNN_list = []
        for it in range(indices.shape[0]):
            KNN_list.append(pd.DataFrame(zip([it]*indices[it].shape[0], indices[it], distances[it])))
    
    if model == 'KNN':
        nbrs = sklearn.neighbors.NearestNeighbors(n_neighbors=k_cutoff+1).fit(coor)
        distances, indices = nbrs.kneighbors(coor)
        KNN_list = []
        for it in range(indices.shape[0]):
            KNN_list.append(pd.DataFrame(zip([it]*indices.shape[1],indices[it,:], distances[it,:])))

    KNN_df = pd.concat(KNN_list)
    KNN_df.columns = ['Cell1', 'Cell2', 'Distance']

    Spatial_Net = KNN_df.copy()
    Spatial_Net = Spatial_Net.loc[Spatial_Net['Distance']>0,]
    id_cell_trans = dict(zip(range(coor.shape[0]), np.array(coor.index), ))
    Spatial_Net['Cell1'] = Spatial_Net['Cell1'].map(id_cell_trans)
    Spatial_Net['Cell2'] = Spatial_Net['Cell2'].map(id_cell_trans)
    if verbose:
        print('The graph contains %d edges, %d cells.' %(Spatial_Net.shape[0], adata.n_obs))
        print('%.4f neighbors per cell on average.' %(Spatial_Net.shape[0]/adata.n_obs))

    adata.uns['Spatial_Net'] = Spatial_Net

def Stats_Spatial_Net(adata):
    import matplotlib.pyplot as plt
    Num_edge = adata.uns['Spatial_Net']['Cell1'].shape[0]
    Mean_edge = Num_edge/adata.shape[0]
    plot_df = pd.value_counts(pd.value_counts(adata.uns['Spatial_Net']['Cell1']))
    plot_df = plot_df/adata.shape[0]
    fig, ax = plt.subplots(figsize=[3,2])
    plt.ylabel('Percentage')
    plt.xlabel('')
    plt.title('Number of Neighbors (Mean=%.2f)'%Mean_edge)
    ax.bar(plot_df.index, plot_df)
    plt.close('all')

def mclust_R(adata, num_cluster, modelNames='EEE', used_obsm='pred', random_seed=0):
    """\
    Clustering using the mclust algorithm.
    The parameters are the same as those in the R package mclust.
    """
    
    np.random.seed(random_seed)
    import rpy2.robjects as robjects
    robjects.r.library("mclust")

    import rpy2.robjects.numpy2ri
    rpy2.robjects.numpy2ri.activate()
    r_random_seed = robjects.r['set.seed']
    r_random_seed(random_seed)
    rmclust = robjects.r['Mclust']

    res = rmclust(rpy2.robjects.numpy2ri.numpy2rpy(adata.obsm[used_obsm]), num_cluster, modelNames, verbose=False)
    mclust_res = np.array(res[-2])

    adata.obs['mclust'] = mclust_res
    adata.obs['mclust'] = adata.obs['mclust'].astype('int')
    adata.obs['mclust'] = adata.obs['mclust'].astype('category')
    return adata

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
