from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import sklearn.neighbors
import sklearn
from sklearn.metrics.cluster import adjusted_rand_score

import torch
import torch.nn.functional as F
from torch_geometric.nn import TransformerConv, LayerNorm, GATConv, GCNConv
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
from torch_geometric.data import Data

import matplotlib.pyplot as plt
import seaborn as sns
import phenograph
import numpy as np

from copy import deepcopy
import pickle
import argparse
import scanpy as sc
import cv2
from tqdm import tqdm
import random
import pandas as pd
import scipy.sparse as sp
import os
from scipy.optimize import linear_sum_assignment
import math

cudnn.deterministic = True
cudnn.benchmark = False

os.environ['R_HOME'] = '/opt/R/4.0.2/lib/R'
os.environ['R_USER'] = '~/anaconda3/lib/python3.8/site-packages/rpy2'
os.environ['LD_LIBRARY_PATH'] = '/opt/R/4.0.2/lib/R/lib'
os.environ['PYTHONHASHSEED'] = '1234'
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

class DataContrast(torch.nn.Module):
    def __init__(self, hidden_dims, ncluster, nspots):
        super().__init__()
        [in_dim, img_dim, num_hidden, out_dim] = hidden_dims
        self.conv1 = GCNConv(in_dim, 2048)
        self.conv2 = GCNConv(2048, 4096)
        self.emb = GCNConv(4096, out_dim)

        self.conv3 = GCNConv(out_dim, num_hidden)
        self.conv4 = GCNConv(num_hidden, in_dim)

        mask = torch.Tensor(nspots, in_dim)
        torch.nn.init.uniform_(mask)
        mask = (mask > 0.5).float()
        self.mask = torch.nn.Parameter(mask)

        # self.mask = self.mask.float()

        # self.proj = TransformerConv(num_hidden, ncluster)
        self.proj = GCNConv(4096, ncluster)

        self.activate = F.elu
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, xi, edge_index):
        # print(xi.shape)
        hi1 = self.activate(self.conv1(xi, edge_index))
        hi2 = self.activate(self.conv2(hi1, edge_index))
        emb = self.activate(self.emb(hi2, edge_index))
        # combine1 = torch.concat([emb, hi2], dim=1)
        up1 = self.activate(self.conv3(emb, edge_index))
        # combine2 = torch.concat([up1, hi1], dim=1)
        up2 = self.conv4(up1, edge_index)
        ci = self.softmax(self.proj(hi2, edge_index))
        
        # print(xi.shape, self.mask.shape)
        xj = xi * self.mask
        hj1 = self.activate(self.conv1(xj, edge_index))
        hj2 = self.activate(self.conv2(hj1, edge_index))
        cj = self.softmax(self.proj(hj2, edge_index))

        return hi2, hj2, ci, cj, up2

class ImgContrast(torch.nn.Module):
    def __init__(self, hidden_dims, ncluster):
        super().__init__()
        [in_dim, img_dim, num_hidden, out_dim] = hidden_dims
        self.imgconv1 = TransformerConv(img_dim, num_hidden)
        self.imgconv2 = TransformerConv(num_hidden, out_dim)

        self.proj = TransformerConv(out_dim, ncluster)
        self.activate = F.elu


    def forward(self, xi, xj, edge_index):
        hi1 = self.activate(self.imgconv1(xi, edge_index))
        hi2 = self.imgconv2(hi1, edge_index)
        ci = self.proj(hi1, edge_index)

        hj1 = self.activate(self.imgconv1(xj, edge_index))
        hj2 = self.imgconv(hj1, edge_index)
        cj = self.proj(hj2, edge_index)

        return hi2, hj2, ci, cj
    

class TransImg2(torch.nn.Module):
    def __init__(self, hidden_dims):
        super().__init__()
        [in_dim, img_dim, num_hidden, out_dim] = hidden_dims
        # [in_dim, emb_dim, img_dim, num_hidden, out_dim] = hidden_dims

        self.conv1 = TransformerConv(in_dim, num_hidden, heads=1, dropout=0.1, beta=True)
        # self.conv1 = TransformerConv(in_dim + emb_dim, num_hidden)#, heads=1, dropout=0.1, beta=True)
        self.conv2 = TransformerConv(num_hidden, out_dim, heads=1, dropout=0.1, beta=True)
        self.conv3 = TransformerConv(out_dim, num_hidden, heads=1, dropout=0.1, beta=True)
        self.conv4 = TransformerConv(num_hidden, in_dim, heads=1, dropout=0.1, beta=True)

        self.imgconv1 = TransformerConv(img_dim, num_hidden, heads=1, dropout=0.1, beta=True)
        self.imgconv2 = TransformerConv(num_hidden, out_dim, heads=1, dropout=0.1, beta=True)
        self.imgconv3 = TransformerConv(out_dim, num_hidden, heads=1, dropout=0.1, beta=True)
        self.imgconv4 = TransformerConv(num_hidden, img_dim, heads=1, dropout=0.1, beta=True)

        self.neck = TransformerConv(out_dim * 2, out_dim, heads=1, dropout=0.1, beta=True)
        self.neck2 = TransformerConv(out_dim, out_dim, heads=1, dropout=0.1, beta=True)
        self.c3 = TransformerConv(out_dim, num_hidden, heads=1, dropout=0.1, beta=True)
        self.c4 = TransformerConv(num_hidden, in_dim, heads=1, dropout=0.1, beta=True)

        # layernorm 
        self.norm1 = LayerNorm(num_hidden)
        self.norm2 = LayerNorm(out_dim)
        # relu
        self.activate = F.elu

    def forward(self, features, img_feat, edge_index):
        h1 = self.activate(self.conv1(features, edge_index))
        h2 = self.conv2(h1, edge_index)
        h3 = self.activate(self.conv3(h2, edge_index))
        h4 = self.conv4(h3, edge_index)

        img1 = self.activate(self.imgconv1(img_feat, edge_index))
        img2 = self.imgconv2(img1, edge_index)
        img3 = self.activate(self.imgconv3(img2, edge_index))
        img4 = self.imgconv4(img3, edge_index)

        concat = torch.cat([h2, img2], dim=1)
        combine = self.activate(self.neck(concat, edge_index))
        c2 = self.neck2(combine, edge_index)
        c3 = self.activate(self.c3(c2, edge_index))
        c4 = self.c4(c3, edge_index)

        return h2, img2, c2, h4, img4, c4



class TransImg(torch.nn.Module):
    def __init__(self, hidden_dims):
        super().__init__()
        [in_dim, img_dim, num_hidden, out_dim] = hidden_dims
        # [in_dim, emb_dim, img_dim, num_hidden, out_dim] = hidden_dims

        self.conv1 = TransformerConv(in_dim, num_hidden)
        self.conv2 = TransformerConv(num_hidden, out_dim)
        self.conv3 = TransformerConv(out_dim, num_hidden)
        self.conv4 = TransformerConv(num_hidden, in_dim)

        self.imgconv1 = TransformerConv(img_dim, num_hidden)
        self.imgconv2 = TransformerConv(num_hidden, out_dim)
        self.imgconv3 = TransformerConv(out_dim, num_hidden)
        self.imgconv4 = TransformerConv(num_hidden, in_dim)

        self.neck = TransformerConv(out_dim * 2, out_dim)
        self.neck2 = TransformerConv(out_dim, out_dim)
        self.c3 = TransformerConv(out_dim, num_hidden)
        self.c4 = TransformerConv(num_hidden, in_dim)

        # layernorm 
        self.norm1 = LayerNorm(num_hidden)
        self.norm2 = LayerNorm(out_dim)
        # relu
        self.activate = F.elu

    def forward(self, features, img_feat, edge_index):
        h1 = self.activate(self.conv1(features, edge_index))
        h2 = self.conv2(h1, edge_index)
        h3 = self.activate(self.conv3(h2, edge_index))
        h4 = self.conv4(h3, edge_index)

        img1 = self.activate(self.imgconv1(img_feat, edge_index))
        img2 = self.imgconv2(img1, edge_index)
        img3 = self.activate(self.imgconv3(img2, edge_index))
        img4 = self.imgconv4(img3, edge_index)

        concat = torch.cat([h2, img2], dim=1)
        combine = self.activate(self.neck(concat, edge_index))
        c2 = self.neck2(combine, edge_index)
        c3 = self.activate(self.c3(c2, edge_index))
        c4 = self.c4(c3, edge_index)

        return h2, img2, c2, h4, img4, c4



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
        data = Data(edge_index=torch.LongTensor(np.array(
            [edgeList[0], edgeList[1]])), x=torch.FloatTensor(adata.X))  # .todense()
        img = Data(edge_index=torch.LongTensor(np.array(
            [edgeList[0], edgeList[1]])), x=torch.FloatTensor(adata.obsm['imgs']))
    else:
        data = Data(edge_index=torch.LongTensor(np.array(
            [edgeList[0], edgeList[1]])), x=torch.FloatTensor(adata.X.todense()))  # .todense()
        # test, using I as adj
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




# identify graph
def build_identity_graph(root, save_path):
    with open(root, 'rb') as f:
        data = pickle.load(f)
    
    data_a = data['gene_data_dropout']
    ids = [i for i in range(len(data_a))]

    df = pd.DataFrame()
    df['Cell1'] = ids
    df['Cell2'] = ids

    df.to_csv(save_path)

    return data, df

def build_gt_graph(root, save_path):
    with open(root, 'rb') as f:
        data = pickle.load(f)
    
    cell_cid = data['cell_cluster']
    cell1 = []
    cell2 = []
    
    for i in range(cell_cid.shape[0]):
        for j in range(cell_cid.shape[0]):
            if cell_cid[i] == cell_cid[j]:
                cell1.append(i)
                cell2.append(j)
    df = pd.DataFrame()
    df['Cell1'] = cell1
    df['Cell2'] = cell2
    return data, df
        

def build_spatial_graph(root, save_path, radius=3):
    with open(root, 'rb') as f:
        data = pickle.load(f)
    
    sx, sy = data['sx'], data['sy']
    coor = pd.DataFrame(np.stack([sx, sy], axis=1))
    coor.columns = ['imagerow', 'imagecol']
    nbrs = sklearn.neighbors.NearestNeighbors(radius=radius).fit(coor)
    distances, indices = nbrs.radius_neighbors(coor, return_distance=True)
    KNN_list = []
    for it in range(indices.shape[0]):
        KNN_list.append(pd.DataFrame(zip([it]*indices[it].shape[0], indices[it], distances[it])))
    KNN_df = pd.concat(KNN_list)
    KNN_df.columns = ['Cell1', 'Cell2', 'Distance']
    spatial_net = KNN_df.copy()
    spatial_net = spatial_net.loc[spatial_net['Distance']>0,]
    spatial_net.to_csv(save_path)
    
    return data, spatial_net
    
def mclust_R(array, num_cluster, modelNames='EEE', seed=10):
    np.random.seed(seed)
    import rpy2.robjects as robjects
    robjects.r.library("mclust")
    import rpy2.robjects.numpy2ri
    rpy2.robjects.numpy2ri.activate()
    r_random_seed = robjects.r['set.seed']
    r_random_seed(random_seed)
    rmclust = robjects.r['Mclust']
    res = rmclust(rpy2.robjects.numpy2ri.numpy2rpy(array), num_cluster, modelNames, verbose=False)
    mclust_res = np.array(res[-2])
    return mclust_res

# generate pesudo label
def get_cluster(x, n_latent):
    np.random.seed(1234)
    feat = PCA(n_components=n_latent).fit_transform(x)
    label, _, _ = phenograph.cluster(feat)
    return label

def transfer_data(root_dir, name='1', graph_type='gt'):
    pkl_file = os.path.join(root_dir, name+'.pkl')
    csv_file = os.path.join(root_dir, name+'.csv')
    adata_file = os.path.join(root_dir, name+'.h5ad')

    if graph_type == 'identity':
        data, graph = build_identity_graph(pkl_file, csv_file)
    elif graph_type == 'spatial':
        data, graph = build_spatial_graph(pkl_file, csv_file)
    elif graph_type == 'gt':
        data, graph = build_gt_graph(pkl_file, csv_file)

    e0 = graph['Cell1'].to_numpy()
    e1 = graph['Cell2'].to_numpy()
    edgeList = np.array((e0, e1))
    
    gene = data['gene_data_dropout']
    img = data['img_data_dropout']
    label = data['domain_cluster']
    label2 = data['cell_cluster']
    colors = []
    my_cmap = sns.color_palette('muted', as_cmap=True)
    for cid in data['domain_cluster']:
        colors.append(my_cmap[cid])

    gene_data = Data(edge_index=torch.LongTensor(np.array(
            [edgeList[0], edgeList[1]])), x=torch.FloatTensor(gene))
    img_data = Data(edge_index=torch.LongTensor(np.array(
            [edgeList[0], edgeList[1]])), x=torch.FloatTensor(img))

    return data, gene_data, img_data, label, label2, colors

def run_sigra_model(root_dir, name, hidden_dims=[512, 100],
    lr = 1e-3, weight_decay = 0.0001, n_epochs=200, gradient_clipping=5.,
    save_path = 'simulation_model', seed = 10,
    lambda_1=1, lambda_2=1, lambda_3=1, graph_type='spatial', gt_type='domain', vis=True):
    
    print(name)
    
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    

    df = pd.DataFrame()

    data_dict, data, img, label_domain, label_cell, colors = transfer_data(root_dir, name, graph_type=graph_type)
    if gt_type == 'domain':
        label_true = label_domain
        df['label_domain'] = label_domain
    else:
        label_true = label_cell
        df['label_cell'] = label_domain


    np.random.seed(seed)
    gene_data = PCA(n_components=100).fit_transform(data_dict['gene_data_dropout'])

    np.random.seed(seed)
    img_data = PCA(n_components=100).fit_transform(data_dict['img_data_dropout'])
    
    my_cmap = sns.color_palette('muted', as_cmap=True)
    pred, _, _ = phenograph.cluster(gene_data)
    fig, axs = plt.subplots(1,3, figsize=(15,5))
    size = 10
    ax = axs[0]
    X_embedded = TSNE(n_components=2).fit_transform(gene_data)
    ax.scatter(X_embedded[:, 0], X_embedded[:, 1], c=colors, s=size)
    ax.title.set_text('gene, ARI = %01.3f' % adjusted_rand_score(label_domain, pred))
    df['gene_gca'] = pred

    pred, _, _ = phenograph.cluster(img_data)
    ax = axs[1]
    X_embedded = TSNE(n_components=2).fit_transform(img_data)
    ax.scatter(X_embedded[:, 0], X_embedded[:, 1], c=colors, s=size)
    ax.title.set_text('img, ARI = %01.3f' % adjusted_rand_score(label_domain, pred))
    df['img_gca'] = pred

    device = 'cuda:0'
    model = TransImg(hidden_dims=[data.x.shape[1], img.x.shape[1]] + hidden_dims).to(device)
    data = data.to(device)
    img = img.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    for epoch in tqdm(range(1, n_epochs+1)):
        model.train()
        optimizer.zero_grad()
        gz,iz,cz, gout,iout,cout = model(data.x, img.x, data.edge_index)

        gloss = F.mse_loss(data.x, gout)
        iloss = F.mse_loss(data.x, iout)
        closs = F.mse_loss(data.x, cout)
        loss = gloss * lambda_1 + iloss * lambda_2 + closs * lambda_3
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)
        optimizer.step()
    
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    torch.save(model.state_dict(), os.path.join(save_path, 'final.pth'))

    with torch.no_grad():
        model.load_state_dict(torch.load(os.path.join(save_path, 'final.pth')))
        model.eval()
        gz,iz,cz, gout,iout,cout = model(data.x, img.x, data.edge_index)
    

    cz = cz.detach().cpu().numpy()
    sigra_pred, _, _ = phenograph.cluster(cz)
    plt.subplot(1, 3, 3)
    ax = axs[2]
    X_embedded = TSNE(n_components=2).fit_transform(cz)
    ax.scatter(X_embedded[:, 0], X_embedded[:, 1], c=colors, s=size)
    ax.title.set_text('Sigra, ARI = %01.3f' % adjusted_rand_score(label_domain, sigra_pred))

    ari = adjusted_rand_score(label_domain, sigra_pred)
    df['sigra'] = sigra_pred
    df.to_csv(os.path.join(save_path, name+'.csv'))

    print(name, '%.4f'%(ari))
    return sigra_pred, cz

def gen_feats(samples, Z, feat, latent, sigma, dropout):
    np.random.seed(123)
    A = np.random.random([feat, latent]) - 0.5
    noise = np.random.normal(0, sigma, size=[feat, samples])
    X = np.dot(A, Z).transpose()
    X[X < 0] = 0
    cutoff = np.exp(-dropout * (X ** 2))
    X = X + noise.T
    X[X < 0] = 0
    Y = deepcopy(X)
    rand_matrix = np.random.random(Y.shape)
    zero_mask = rand_matrix < cutoff
    Y[zero_mask] = 0
    return X, Y


def simulate_dominate_cell_type(
    spatial_clusters,
    spatial_samples,
    regions,
    gene_feat=500,
    img_feat = 500,
    gene_latent=30,
    img_latent=30,
    sigma_gene=0.1,
    sigma_img=0.1,
    dropout_gene=0.5,
    dropout_img=0.1,
    seed=123,
    use_prior=None,
):
    np.random.seed(seed)
    data = {}
    cluster_ids = np.random.randint(spatial_clusters[0], spatial_clusters[1], size=spatial_samples)
    
    data['cell_cluster'] = cluster_ids
    
    # assign spatial spots to cluster_ids
    coordinate = np.zeros((cluster_ids.shape[0], 2))
    gene_data = np.zeros((cluster_ids.shape[0], gene_feat))
    img_data = np.zeros((cluster_ids.shape[0], img_feat))
    
    Z_a = np.zeros([gene_latent, spatial_samples])
    Z_b = np.zeros([img_latent, spatial_samples])
    mus = []
    for spa_clus in list(set(cluster_ids)):
        idx = (cluster_ids == spa_clus)
        range_x, range_y = regions[spa_clus - spatial_clusters[0]]
        sx = np.random.uniform(range_x[0], range_x[1], size=idx.sum())
        sy = np.random.uniform(range_y[0], range_y[1], size=idx.sum())
        coordinate[idx, 0] = sx
        coordinate[idx, 1] = sy        
        # generate gene feats
        # generate img feats
        cluster_mu1 = np.random.random([gene_latent]) - 0.5
#         cluster_mu2 = np.random.random([img_latent]) - 0.5
        cluster_mu2 = cluster_mu1
        mus.append((cluster_mu1, cluster_mu2))
        Z_a[:, idx] = np.random.multivariate_normal(mean=cluster_mu1, cov=0.1*np.eye(gene_latent), size=idx.sum()).transpose()
        Z_b[:, idx] = np.random.multivariate_normal(mean=cluster_mu2, cov=0.1*np.eye(img_latent), size=idx.sum()).transpose()
    
    data['sx'] = coordinate[:, 0]
    data['sy'] = coordinate[:, 1]

    X_a, Y_a = gen_feats(spatial_samples, Z_a, gene_feat, gene_latent, sigma_gene, dropout_gene)
    data['gene_data_full'] = X_a
    data['gene_data_dropout'] = Y_a
    
    X_b, Y_b = gen_feats(spatial_samples, Z_b, img_feat, img_latent, sigma_img, dropout_img)
    data['img_data_full'] = X_b
    data['img_data_dropout'] = Y_b
    
    
    # draw spatial spots
    colors = []
    my_cmap = sns.color_palette('muted', as_cmap=True)
    for cid in cluster_ids:
        colors.append(my_cmap[cid])
    plt.scatter(coordinate[:, 0], coordinate[:, 1], c=colors)
    
    return data, mus

def simulate_other_cell_type(
    spatial_clusters,
    spatial_samples,
    regions,
    gene_feat=500,
    img_feat = 500,
    gene_latent=30,
    img_latent=30,
    sigma_gene=0.1,
    sigma_img=0.1,
    dropout_gene=0.5,
    dropout_img=0.1,
    seed=123,
    use_prior=None,
):
    np.random.seed(seed)
    data = {}
    cluster_ids = np.random.randint(spatial_clusters[0], spatial_clusters[1], size=spatial_samples)
    
    data['cell_cluster'] = cluster_ids
    
    # assign spatial spots to cluster_ids
    coordinate = np.zeros((cluster_ids.shape[0], 2))
    gene_data = np.zeros((cluster_ids.shape[0], gene_feat))
    img_data = np.zeros((cluster_ids.shape[0], img_feat))
    
    Z_a = np.zeros([gene_latent, spatial_samples])
    Z_b = np.zeros([img_latent, spatial_samples])
#     mus = []
    for spa_clus in list(set(cluster_ids)):
        idx = (cluster_ids == spa_clus)
        range_x, range_y = regions[spa_clus - spatial_clusters[0]]
        sx = np.random.uniform(range_x[0], range_x[1], size=idx.sum())
        sy = np.random.uniform(range_y[0], range_y[1], size=idx.sum())
        coordinate[idx, 0] = sx
        coordinate[idx, 1] = sy        
        # generate gene feats
        # generate img feats
#         cluster_mu1 = np.random.random([gene_latent]) - 0.5
#         cluster_mu2 = np.random.random([img_latent]) - 0.5
#         mus.append((cluster_mu1, cluster_mu2))
        cluster_mu1, cluster_mu2 = use_prior[spa_clus - spatial_clusters[0]]
        Z_a[:, idx] = np.random.multivariate_normal(mean=cluster_mu1, cov=0.1*np.eye(gene_latent), size=idx.sum()).transpose()
        Z_b[:, idx] = np.random.multivariate_normal(mean=cluster_mu2, cov=0.1*np.eye(img_latent), size=idx.sum()).transpose()
    
    data['sx'] = coordinate[:, 0]
    data['sy'] = coordinate[:, 1]

    X_a, Y_a = gen_feats(spatial_samples, Z_a, gene_feat, gene_latent, sigma_gene, dropout_gene)
    data['gene_data_full'] = X_a
    data['gene_data_dropout'] = Y_a
    
    X_b, Y_b = gen_feats(spatial_samples, Z_b, img_feat, img_latent, sigma_img, dropout_img)
    data['img_data_full'] = X_b
    data['img_data_dropout'] = Y_b
    
    
    # draw spatial spots
    colors = []
    my_cmap = sns.color_palette('muted', as_cmap=True)
    for cid in cluster_ids:
        colors.append(my_cmap[cid])
    plt.scatter(coordinate[:, 0], coordinate[:, 1], c=colors)
    
    return data

def combine(spatial_data, mix_data, spatial_regions):
    # assign cluster to mix_data
    combine_data = {}
    
    for k, v in spatial_data.items():
        v2 = mix_data[k]
        v3 = np.concatenate([v, v2])
        combine_data[k] = v3
    
    combine_data['domain_cluster'] = []
    for (sx, sy) in zip(combine_data['sx'], combine_data['sy']):
        for idx, (rx, ry) in enumerate(spatial_regions):
            if sx >= rx[0] and sx <= rx[1] and sy >= ry[0] and sy <= ry[1]:
                combine_data['domain_cluster'].append(idx)
    
    return combine_data

def save_data_pickle(data, save_name):
    with open(save_name, 'wb') as f:
        pickle.dump(data, f)

def run_pca(root_dir, name, label_type='domain_cluster', save_path='simulation_results/1.csv'):
    pkl_file = os.path.join(root_dir, name+'.pkl')
    csv_file = os.path.join(root_dir, name+'.csv')
    adata_file = os.path.join(root_dir, name+'.h5ad')
    with open(pkl_file, 'rb') as f:
        data = pickle.load(f)
    
    data_a = data['gene_data_dropout']
    data_b = data['img_data_dropout']
    label_true = data[label_type]
    
    df = pd.DataFrame()
    df['label_domain'] = data['domain_cluster']
    df['label_cell'] = data['cell_cluster']
    
    gene_feat = PCA(n_components=30).fit_transform(data_a)
    img_feat = PCA(n_components=30).fit_transform(data_b)
    
    feat_df = pd.DataFrame(gene_feat)
    feat_df.to_csv('simulation_results/gene_emb_%s.csv'%(str(name)))
    
    feat_df = pd.DataFrame(img_feat)
    feat_df.to_csv('simulation_results/img_emb_%s.csv'%(str(name)))
    
    my_cmap = sns.color_palette('muted', as_cmap=True)
    colors = []
    for cid in data['domain_cluster']:
        colors.append(my_cmap[cid])
    size = 10
     # gene
    pred1, _, _ = phenograph.cluster(gene_feat)
    plt.figure(figsize=(17, 5))
    plt.subplot(1, 3, 1)
    X_embedded = TSNE(n_components=2).fit_transform(gene_feat)
    plt.scatter(X_embedded[:, 0], X_embedded[:, 1], c=colors, s=size)
    plt.title('gene, ARI = %01.3f' % adjusted_rand_score(label_true, pred1))

#     for i in np.unique(label_true):
#         idx = np.nonzero(label_true == i)[0]
#         plt.scatter(X_embedded[idx, 0], X_embedded[idx, 1])
#         plt.title('gene, ARI = %01.3f' % adjusted_rand_score(label_true, pred1))
    ## img
    pred2, _, _ = phenograph.cluster(img_feat)

#     plt.figure(figsize=(17, 5))

    plt.subplot(1, 3, 2)
    X_embedded = TSNE(n_components=2).fit_transform(img_feat)
    plt.scatter(X_embedded[:, 0], X_embedded[:, 1], c=colors, s=size)
    plt.title('img, ARI = %01.3f' % adjusted_rand_score(label_true, pred2))


    df['gene_pca'] = pred1
    df['img_pca'] = pred2
    df.to_csv(save_path)
    
#     for i in np.unique(label_true):
#         idx = np.nonzero(label_true == i)[0]
#         plt.scatter(X_embedded[idx, 0], X_embedded[idx, 1])
#         plt.title('img, ARI = %01.3f' % adjusted_rand_score(label_true, pred2))

def draw_spatial(data, cluster='cell_cluster'):
    colors = []
    my_cmap = sns.color_palette('muted', as_cmap=True)
    for cid in data[cluster]:
        colors.append(my_cmap[cid])
    sc = plt.scatter(data['sx'], data['sy'], c=colors, s=5, cmap=my_cmap)
#     plt.colorbar(sc)

def run_pca_feat(data, label_type='cell_cluster'):
    data_a = data['gene_data_dropout']
    data_b = data['img_data_dropout']
    label_true = data[label_type]
    
    gene_feat = PCA(n_components=30).fit_transform(data_a)
    img_feat = PCA(n_components=30).fit_transform(data_b)
    
    my_cmap = sns.color_palette('muted', as_cmap=True)
    
    colors = []
    for cid in data[label_type]:
        colors.append(my_cmap[cid])
        
     # gene
    pred1, _, _ = phenograph.cluster(gene_feat)
    plt.figure(figsize=(17, 5))
    plt.subplot(1, 3, 1)
    X_embedded = TSNE(n_components=2).fit_transform(gene_feat)
    plt.scatter(X_embedded[:, 0], X_embedded[:, 1], c=colors)
    plt.title('gene, ARI = %01.3f' % adjusted_rand_score(label_true, pred1))

#     for i in np.unique(label_true):
#         idx = np.nonzero(label_true == i)[0]
#         plt.scatter(X_embedded[idx, 0], X_embedded[idx, 1])
#         plt.title('gene, ARI = %01.3f' % adjusted_rand_score(label_true, pred1))
    ## img
    pred2, _, _ = phenograph.cluster(img_feat)

#     plt.figure(figsize=(17, 5))
    
    plt.subplot(1, 3, 2)
    X_embedded = TSNE(n_components=2).fit_transform(img_feat)
    plt.scatter(X_embedded[:, 0], X_embedded[:, 1], c=colors)
    plt.title('img, ARI = %01.3f' % adjusted_rand_score(label_true, pred2))

#     for i in np.unique(label_true):
#         idx = np.nonzero(label_true == i)[0]
#         plt.scatter(X_embedded[idx, 0], X_embedded[idx, 1])
#         plt.title('img, ARI = %01.3f' % adjusted_rand_score(label_true, pred2))
def gen_data(
    spatial_cluster=(0,4), mixed_cluster=(4,9),
    spatial_samples = 1000, ratio=0.3,
    spa_regions = [((10, 240),(10, 240)),((10, 240),(260, 490)),((260,490),(10, 240)),((260,490),(260,490))],
    mixed_regions = [((10, 240), (10, 240)),((10, 240), (10, 240)),((260, 490), (10, 240)),((260, 490), (10, 240)),((10, 240), (260, 490))],
    prior = [[0, 0.1], [0.1, 0.5], [0, 0.1], [0.1, 0.5], [0.1, 0.1]],
    add_noise_area = [0,0,2,2,1],
    gene_feat=500, img_feat=500, gene_latent=30, img_latent=30,
    sigma_gene=0.5, sigma_img=0.5, dropout_gene=0.5, dropout_img=0.5,
    noise_sigma_gene=0.5, noise_sigma_img=0.5, noise_dropout_gene=0.5, noise_dropout_img=0.5,
    seed=123, name=1, 
):
    np.random.seed(seed)
    spatial_data, mus = simulate_dominate_cell_type(spatial_cluster, spatial_samples, spa_regions,
                        gene_feat=gene_feat, img_feat=img_feat, gene_latent=gene_latent, img_latent=img_latent,                        
                        sigma_gene=sigma_gene, sigma_img=sigma_img, dropout_gene=dropout_gene, dropout_img=dropout_img)
    

    run_pca_feat(spatial_data, label_type='cell_cluster')
    mixed_samples = int(spatial_samples * ratio)
    mu_prior = [
        (mus[add_noise_area[i]][0] + prior[i][0], mus[add_noise_area[i]][1] + prior[i][1]) for i in range(len(prior))
    ]

    noise_data = simulate_other_cell_type(
        mixed_cluster, mixed_samples, mixed_regions, use_prior=mu_prior,
        sigma_gene=noise_sigma_gene, sigma_img=noise_sigma_img,
        dropout_gene=noise_dropout_gene, dropout_img=noise_dropout_img)
    
    run_pca_feat(noise_data, label_type='cell_cluster')
    combine_data = combine(spatial_data, noise_data, spa_regions)
    if not os.path.exists('simulation'):
        os.mkdir('simulation')

    save_name = 'simulation/spatial%s.pkl'%(name)
    save_data_pickle(combine_data, save_name)
    
    csv_save = 'simulation_results/spatial%s.csv'%(name)
    if not os.path.exists('simulation_results'):
        os.mkdir('simulation_results')
        
    run_pca('simulation/', 'spatial%s'%(name), label_type='domain_cluster', save_path=csv_save)

def save_method_pred(csv_path, pred, method_name):
    df = pd.read_csv(csv_path, header=0, index_col=0)
    df[method_name] = pred
    df.to_csv(csv_path)

if __name__ == '__main__':
    spatial_cluster = (0,4)
    noise_cluster = (4,9)
    spatial_samples = 1000
    ratio = 0.3
    spa_regions = [
        ((10, 240),(10, 240)),
        ((10, 240),(260, 490)),
        ((260,490),(10, 240)),
        ((260,490),(260,490))
    ]
    mix_regions = [
        ((10, 240), (10, 240)),
        ((10, 240), (10, 240)),
        ((260, 490), (10, 240)),
        ((260, 490), (10, 240)),
        ((10, 240), (260, 490))
    ]
    add_noise_area = [0,0,2,2,1]
    gene_feat=500
    img_feat=500
    gene_latent=30
    img_latent=30
    sigma_gene=0.5
    sigma_img=0.5
    dropout_gene=0.4
    dropout_img=0.4
    noise_sigma_gene=0.5
    noise_sigma_img=0.5
    noise_dropout_gene=0.3
    noise_dropout_img=0.3
    # seed=123
    name=1

    seeds = [i for i in range(10)]
    for seed in seeds:
        name = str(seed) + '_' + str(dropout_gene)
        
        # prior = [[0, 0.1], [0.1, 0.5], [0, 0.1], [0.1, 0.5], [0.1, 0.1]]
        prior = []
        for i in range(noise_cluster[1] - noise_cluster[0]):
            if i % 2 == 0:
                # generate offset with relative large distance
                offset1, offset2 = np.random.uniform(0, 0.1, size=1), np.random.uniform(0.4, 0.5, size=1)
            else:
                # generate offset with relative small distance
                offset1, offset2 = np.random.uniform(0, 0.1, size=1), np.random.uniform(0, 0.1, size=1)
            prior.append([offset1, offset2])

        np.random.seed(seed)
        gen_data(
            spatial_cluster=spatial_cluster, 
            mixed_cluster=noise_cluster,
            spatial_samples = spatial_samples,
            ratio=ratio,
            spa_regions = spa_regions,
            mixed_regions = mix_regions,
            prior = prior,
            add_noise_area = add_noise_area,
            gene_feat=gene_feat, 
            img_feat=img_feat, 
            gene_latent=gene_latent, 
            img_latent=img_latent,
            sigma_gene=sigma_gene, 
            sigma_img=sigma_img, 
            dropout_gene=dropout_gene, 
            dropout_img=dropout_img,
            noise_sigma_gene=noise_sigma_gene, 
            noise_sigma_img=noise_sigma_img, 
            noise_dropout_gene=noise_dropout_gene, 
            noise_dropout_img=noise_dropout_img,
            seed=seed, 
            name=name
        )
        np.random.seed(seed)
        sigra_pred, sigra_emb = run_sigra_model('simulation/', 'spatial%s'%(name), n_epochs=800, hidden_dims=[512, 30], seed=seed, lambda_1=0.5, lambda_2=0.5, lambda_3=1.0, graph_type='spatial', gt_type='domain')
        emb_df = pd.DataFrame(sigra_emb)
        emb_df.to_csv('simulation_results/sigra_emb_spatial%s.csv'%(str(name)))
        save_method_pred('simulation_results/spatial%s.csv'%(str(name)), sigra_pred, 'sigra')

