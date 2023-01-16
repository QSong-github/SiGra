from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import sklearn.neighbors
import sklearn
import phenograph
from sklearn.metrics.cluster import adjusted_rand_score
import numpy as np
import random
from copy import deepcopy
import pickle
import os

import seaborn as sns


import argparse
import pandas as pd
import os
import scanpy as sc
from sklearn.metrics.cluster import adjusted_rand_score
# from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
import numpy as np
import cv2
import torchvision.transforms as transforms
from utils import Cal_Spatial_Net, Stats_Spatial_Net, seed_everything
# from train_transformer import train_img, test_img

from transModel import TransImg
import torch
from scipy.optimize import linear_sum_assignment
import matplotlib.pyplot as plt

from torch_geometric.data import Data

import phenograph
from sklearn.manifold import TSNE
from tqdm import tqdm
import torch.nn.functional as F
import random
import sklearn
import seaborn as sns

os.environ['R_HOME'] = '/opt/R/4.0.2/lib/R'
os.environ['R_USER'] = '~/anaconda3/lib/python3.8/site-packages/rpy2'
os.environ['LD_LIBRARY_PATH'] = '/opt/R/4.0.2/lib/R/lib'
os.environ['PYTHONHASHSEED'] = '1234'
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'



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
    lambda_1=1, lambda_2=1, lambda_3=1, graph_type='spatial', gt_type='domain', vis=False):
    
    print(name)
    
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    data_dict, data, img, label_domain, label_cell, colors = transfer_data(root_dir, name, graph_type=graph_type)
    if gt_type == 'domain':
        label_true = label_domain
    else:
        label_true = label_cell
    
    np.random.seed(seed)
    gene_data = PCA(n_components=100).fit_transform(data_dict['gene_data_dropout'])
    
    np.random.seed(seed)
    img_data = PCA(n_components=100).fit_transform(data_dict['img_data_dropout'])
    
    my_cmap = sns.color_palette('muted', as_cmap=True)
    pred, _, _ = phenograph.cluster(gene_data)
    if vis:
        fig, axs = plt.subplots(1,3, figsize=(15,5))
        size = 10
        ax = axs[0]
        X_embedded = TSNE(n_components=2).fit_transform(gene_data)
        ax.scatter(X_embedded[:, 0], X_embedded[:, 1], c=colors, s=size)
        ax.title.set_text('gene, ARI = %01.3f' % adjusted_rand_score(label_domain, pred))

    pred, _, _ = phenograph.cluster(img_data)
    if vis:
        ax = axs[1]
        X_embedded = TSNE(n_components=2).fit_transform(img_data)
        ax.scatter(X_embedded[:, 0], X_embedded[:, 1], c=colors, s=size)
        ax.title.set_text('img, ARI = %01.3f' % adjusted_rand_score(label_domain, pred))

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
    if vis:
        plt.subplot(1, 3, 3)
        ax = axs[2]
        X_embedded = TSNE(n_components=2).fit_transform(cz)
        ax.scatter(X_embedded[:, 0], X_embedded[:, 1], c=colors, s=size)
        ax.title.set_text('Sigra, ARI = %01.3f' % adjusted_rand_score(label_domain, sigra_pred))

    ari = adjusted_rand_score(label_domain, sigra_pred)
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

