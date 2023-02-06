import pandas as pd
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score, \
                            homogeneity_completeness_v_measure
from sklearn.metrics.cluster import contingency_matrix
from sklearn.preprocessing import LabelEncoder
import numpy as np
import scanpy as sc
import stlearn as st
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
import sys
import matplotlib.pyplot as plt
import os
import argparse
import cv2

def run_stlearn(sample, imname, BASE_PATH, save_path):
    TILE_PATH = Path("/tmp/{}_tiles".format(sample))
    TILE_PATH.mkdir(parents=True, exist_ok=True)

    OUTPUT_PATH = Path("%s/%s"%(save_path, sample))
    OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

    # data = st.Read10X(BASE_PATH / sample)

    # ground_truth_df = pd.read_csv( BASE_PATH / sample / 'metadata.tsv', sep='\t')
    # ground_truth_df['ground_truth'] = ground_truth_df['layer_guess']
    data = sc.read(os.path.join(BASE_PATH, sample, 'sampledata.h5ad'))
    ground_truth_df = pd.DataFrame()
    ground_truth_df['ground_truth'] = data.obs['merge_cell_type']

    img = cv2.imread(os.path.join(BASE_PATH, sample, 'CellComposite_%s.jpg'%(imname)))
    data.uns['spatial'] = {id: {'images': {'hires': img/255.0}, 'use_quality': 'hires'}}
    data.obs['imagerow'] = data.obs['cx']
    data.obs['imagecol'] = data.obs['cy']
    data.obs['array_row'] = data.obs['cx']
    data.obs['array_col'] = data.obs['cy']


    le = LabelEncoder()
    ground_truth_le = le.fit_transform(list(ground_truth_df['ground_truth'].values))

    # n_cluster = len((set(ground_truth_df['ground_truth']))) - 1
    n_cluster = 3
    data.obs['ground_truth'] = data.obs['merge_cell_type']

    ground_truth_df["ground_truth_le"] = ground_truth_le 
    st.pp.filter_genes(data,min_cells=1)
    st.pp.normalize_total(data)
    st.pp.log1p(data)
    st.em.run_pca(data,n_comps=15)
    st.pp.tiling(data, TILE_PATH)
    st.pp.extract_feature(data)
    
    st.spatial.SME.SME_normalize(data, use_data="raw", weights="physical_distance")
    data_ = data.copy()
    data_.X = data_.obsm['raw_SME_normalized']
    st.pp.scale(data_)
    st.em.run_pca(data_,n_comps=30)
    st.tl.clustering.kmeans(data_, n_clusters=n_cluster, use_data="X_pca", key_added="X_pca_kmeans")
    
    df = data_.obs.dropna()

    ari = adjusted_rand_score(df['X_pca_kmeans'], df['ground_truth'])
    ari = round(ari, 2)
    print(ari)
    return ari, data_

def calculate_clustering_matrix(pred, gt, sample, methods_):
    df = pd.DataFrame(columns=['Sample', 'Score', 'PCA_or_UMAP', 'Method', "test"])

    pca_ari = adjusted_rand_score(pred, gt)
    df1 = pd.Series([sample, pca_ari, "pca", methods_, "Adjusted_Rand_Score"],
                             index=['Sample', 'Score', 'PCA_or_UMAP', 'Method', "test"])
    pd.concat((df, df1), axis=0)
    # df = df.append(), ignore_index=True)


    pca_nmi = normalized_mutual_info_score(pred, gt)
    df1 = pd.Series([sample, pca_nmi, "pca", methods_, "Normalized_Mutual_Info_Score"],
                             index=['Sample', 'Score', 'PCA_or_UMAP', 'Method', "test"])
    # df = df.append(), ignore_index=True)
    pd.concat((df, df1), axis=0)


    pca_purity = purity_score(pred, gt)
    df1 = pd.Series([sample, pca_purity, "pca", methods_, "Purity_Score"],
                             index=['Sample', 'Score', 'PCA_or_UMAP', 'Method', "test"])
    # df = df.append(), ignore_index=True)
    pd.concat((df, df1), axis=0)


    pca_homogeneity, pca_completeness, pca_v_measure = homogeneity_completeness_v_measure(pred, gt)
    # df = df.append(), ignore_index=True)
    df1 = pd.Series([sample, pca_homogeneity, "pca", methods_, "Homogeneity_Score"],
                             index=['Sample', 'Score', 'PCA_or_UMAP', 'Method', "test"])
    pd.concat((df, df1), axis=0)


    # df = df.append(), ignore_index=True)
    df1 = pd.Series([sample, pca_completeness, "pca", methods_, "Completeness_Score"],
                             index=['Sample', 'Score', 'PCA_or_UMAP', 'Method', "test"])
    pd.concat((df, df1), axis=0)


    # df = df.append(), ignore_index=True)
    df1 = pd.Series([sample, pca_v_measure, "pca", methods_, "V_Measure_Score"],
                             index=['Sample', 'Score', 'PCA_or_UMAP', 'Method', "test"])
    pd.concat((df, df1), axis=0)

    return df

def purity_score(y_true, y_pred):
    # compute contingency matrix (also called confusion matrix)
    cm = contingency_matrix(y_true, y_pred)
    # return purity
    return np.sum(np.amax(cm, axis=0)) / np.sum(cm)


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

    BASE_PATH = Path(opt.root)
    samples = ['fov%d'%(i) for i in range(1, opt.num_fov+1)]
    imnames = ['F0%02d'%(i) for i in range(1, opt.num_fov+1)]
    for s, imname in zip(samples, imnames):
        ari, adata = run_stlearn(s, imname, BASE_PATH, opt.save_path)
        
        ind = adata.obs['X_pca_kmeans'].isna()
        adata = adata[~ind]

        df = pd.DataFrame(adata.obsm['raw_SME_normalized'], index=adata.obs.index)
        df.to_csv(os.path.join(opt.save_path,s,'%s_emb.csv'%s))

        df = pd.DataFrame(index=adata.obs.index)
        df['cluster'] = adata.obs['X_pca_kmeans']
        df['merge_cell_type'] = adata.obs['merge_cell_type']
        df.to_csv(os.path.join(opt.save_path,s,'%s_cluster.csv'%s))