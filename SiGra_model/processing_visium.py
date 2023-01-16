import scanpy as sc
import pandas as pd
import random
import numpy as np
import torch
import os
import cv2
import matplotlib.pyplot as plt
import scipy
import tqdm
from utils import prefilter_genes,prefilter_specialgenes

seed = 0
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

def precessing(dirs):
    # 1. read the dataset and mask the data
    adata = sc.read_visium(dirs, count_file='filtered_feature_bc_matrix.h5', load_images=True)
    adata.var_names_make_unique()
    # 2. filter genes first
    idx1, adata = prefilter_genes(adata, min_cells=3) # avoiding all genes are zeros
    idx2, adata = prefilter_specialgenes(adata)

    # 3. save the raw data as csv
    adata.write(os.path.join(dirs, 'sampledata.h5ad'))
    sample_csv = pd.DataFrame(adata.X.toarray(), index=adata.obs.index, columns=adata.var.index)
    sample_csv.to_csv(os.path.join(dirs, 'sample_data.csv'))


# if __name__ == '__main__':
def processing_10x():
    root = '../dataset/DLPFC'
    # sub_id = ['151507', '151508', '151509', '151510', 
    #         '151669', '151670', '151671', '151672',
    #         '151673', '151674', '151675', '151676']

    sub_id = ['151507']

    for id in tqdm.tqdm(sub_id):
        dirs = os.path.join(root, id)
        if not os.path.exists(dirs):
            os.makedirs(dirs)
        precessing(dirs)
