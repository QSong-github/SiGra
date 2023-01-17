import cv2
from PIL import Image
import scanpy as sc
import matplotlib.pyplot as plt
import os
import anndata as ad
import numpy as np
from sklearn.metrics.cluster import adjusted_rand_score
from scipy.optimize import linear_sum_assignment
import scanpy as sc

adata = sc.read('domain_ari/ssam_adata.h5ad')

adata.obs

import pandas as pd

csv = pd.read_csv('lung9-1.csv', index_col=0, header=0)

csv

csv['pathology'] = adata.obs['domain_gt']

ari = adjusted_rand_score(csv['pathology'], csv['ssam_pred'])
print(ari)

ari = adjusted_rand_score(csv['pathology'], csv['ssam_gt'])
print(ari)


