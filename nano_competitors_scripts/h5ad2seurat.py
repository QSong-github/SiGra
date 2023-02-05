import numpy as np
import pandas as pd
import scanpy as sc
import diopy

adata = sc.read('../dataset/nanostring_Lung5_Rep1/fov3/sampledata.h5ad')
diopy.output.write_h5(adata, file='../dataset/nanostring_Lung5_Rep1/fov3/sampledata.h5seurat')