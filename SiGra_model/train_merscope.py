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
from train_transformer import train_nano_fov, test_nano_fov_batch
from sklearn.decomposition import PCA
import torch
import matplotlib.pyplot as plt
import random
from scipy.optimize import linear_sum_assignment
from tqdm import tqdm

os.environ['R_HOME'] = '/opt/R/4.0.2/lib/R'
os.environ['R_USER'] = '~/anaconda3/lib/python3.8/site-packages/rpy2'
os.environ['LD_LIBRARY_PATH'] = '/opt/R/4.0.2/lib/R/lib'
os.environ['PYTHONHASHSEED'] = '1234'
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

def norm(img):
    return (img - img.min()) / (img.max() - img.min())

def resize(img, scale_precent=25):
    # downsample the image to fit into the model
    width = int(img.shape[1] * scale_precent / 100)
    height = int(img.shape[0] * scale_precent / 100)
    dim = (width, height)
    resized = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
    return resized


def read_processed_adatas(root, fov):
    data_root = os.path.join(root, 'processed_data', 'fov_%d.h5ad'%(fov))
    # print(data_root)
    adata = sc.read(data_root)
    return adata

def gen_adatas(root, fov):
    print(os.path.join(root, 'sample_data', 'fov_%d.h5ad'%(fov)))
    adata = sc.read(os.path.join(root, 'sample_data', 'fov_%d.h5ad'%(fov)))
    adata.var_names_make_unique()
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    names = adata.obs.index.tolist()

    imgs = []

    for name in names:
        # read gray scale images
        b1 = cv2.imread(os.path.join(root, 'cut_images', 'bound1', name+'_zIndex_3.tif'), 0)
        b2 = cv2.imread(os.path.join(root, 'cut_images', 'bound2', name+'_zIndex_3.tif'), 0)
        b3 = cv2.imread(os.path.join(root, 'cut_images', 'bound3', name+'_zIndex_3.tif'), 0)
        dapi = cv2.imread(os.path.join(root, 'cut_images', 'DAPI', name+'_zIndex_3.tif'), 0)
        # print(b1.shape, b2.shape, b3.shape, dapi.shape)

        # check size
        b1 = resize(b1[:200, :200])
        b2 = resize(b2[:200, :200])
        b3 = resize(b3[:200, :200])
        dapi = resize(dapi[:200, :200])

        b1 = torch.log1p(torch.from_numpy(b1))
        b2 = torch.log1p(torch.from_numpy(b2))
        b3 = torch.log1p(torch.from_numpy(b3))
        dapi = torch.log1p(torch.from_numpy(dapi))

        img = torch.stack([b1, b2, b3, dapi]) # 4 * h * w
        img = img.flatten() 
        imgs.append(img)

    imgs = torch.stack(imgs) # n * 4 * h * w
    adata.obsm['imgs'] = imgs.numpy()

    Cal_Spatial_Net(adata, rad_cutoff=150)
    Stats_Spatial_Net(adata)

    # save adata
    if not os.path.exists('../dataset/mouseLiver/processed_data'):
        os.makedirs('../dataset/mouseLiver/processed_data')
    
    sp = '../dataset/mouseLiver/processed_data'
    adata.write(os.path.join(sp, 'fov_%d.h5ad'%(fov)))

    return adata


def infer(opt):
    seed_everything(opt.seed)
    seed = opt.seed
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    adatas = list()
    if not opt.processed:
        for fov in tqdm(range(0,83)):
            adata = gen_adatas(opt.root, fov)
            adatas.append(adata)
    else:
        for fov in tqdm(range(0, 83)):
            adata = read_processed_adatas(opt.root, fov)
            adatas.append(adata)

    sp = os.path.join(opt.save_path, 'all')
    test_nano_fov_batch(opt, adatas, hidden_dims=[512, 30],  n_epochs=opt.epochs, save_loss=True, 
                lr=opt.lr, random_seed=opt.seed, save_path=sp, ncluster=opt.ncluster, repeat=0)

def main(opt):
    seed_everything(opt.seed)

    adatas = list()
    # we generate 83 fovs. Note some fovs had little cells (< 100)
    # we remove thoses fovs so that the total length is smaller than 100.
    if not opt.processed:
        for fov in tqdm(range(0,83)):
            adata = gen_adatas(opt.root, fov)
            adatas.append(adata)
    else:
        for fov in tqdm(range(0, 83)):
            adata = read_processed_adatas(opt.root, fov)
            adatas.append(adata)

    sp = os.path.join(opt.save_path, 'all')
    if not os.path.exists(sp):
        os.makedirs(sp)

    train_nano_fov(opt, adatas, hidden_dims=[512, 30], n_epochs=opt.epochs, 
                save_loss=True, lr=opt.lr, random_seed=opt.seed, save_path=sp,
                ncluster=opt.ncluster, repeat=0)


# if __name__ == '__main__':
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--lr', type=float, default=1e-3)
#     parser.add_argument('--root', type=str, default='../dataset/mouseLiver')
#     parser.add_argument('--epochs', type=int, default=1000)
#     parser.add_argument('--seed', type=int, default=1234)
#     parser.add_argument('--save_path', type=str, default='../checkpoint/merscope_all')
#     parser.add_argument('--ncluster', type=int, default=14)
#     parser.add_argument('--repeat', type=int, default=1)
#     parser.add_argument('--test_only', type=int, default=0)
#     parser.add_argument('--pretrain', type=str, default='final.pth')
#     parser.add_argument('--processed', type=int, default=1)
#     opt = parser.parse_args()

def train_mscope(opt):
    if opt.test_only:
        infer(opt)
    else:
        main(opt)



