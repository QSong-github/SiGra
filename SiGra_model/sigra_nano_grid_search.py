import argparse
import pandas as pd
import os
import scanpy as sc
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
import numpy as np
import cv2
import torchvision.transforms as transforms
from utils import Cal_Spatial_Net, Stats_Spatial_Net, _hungarian_match, seed_everything
# from train_transformer import train_nano_fov, test_nano_fov
from sklearn.decomposition import PCA
import torch
import matplotlib.pyplot as plt
import random
from scipy.optimize import linear_sum_assignment
from utils import Transfer_img_Data, seed_everything, mclust_R
from torch_geometric.loader import NeighborLoader, NeighborSampler, DataLoader
from transModel import TransImg
import torch.nn.functional as F
import scipy.sparse as sp

os.environ['R_HOME'] = '/opt/R/4.0.2/lib/R'
os.environ['R_USER'] = '~/anaconda3/lib/python3.8/site-packages/rpy2'
os.environ['LD_LIBRARY_PATH'] = '/opt/R/4.0.2/lib/R/lib'
os.environ['PYTHONHASHSEED'] = '1234'
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

def gen_adatas(opt, root, id, img_name):
    adata = sc.read(os.path.join(root, id, 'sampledata.h5ad'))
    adata.var_names_make_unique()
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    ncluster = len(set(adata.obs['merge_cell_type']))

    print(os.path.join(root, id, 'CellComposite_%s.jpg'%(img_name)))
    img = cv2.imread(os.path.join(root, id, 'CellComposite_%s.jpg'%(img_name)))
    height, width, c = img.shape
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    overlay = cv2.imread(os.path.join(root, id, 'CompartmentLabels_%s.tif'%(img_name)))
    overlay = cv2.cvtColor(overlay, cv2.COLOR_BGR2GRAY)
    print(overlay.shape)

    # if opt.use_gray:
    #     img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    transform = transforms.ToTensor()
    img = transform(img)
    overlay = transform(overlay)

    patchs = []
    w, h = 60, 60
    
    for coor in adata.obsm['spatial']:
        x, y = coor
        img_p = img[:, int(y-h):int(y+h), int(x-w): int(x+w)]

        patchs.append(img_p.flatten()) # 4 * h * w
    patchs = np.stack(patchs)


    df = pd.DataFrame(patchs, index=adata.obs.index)
    adata.obsm['imgs'] = df


    Cal_Spatial_Net(adata, rad_cutoff=80)
    Stats_Spatial_Net(adata)
    return adata

@torch.no_grad()
def test_nano_fov(opt, adatas, model_name=None, hidden_dims=[512, 30], n_epochs=1000, lr=0.001, key_added='STAGATE',
                gradient_clipping=5.,  weight_decay=0.0001, verbose=True, 
                random_seed=0, save_loss=False, save_reconstrction=False, 
                device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'),
                save_path='../checkpoint/trans_gene/', ncluster=7, repeat=1,
                g_weight=0.1, i_weight=0.1, c_weight=1.0):
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
    
    datas, imgs = [], []
    gt_frame = None
    datas, gene_dims = [], []
    gene_dim = 0
    img_dim = 0 # gene and img dim is same for all fovs

    # [fov1(feat, graph), fov2(feat, graph), fov3, ... fov20]
    # 

    for adata in adatas:
        adata.X = sp.csr_matrix(adata.X)
        data, img = Transfer_img_Data(adata)
        gene_dim = data.x.shape[1]
        img_dim = img.x.shape[1]
        data.x = torch.cat([data.x, img.x], dim=1)

        datas.append(data)

    import anndata
    adata = anndata.concat(adatas)

    loader = DataLoader(datas, batch_size=1, shuffle=False)
    model = TransImg(hidden_dims=[gene_dim, img_dim] + hidden_dims).to(device)
    if model_name is not None:
        model.load_state_dict(torch.load(os.path.join(save_path, model_name)))
    else:
        print(os.path.join(save_path, opt.pretrain))
        model.load_state_dict(torch.load(os.path.join(save_path, opt.pretrain)))

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

    hidden_matrix = None
    gene_matrix = None
    img_matrix = None
    couts = None
    losses = 0
    count = 0
    for batch in loader:
        batch = batch.to(device)
        # print(batch)
        bgene = batch.x[:, :gene_dim]
        bimg = batch.x[:, gene_dim:]
        # exit(0)
        edge_index = batch.edge_index
        gz,iz,cz, gout,iout,cout = model(bgene, bimg, edge_index)

        gloss = F.mse_loss(bgene, gout)
        iloss = F.mse_loss(bgene, iout)
        closs = F.mse_loss(bgene, cout)
        loss = (g_weight * gloss + i_weight * iloss + c_weight * closs) / (g_weight + i_weight + c_weight)
        losses += loss.item()
        count += 1

        print(cz.shape)
        if hidden_matrix is None:
            hidden_matrix = cz.detach().cpu()
            gene_matrix = gz.detach().cpu()
            couts = cout.detach().cpu()
            img_matrix = iz.detach().cpu()
        else:
            hidden_matrix = torch.cat([hidden_matrix, cz.detach().cpu()], dim=0)
            gene_matrix = torch.cat([gene_matrix, gz.detach().cpu()], dim=0)
            img_matrix = torch.cat([img_matrix, iz.detach().cpu()], dim=0)
            couts = torch.cat([couts, cout.detach().cpu()], dim=0)
    losses = losses.item() / count
    # exit(0)
    hidden_matrix = hidden_matrix.numpy()
    gene_matrix = gene_matrix.numpy()
    img_matrix = img_matrix.numpy()
    adata.obsm['pred'] = hidden_matrix
    adata.obsm['gene_pred'] = gene_matrix
    adata.obsm['img_pred'] = img_matrix
    couts = couts.numpy().astype(np.float32)
    couts[couts < 0] = 0
    adata.layers['recon'] = couts
    return adata, losses

def train_nano_fov(opt, adatas, hidden_dims=[512, 30], n_epochs=1000, lr=0.001, key_added='STAGATE',
                gradient_clipping=5.,  weight_decay=0.0001, verbose=True, 
                random_seed=0, save_loss=False, save_reconstrction=False, 
                device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'),
                save_path='../checkpoint/trans_gene/', ncluster=7, repeat=1,
                gene_weight=0.1, img_weight=0.1, combine_weight=1.0):
    # seed_everything(random_seed)
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
    
    datas, gene_dims = [], []
    gene_dim = 0
    img_dim = 0 # gene and img dim is same for all fovs
    for adata in adatas:
        adata.X = sp.csr_matrix(adata.X)
        data, img = Transfer_img_Data(adata)
        gene_dim = data.x.shape[1]
        img_dim = img.x.shape[1]
        data.x = torch.cat([data.x, img.x], dim=1)
        datas.append(data)
    loader = DataLoader(datas, batch_size=1, shuffle=True)

    model = TransImg(hidden_dims=[gene_dim, img_dim] + hidden_dims).to(device)
    torch.save(model.state_dict(), os.path.join(save_path, 'init.pth'))
    # data = data.to(device)
    # img = img.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

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
    
    for epoch in tqdm(range(1, n_epochs+1)):
        # for data, img in zip(datas, imgs):
        for i, batch in enumerate(loader):
            model.train()
            batch = batch.to(device)
            optimizer.zero_grad()
            bgene = batch.x[:, :gene_dim]
            bimg = batch.x[:, gene_dim:]
            edge_index = batch.edge_index
            gz,iz,cz, gout,iout,cout = model(bgene, bimg, edge_index)

            gloss = F.mse_loss(bgene, gout)
            iloss = F.mse_loss(bgene, iout)
            closs = F.mse_loss(bgene, cout)
            loss = gene_weight * gloss + img_weight * iloss + combine_weight * closs
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)
            optimizer.step()
        
        if epoch > 500 and epoch % 100 == 0:
            torch.save(model.state_dict(), os.path.join(save_path, 'final_%d_%d.pth'%(epoch, repeat)))

    torch.save(model.state_dict(), os.path.join(save_path, 'final_%d.pth'%(repeat)))
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            bgene = batch.x[:, :gene_dim]
            bimg = batch.x[:, gene_dim:]
            edge_index = batch.edge_index
            gz,iz,cz, gout,iout,cout = model(bgene, bimg, edge_index)

            gloss = F.mse_loss(bgene, gout)
            iloss = F.mse_loss(bgene, iout)
            closs = F.mse_loss(bgene, cout)
            loss = (gene_weight * gloss + img_weight * iloss + combine_weight * closs) / (gene_weight + img_weight + combine_weight)
            losses += loss.item()
            count += 1

            print(cz.shape)
            if hidden_matrix is None:
                hidden_matrix = cz.detach().cpu()
                gene_matrix = gz.detach().cpu()
                couts = cout.detach().cpu()
                img_matrix = iz.detach().cpu()
            else:
                hidden_matrix = torch.cat([hidden_matrix, cz.detach().cpu()], dim=0)
                gene_matrix = torch.cat([gene_matrix, gz.detach().cpu()], dim=0)
                img_matrix = torch.cat([img_matrix, iz.detach().cpu()], dim=0)
                couts = torch.cat([couts, cout.detach().cpu()], dim=0)
        losses = losses.item() / count
        hidden_matrix = hidden_matrix.numpy()
        gene_matrix = gene_matrix.numpy()
        img_matrix = img_matrix.numpy()
        adata.obsm['pred'] = hidden_matrix
        adata.obsm['gene_pred'] = gene_matrix
        adata.obsm['img_pred'] = img_matrix
        couts = couts.numpy().astype(np.float32)
        couts[couts < 0] = 0
        adata.layers['recon'] = couts
    return adata, losses

def train(opt, r=0):
    seed_everything(opt.seed)
    ids = ['fov%d'%(i) for i in range(1, opt.num_fov+1)]
    img_names = ['F0%02d'%(i) for i in range(1, opt.num_fov+1)]

    adatas = list()
    for id, name in zip(ids, img_names):
        adata = gen_adatas(opt.root, id, name)
        adatas.append(adata)

    sp = os.path.join(opt.save_path, 'all')
    if not os.path.exists(sp):
        os.makedirs(sp)

    adata, losses = train_nano_fov(opt, adatas, hidden_dims=[opt.h_dim1, opt.h_dim2],  n_epochs=opt.epochs, save_loss=True, 
                lr=opt.lr, random_seed=opt.seed, save_path=sp, ncluster=opt.ncluster, repeat=r)
    



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--root', type=str, default='../dataset/nanostring_Lung5_Rep1')
    parser.add_argument('--epochs', type=int, default=600)
    parser.add_argument('--id', type=str, default='lung5-1')
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--save_path', type=str, default='../checkpoint/sigra_nano_lung5-1')
    parser.add_argument('--ncluster', type=int, default=7)
    parser.add_argument('--repeat', type=int, default=0)
    parser.add_argument('--use_gray', type=float, default=0)
    parser.add_argument('--test_only', type=int, default=0)
    parser.add_argument('--pretrain', type=str, default='final_0.pth')
    parser.add_argument('--ratio', type=float, default=0.1)
    parser.add_argument('--g_weight', type=float, default=0.1)
    parser.add_argument('--i_weight', type=float, default=0.1)
    parser.add_argument('--c_weight', type=float, default=1)
    parser.add_argument('--use_combine', type=int, default=1)
    parser.add_argument('--use_img_loss', type=int, default=0)
    parser.add_argument('--h_dim1', type=int, default=64)
    parser.add_argument('--h_dim2', type=int, default=32)
    parser.add_argument('--cluster_method', type=str, default='leiden')
    parser.add_argument('--num_fov', type=int, default=30)
    opt = parser.parse_args()

    logger = open('../result/pruning_nano_parameter3.txt', 'a')
    logger.write('id: %s, h_dim1: %d, h_dim2: %d, '%(opt.id, opt.h_dim1, opt.h_dim2))

    if opt.test_only:
        ari_all, val_loss, ss, chs, dbs = infer(opt, umap_save_path='../results/nano/%s/figures'%(opt.id))
    else:
        if not os.path.exists(os.path.join(opt.save_path, opt.id)):
            os.makedirs(os.path.join(opt.save_path, opt.id))
        train(opt)
        ari_all, val_loss, ss, chs, dbs = infer(opt, umap_save_path='../results/nano/%s/figures'%(opt.id))

    logger.write('ari_all: %.4f, val_loss: %.4f, silhouette_score: %.4f, calinski_harabasz_score: %.4f, davies_bouldin_score: %.4f\n\n'%(ari_all, val_loss, ss, chs, dbs))

