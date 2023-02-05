import numpy as np
import pandas as pd
from tqdm import tqdm
import scipy.sparse as sp
from torch_geometric.loader import NeighborLoader, NeighborSampler, DataLoader

from transModel import TransImg, TransImg2
from sklearn.metrics.cluster import adjusted_rand_score
# from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
# from torch_geometric.loader import NeighborLoader

from utils import Transfer_img_Data, seed_everything, mclust_R
from sklearn.decomposition import PCA

import torch
import torch.backends.cudnn as cudnn

cudnn.deterministic = True
cudnn.benchmark = False
import torch.nn.functional as F
import matplotlib.pyplot as plt
import scanpy as sc
import os
os.environ['PYTHONHASHSEED'] = '1234'
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
os.environ['LD_LIBRARY_PATH'] = '/usr/local/cuda-10.2/bin:$PATH'

def test_one_branch(adata, hidden_dims=[512, 30], n_epochs=1000, lr=0.001, key_added='STAGATE',
                gradient_clipping=5.,  weight_decay=0.0001, verbose=True, 
                random_seed=0, save_loss=False, save_reconstrction=False, 
                device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'),
                save_path='../checkpoint/trans_gene/', ncluster=7, repeat=1, use_img_loss=0, use_combine=1,
                X_train_idx=None, X_test_idx=None, lambda_1 = 1, lambda_2 = 1, lambda_3 = 1, branch='gene'):
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

    # adata.X = sp.csr_matrix(adata.X)
    if 'highly_variable' in adata.var.columns:
        adata_Vars =  adata[:, adata.var['highly_variable']]
    else:
        adata_Vars = adata
    data, img = Transfer_img_Data(adata_Vars)
    if branch == 'img_img':
        use_img_loss = 1
    else:
        use_img_loss = 0
    model = TransImg(hidden_dims=[data.x.shape[1], img.x.shape[1]] + hidden_dims, use_img_loss=use_img_loss).to(device)
    print(load_path)
    # model.load_state_dict(torch.load(os.path.join(save_path, 'final_%d.pth'%(repeat))))
    model.load_state_dict(torch.load(load_path))
    data = data.to(device)
    img = img.to(device)
    model.eval()
    gz,iz,cz, gout,iout,cout = model(data.x, img.x, data.edge_index)

    if branch == 'gene':
        predz = gz
        predout = gout
    elif branch == 'img_gene':
        predz = iz
        predout = iout
    elif branch == 'img_img':
        predz = iz
        predout = iout
    elif branch == 'combine_only':
        predz = cz
        predout = cout
    else:
        raise NotImplementedError()

    adata.obsm['pred'] = predz.clone().detach().cpu().numpy()
    output = predout.clone().detach().cpu().numpy()
    output[output < 0] = 0
    # adata.layers['recon'] = np.zeros((adata.shape[0], adata.shape[1]))
    # adata.layers['recon'][:, adata.var['highly_variable']] = output

    adata_Vars.obsm['imgs'] = adata_Vars.obsm['imgs'].to_numpy()
    # adata_Vars.write('./151676_var.h5ad')

    adata_recon = adata_Vars.copy()
    adata_recon.X = output
    # adata_recon.write('./151676_recon.h5ad')
    return adata    


# def test_img(adata, load_path, hidden_dims=[512, 30], key_added='pred', device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'),
#                 save_path='../checkpoint/trans_gene/', random_seed=0):
def test_img(adata, load_path, hidden_dims=[512, 30], key_added='STAGATE',
            verbose=True, random_seed=0, save_loss=False, save_reconstrction=False, 
            device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'),
            save_path='../checkpoint/trans_gene/', ncluster=7, repeat=1, use_img_loss=0, use_combine=1,
            X_train_idx=None, X_test_idx=None, lambda_1 = 1, lambda_2 = 1, lambda_3 = 1):
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

    # adata.X = sp.csr_matrix(adata.X)
    if 'highly_variable' in adata.var.columns:
        adata_Vars =  adata[:, adata.var['highly_variable']]
    else:
        adata_Vars = adata
    data, img = Transfer_img_Data(adata_Vars)
    data = data.to(device)
    img = img.to(device)
    # model = TransImg(hidden_dims=[data.x.shape[1], img.x.shape[1]] + hidden_dims).to(device)
    # print(load_path)
    # # model.load_state_dict(torch.load(os.path.join(save_path, 'final_%d.pth'%(repeat))))
    # model.load_state_dict(torch.load(load_path))
    # data = data.to(device)
    # img = img.to(device)
    # model.eval()
    # gz,iz,cz, gout,iout,cout = model(data.x, img.x, data.edge_index)
    # adata.obsm['pred'] = cz.clone().detach().cpu().numpy()
    # output = cout.clone().detach().cpu().numpy()
    # output[output < 0] = 0
    # adata.layers['recon'] = np.zeros((adata.shape[0], adata.shape[1]))
    # adata.layers['recon'][:, adata.var['highly_variable']] = output

    # adata_Vars.obsm['imgs'] = adata_Vars.obsm['imgs'].to_numpy()
    # adata_Vars.write('./151676_var.h5ad')

    # adata_recon = adata_Vars.copy()
    # adata_recon.X = output
    # adata_recon.write('./151676_recon.h5ad')
    # return adata

    model = TransImg(hidden_dims=[data.x.shape[1], img.x.shape[1]] + hidden_dims, use_img_loss=use_img_loss).to(device)

    with torch.no_grad():
        model.load_state_dict(torch.load(load_path))
        model.eval()
        gz,iz,cz, gout,iout,cout = model(data.x, img.x, data.edge_index)

        gloss = F.mse_loss(data.x, gout)
        if use_img_loss:
            iloss = F.mse_loss(img.x, iout)
        else:
            iloss = F.mse_loss(data.x, iout)
        
        if use_combine:
            closs = F.mse_loss(data.x, cout)
            loss = gloss * lambda_1 + iloss * lambda_2 + closs * lambda_3
            loss = loss / (lambda_1 + lambda_2 + lambda_3)
        else:
            loss = gloss + iloss



        if not os.path.exists(save_path):
            os.makedirs(save_path)
        adata_Vars.obsm['pred'] = cz.clone().detach().cpu().numpy()
        sc.pp.neighbors(adata_Vars, use_rep='pred')
        sc.tl.umap(adata_Vars)
        plt.rcParams["figure.figsize"] = (3, 3)
        sc.settings.figdir = save_path
        ax = sc.pl.umap(adata_Vars, color=['Ground Truth'], show=False, title='combined latent variables')
        plt.savefig(os.path.join(save_path, 'umap_final.pdf'), bbox_inches='tight')
    
        adata_Vars.obsm['pred'] = cz.to('cpu').detach().numpy().astype(np.float32)
        output = cout.to('cpu').detach().numpy().astype(np.float32)
        output[output < 0] = 0
        adata_Vars.layers['recon'] = output
    plt.close('all')
    return adata_Vars, loss.item()

@torch.no_grad()
def test_nano(opt, adata, hidden_dims=[512, 30], n_epochs=1000, lr=0.001, key_added='STAGATE',
                gradient_clipping=5.,  weight_decay=0.0001, verbose=True, 
                random_seed=0, save_loss=False, save_reconstrction=False, 
                device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'),
                save_path='../checkpoint/trans_gene/', ncluster=7, repeat=1, save_name=''):
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

    adata.X = sp.csr_matrix(adata.X)
    # adata.X = adata.X.A
    if 'highly_variable' in adata.var.columns:
        adata_Vars =  adata[:, adata.var['highly_variable']]
    else:
        adata_Vars = adata

    if verbose:
        print('Size of Input: ', adata_Vars.shape)
    if 'Spatial_Net' not in adata.uns.keys():
        raise ValueError("Spatial_Net is not existed! Run Cal_Spatial_Net first!")

    data, img = Transfer_img_Data(adata_Vars)
    model = TransImg(hidden_dims=[data.x.shape[1], img.x.shape[1]] + hidden_dims).to(device)

    model_path = os.path.join(save_path, opt.pretrain)
    print(model_path)    
    model.load_state_dict(torch.load(model_path))
    data = data.to(device)
    img = img.to(device)

    model.eval()
    gz,iz,cz, gout,iout,cout = model(data.x, img.x, data.edge_index)

    adata.obsm['pred'] = cz.clone().detach().cpu().numpy()
    adata.obsm['pred'] = cz.to('cpu').detach().numpy().astype(np.float32)

    adata.obsm['gpred'] = gz.clone().detach().cpu().numpy()
    adata.obsm['gpred'] = gz.to('cpu').detach().numpy().astype(np.float32)

    adata.obsm['ipred'] = iz.clone().detach().cpu().numpy()
    adata.obsm['ipred'] = iz.to('cpu').detach().numpy().astype(np.float32)

    return adata

@torch.no_grad()
def test_nano_batch(opt, adata, hidden_dims=[512, 30], n_epochs=1000, lr=0.001, key_added='STAGATE',
                gradient_clipping=5.,  weight_decay=0.0001, verbose=True, 
                random_seed=0, save_loss=False, save_reconstrction=False, 
                device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'),
                save_path='../checkpoint/trans_gene/', ncluster=7, repeat=1):
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

    adata.X = sp.csr_matrix(adata.X)
    # adata.X = adata.X.A
    if 'highly_variable' in adata.var.columns:
        adata_Vars =  adata[:, adata.var['highly_variable']]
    else:
        adata_Vars = adata

    if verbose:
        print('Size of Input: ', adata_Vars.shape)
    if 'Spatial_Net' not in adata.uns.keys():
        raise ValueError("Spatial_Net is not existed! Run Cal_Spatial_Net first!")

    data, img = Transfer_img_Data(adata_Vars)
    model = TransImg(hidden_dims=[data.x.shape[1], img.x.shape[1]] + hidden_dims).to(device)

    model_path = os.path.join(save_path, opt.pretrain)
    print(model_path)
    model.load_state_dict(torch.load(model_path))

    gene_dim = data.x.shape[1]
    img_dim = img.x.shape[1]
    data.x = torch.cat([data.x, img.x], dim=1)
    loader = NeighborLoader(data, 
                            num_neighbors=[5,5],
                            batch_size=512,
                            shuffle=False
                            )
    model.eval()
    czs = None
    for batch in loader:
        batch = batch.to(device)
        b_gene = batch.x[:, :gene_dim]
        b_img = batch.x[:, gene_dim:]
        gz,iz,cz, gout,iout,cout = model(b_gene, b_img, batch.edge_index)

        if czs is None:
            czs = cz.clone().detach().cpu()
        else:
            czs = torch([czs, cz], dim=0)

    adata.obsm['pred'] = czs.numpy()
    return adata 

def train_nano_batch(opt, adata, hidden_dims=[512, 30], n_epochs=1000, lr=0.001, key_added='STAGATE',
                gradient_clipping=5.,  weight_decay=0.0001, verbose=True, 
                random_seed=0, save_loss=False, save_reconstrction=False, 
                device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'),
                save_path='../checkpoint/trans_gene/', ncluster=7, repeat=1):
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

    adata.X = sp.csr_matrix(adata.X)
    # adata.X = adata.X.A
    if 'highly_variable' in adata.var.columns:
        adata_Vars =  adata[:, adata.var['highly_variable']]
    else:
        adata_Vars = adata

    if verbose:
        print('Size of Input: ', adata_Vars.shape)
    if 'Spatial_Net' not in adata.uns.keys():
        raise ValueError("Spatial_Net is not existed! Run Cal_Spatial_Net first!")

    data, img = Transfer_img_Data(adata_Vars)
    gene_dim = data.x.shape[1]
    img_dim = img.x.shape[1]
    model = TransImg(hidden_dims=[data.x.shape[1], img.x.shape[1]] + hidden_dims).to(device)

    # create a minibatch 
    data.x = torch.cat([data.x, img.x], dim=1)
    # data = data.cuda()
    loader = NeighborLoader(data, 
                            num_neighbors=[5,5],
                            batch_size=512,
                            shuffle=True
                            )
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    for epoch in range(1, n_epochs+1):
        for bid, batch in enumerate(loader):
            model.train()
            batch = batch.to(device)
            optimizer.zero_grad()
            b_gene = batch.x[:, :gene_dim]
            b_img = batch.x[:, gene_dim:]
            gz,iz,cz, gout,iout,cout = model(b_gene, b_img, batch.edge_index)

            gloss = F.mse_loss(b_gene, gout)
            iloss = F.mse_loss(b_gene, iout)
            closs = F.mse_loss(b_gene, cout)
            loss = gloss + iloss + closs
            print('epoch: %d, batch: %d, loss: %.2f, gloss: %.2f, iloss: %.2f, closs: %.2f'%(epoch,bid, loss, gloss, iloss, closs))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)
            optimizer.step()
    torch.save(model.state_dict(), os.path.join(save_path, 'final_%d.pth'%(repeat)))


@torch.no_grad()
def test_nano_fov_batch(opt, adatas, model_name=None, hidden_dims=[512, 30], n_epochs=1000, lr=0.001, key_added='STAGATE',
                gradient_clipping=5.,  weight_decay=0.0001, verbose=True, 
                random_seed=0, save_loss=False, save_reconstrction=False, 
                device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'),
                save_path='../checkpoint/trans_gene/', ncluster=7, repeat=1):
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

    for adata in adatas:
        adata.X = sp.csr_matrix(adata.X)
        data, img = Transfer_img_Data(adata)
        # print(data.x.shape, img.x.shape)
        gene_dim = data.x.shape[1]
        img_dim = img.x.shape[1]
        data.x = torch.cat([data.x, img.x], dim=1)
        # data = data.to(device)
        # img = img.to(device)
        datas.append(data)
        # imgs.append(img)
        
        # if gt_frame is None:
        #     gt_frame = adata.obs['merge_cell_type']
        # else:
        #     gt_frame = pd.concat([gt_frame, adata.obs['merge_cell_type']])
    import anndata
    adata = anndata.concat(adatas)

    loader = DataLoader(datas, batch_size=1, shuffle=False)
    model = TransImg(hidden_dims=[gene_dim, img_dim] + hidden_dims).to(device)
    # torch.save(model.state_dict(), os.path.join(save_path, 'init.pth'))
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
    for idx, batch in enumerate(loader):
        batch = batch.to(device)
        # print(batch)
        bgene = batch.x[:, :gene_dim]
        bimg = batch.x[:, gene_dim:]
        # exit(0)
        edge_index = batch.edge_index
        gz,iz,cz, gout,iout,cout = model(bgene, bimg, edge_index)

        # cannot load into memory
        # save each fov first
        latent_path = os.path.join('merscope', 'preds')
        if not os.path.exists(latent_path):
            os.makedirs(latent_path)
        nlists = ['gene_latent','image_latent', 'combine_latent', 'gene_recon', 'image_recon', 'combine_recon']
        for n in nlists:
            if not os.path.exists(os.path.join(latent_path, n)):
                os.makedirs(os.path.join(latent_path, n))

        df = pd.DataFrame(gz.detach().cpu().numpy(), index=adatas[idx].obs.index)
        df.to_csv(os.path.join(latent_path, 'gene_latent', 'fov_%d.csv'%(idx)))

        df = pd.DataFrame(iz.detach().cpu().numpy(), index=adatas[idx].obs.index)
        df.to_csv(os.path.join(latent_path, 'image_latent', 'fov_%d.csv'%(idx)))

        df = pd.DataFrame(cz.detach().cpu().numpy(), index=adatas[idx].obs.index)
        df.to_csv(os.path.join(latent_path, 'combine_latent', 'fov_%d.csv'%(idx)))
            
        df = pd.DataFrame(gout.detach().cpu().numpy(), index=adatas[idx].obs.index)
        df.to_csv(os.path.join(latent_path, 'gene_recon', 'fov_%d.csv'%(idx)))

        df = pd.DataFrame(iout.detach().cpu().numpy(), index=adatas[idx].obs.index)
        df.to_csv(os.path.join(latent_path, 'image_recon', 'fov_%d.csv'%(idx)))

        df = pd.DataFrame(cout.detach().cpu().numpy(), index=adatas[idx].obs.index)
        df.to_csv(os.path.join(latent_path, 'combine_recon', 'fov_%d.csv'%(idx)))

        # print(cz.shape)
        # if hidden_matrix is None:
        #     hidden_matrix = cz.detach().cpu()
        #     gene_matrix = gz.detach().cpu()
        #     couts = cout.detach().cpu()
        #     img_matrix = iz.detach().cpu()
        # else:
        #     hidden_matrix = torch.cat([hidden_matrix, cz.detach().cpu()], dim=0)
        #     gene_matrix = torch.cat([gene_matrix, gz.detach().cpu()], dim=0)
        #     img_matrix = torch.cat([img_matrix, iz.detach().cpu()], dim=0)
        #     couts = torch.cat([couts, cout.detach().cpu()], dim=0)

@torch.no_grad()
def test_nano_fov_ablation(opt, adatas, model_name=None, hidden_dims=[512, 30], n_epochs=1000, lr=0.001, key_added='STAGATE',
                gradient_clipping=5.,  weight_decay=0.0001, verbose=True, 
                random_seed=0, save_loss=False, save_reconstrction=False, 
                device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'),
                save_path='../checkpoint/trans_gene/', ncluster=7, repeat=1,
                gene_only=True, img_only=False, combine_only=False, use_img_loss=False):
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
    if use_img_loss:
        model = TransImg2(hidden_dims=[gene_dim, img_dim] + hidden_dims).to(device)
    else:
        model = TransImg(hidden_dims=[gene_dim, img_dim] + hidden_dims, use_img_loss=use_img_loss).to(device)
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
    gouts = None
    iouts = None
    losses = 0
    for batch in loader:
        batch = batch.to(device)
        # print(batch)
        bgene = batch.x[:, :gene_dim]
        bimg = batch.x[:, gene_dim:]
        # exit(0)
        edge_index = batch.edge_index
        if use_img_loss:
            iz, iout = model(bgene, bimg, edge_index)
            cz = iz
            gz = iz
            cout = iout
            gout = iout
        else:
            gz,iz,cz, gout,iout,cout = model(bgene, bimg, edge_index)

        # gloss = F.mse_loss(bgene, gout)
        # iloss = F.mse_loss(bgene, iout)
        # closs = F.mse_loss(bgene, cout)
        # loss = (gloss + iloss + closs) / 3
        if gene_only:
            loss = F.mse_loss(bgene, gout)
        elif img_only and not use_img_loss:
            loss = F.mse_loss(bgene, iout)
        elif img_only and use_img_loss:
            loss = F.mse_loss(bimg, iout)
        else:
            loss = F.mse_loss(bgene, cout)
        losses += loss.item()

        print(cz.shape)
        if hidden_matrix is None:
            hidden_matrix = cz.detach().cpu()
            gene_matrix = gz.detach().cpu()
            couts = cout.detach().cpu()
            img_matrix = iz.detach().cpu()
            iouts = iout.detach().cpu()
            gouts = gout.detach().cpu()
        else:
            hidden_matrix = torch.cat([hidden_matrix, cz.detach().cpu()], dim=0)
            gene_matrix = torch.cat([gene_matrix, gz.detach().cpu()], dim=0)
            img_matrix = torch.cat([img_matrix, iz.detach().cpu()], dim=0)
            couts = torch.cat([couts, cout.detach().cpu()], dim=0)
            iouts = torch.cat([iouts, iout.detach().cpu()], dim=0)
            gouts = torch.cat([gouts, gout.detach().cpu()], dim=0)

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

    gouts = gouts.numpy().astype(np.float32)
    gouts[gouts < 0] = 0
    adata.layers['gene_recon'] = gouts

    iouts = iouts.numpy().astype(np.float32)
    iouts[iouts < 0] = 0
    adata.layers['img_recon'] = iouts
    return adata, losses

@torch.no_grad()
def test_nano_fov(opt, adatas, model_name=None, hidden_dims=[512, 30], n_epochs=1000, lr=0.001, key_added='sigra',
                gradient_clipping=5.,  weight_decay=0.0001, verbose=True, 
                random_seed=0, save_loss=False, save_reconstrction=False, 
                device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'),
                save_path='../checkpoint/trans_gene/', ncluster=7, repeat=1):
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
        # print(data.x.shape, img.x.shape)
        gene_dim = data.x.shape[1]
        img_dim = img.x.shape[1]
        data.x = torch.cat([data.x, img.x], dim=1)
        # data = data.to(device)
        # img = img.to(device)
        datas.append(data)
        # imgs.append(img)
        
        # if gt_frame is None:
        #     gt_frame = adata.obs['merge_cell_type']
        # else:
        #     gt_frame = pd.concat([gt_frame, adata.obs['merge_cell_type']])
    import anndata
    adata = anndata.concat(adatas)

    loader = DataLoader(datas, batch_size=1, shuffle=False)
    model = TransImg(hidden_dims=[gene_dim, img_dim] + hidden_dims).to(device)
    # torch.save(model.state_dict(), os.path.join(save_path, 'init.pth'))
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
        loss = (gloss + iloss + closs)
        losses += loss.item()

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
    print(losses)
    return adata, losses

# @torch.no_grad()
# def test_nano_fov(opt, adatas, model_name=None, hidden_dims=[512, 30], n_epochs=1000, lr=0.001, key_added='sigra',
#                 gradient_clipping=5.,  weight_decay=0.0001, verbose=True, 
#                 random_seed=0, save_loss=False, save_reconstrction=False, 
#                 device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'),
#                 save_path='../checkpoint/trans_gene/', ncluster=7, repeat=1):
#     seed = random_seed
#     import random
#     random.seed(seed)
#     os.environ['PYTHONHASHSEED'] = str(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed(seed)
#     os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
#     torch.backends.cudnn.deterministic = True
#     torch.backends.cudnn.benchmark = False
    
#     datas, imgs = [], []
#     gt_frame = None
#     datas, gene_dims = [], []
#     gene_dim = 0
#     img_dim = 0 # gene and img dim is same for all fovs

#     # [fov1(feat, graph), fov2(feat, graph), fov3, ... fov20]
#     # 

#     for adata in adatas:
#         adata.X = sp.csr_matrix(adata.X)
#         data, img = Transfer_img_Data(adata)
#         # print(data.x.shape, img.x.shape)
#         gene_dim = data.x.shape[1]
#         img_dim = img.x.shape[1]
#         data.x = torch.cat([data.x, img.x], dim=1)
#         # data = data.to(device)
#         # img = img.to(device)
#         datas.append(data)
#         # imgs.append(img)
        
#         # if gt_frame is None:
#         #     gt_frame = adata.obs['merge_cell_type']
#         # else:
#         #     gt_frame = pd.concat([gt_frame, adata.obs['merge_cell_type']])
#     import anndata
#     adata = anndata.concat(adatas)

#     loader = DataLoader(datas, batch_size=1, shuffle=False)
#     model = TransImg(hidden_dims=[gene_dim, img_dim] + hidden_dims).to(device)
#     # torch.save(model.state_dict(), os.path.join(save_path, 'init.pth'))
#     if model_name is not None:
#         model.load_state_dict(torch.load(os.path.join(save_path, model_name)))
#     else:
#         print(os.path.join(save_path, opt.pretrain))
#         model.load_state_dict(torch.load(os.path.join(save_path, opt.pretrain)))

#     seed = random_seed
#     import random
#     random.seed(seed)
#     os.environ['PYTHONHASHSEED'] = str(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed(seed)
#     os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
#     torch.backends.cudnn.deterministic = True
#     torch.backends.cudnn.benchmark = False

#     hidden_matrix = None
#     gene_matrix = None
#     img_matrix = None
#     couts = None
#     losses = 0
#     for batch in loader:
#         batch = batch.to(device)
#         # print(batch)
#         bgene = batch.x[:, :gene_dim]
#         bimg = batch.x[:, gene_dim:]
#         # exit(0)
#         edge_index = batch.edge_index
#         gz,iz,cz, gout,iout,cout = model(bgene, bimg, edge_index)

#         gloss = F.mse_loss(bgene, gout)
#         iloss = F.mse_loss(bgene, iout)
#         closs = F.mse_loss(bgene, cout)
#         loss = (gloss + iloss + closs) / 3
#         losses += loss.item()

#         print(cz.shape)
#         if hidden_matrix is None:
#             hidden_matrix = cz.detach().cpu()
#             gene_matrix = gz.detach().cpu()
#             couts = cout.detach().cpu()
#             img_matrix = iz.detach().cpu()
#         else:
#             hidden_matrix = torch.cat([hidden_matrix, cz.detach().cpu()], dim=0)
#             gene_matrix = torch.cat([gene_matrix, gz.detach().cpu()], dim=0)
#             img_matrix = torch.cat([img_matrix, iz.detach().cpu()], dim=0)
#             couts = torch.cat([couts, cout.detach().cpu()], dim=0)
#     # exit(0)
#     hidden_matrix = hidden_matrix.numpy()
#     gene_matrix = gene_matrix.numpy()
#     img_matrix = img_matrix.numpy()
#     adata.obsm['pred'] = hidden_matrix
#     adata.obsm['gene_pred'] = gene_matrix
#     adata.obsm['img_pred'] = img_matrix
#     couts = couts.numpy().astype(np.float32)
#     couts[couts < 0] = 0
#     adata.layers['recon'] = couts
#     return adata, losses


def train_nano_fov_ablation(opt, adatas, hidden_dims=[512, 30], n_epochs=1000, lr=0.001, key_added='STAGATE',
                gradient_clipping=5.,  weight_decay=0.0001, verbose=True, 
                random_seed=0, save_loss=False, save_reconstrction=False, 
                device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'),
                save_path='../checkpoint/trans_gene/', ncluster=7, repeat=1,
                gene_weight=0.1, img_weight=0.1, combine_weight=1.0, use_img_loss=False):
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
    if use_img_loss:
        model = TransImg2(hidden_dims=[gene_dim, img_dim] + hidden_dims).to(device)
    else:
        model = TransImg(hidden_dims=[gene_dim, img_dim] + hidden_dims, use_img_loss=use_img_loss).to(device)
    torch.save(model.state_dict(), os.path.join(save_path, 'init.pth'))
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
            if not use_img_loss:
                gz,iz,cz, gout,iout,cout = model(bgene, bimg, edge_index)
            else:
                iz, iout = model(bgene, bimg, edge_index)

            if use_img_loss:
                loss = F.mse_loss(bimg, iout)
            elif img_weight == 0 and combine_weight == 0:
                loss = F.mse_loss(bgene, gout)
            elif gene_weight == 0 and combine_weight == 0:
                loss = F.mse_loss(bgene, iout)
            elif gene_weight == 0 and img_weight == 0:
                loss = F.mse_loss(bgene, cout)
            # gloss = F.mse_loss(bgene, gout)
            # iloss = F.mse_loss(bgene, iout)
            # closs = F.mse_loss(bgene, cout)
            # loss = gene_weight * gloss + img_weight * iloss + combine_weight * closs
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)
            optimizer.step()
        
        if epoch > 500 and epoch % 100 == 0:
        # if epoch > 1:
            torch.save(model.state_dict(), os.path.join(save_path, 'final_%d_%d.pth'%(epoch, repeat)))
            # test_nano_fov(opt, adatas, model_name='final_%d_%d.pth'%(epoch, repeat), save_path=save_path, 
            # ncluster=8)

    torch.save(model.state_dict(), os.path.join(save_path, 'final_%d.pth'%(repeat)))
    return adata

    
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
        # data = data.to(device)
        # img = img.to(device)
        datas.append(data)
        # imgs.append(img)
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
        # if epoch > 1:
            torch.save(model.state_dict(), os.path.join(save_path, 'final_%d_%d.pth'%(epoch, repeat)))
            # test_nano_fov(opt, adatas, model_name='final_%d_%d.pth'%(epoch, repeat), save_path=save_path, 
            # ncluster=8)

    torch.save(model.state_dict(), os.path.join(save_path, 'final_%d.pth'%(repeat)))
    return adata



def train_nano(opt, adata, hidden_dims=[512, 30], n_epochs=1000, lr=0.001, key_added='STAGATE',
                gradient_clipping=5.,  weight_decay=0.0001, verbose=True, 
                random_seed=0, save_loss=False, save_reconstrction=False, 
                device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'),
                save_path='../checkpoint/trans_gene/', ncluster=7, repeat=1):
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

    adata.X = sp.csr_matrix(adata.X)
    # adata.X = adata.X.A
    if 'highly_variable' in adata.var.columns:
        adata_Vars =  adata[:, adata.var['highly_variable']]
    else:
        adata_Vars = adata

    if verbose:
        print('Size of Input: ', adata_Vars.shape)
    if 'Spatial_Net' not in adata.uns.keys():
        raise ValueError("Spatial_Net is not existed! Run Cal_Spatial_Net first!")

    data, img = Transfer_img_Data(adata_Vars)
    model = TransImg(hidden_dims=[data.x.shape[1], img.x.shape[1]] + hidden_dims).to(device)


    torch.save(model.state_dict(), os.path.join(save_path, 'init.pth'))

    data = data.to(device)
    img = img.to(device)

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
    
    # for epoch in range(1, n_epochs):
    for epoch in tqdm(range(1, n_epochs+1)):
        model.train()
        optimizer.zero_grad()
        gz,iz,cz, gout,iout,cout = model(data.x, img.x, data.edge_index)

        gloss = F.mse_loss(data.x, gout)
        iloss = F.mse_loss(data.x, iout)
        closs = F.mse_loss(data.x, cout)
        loss = gloss + iloss + closs
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)
        optimizer.step()

        if epoch > 500 and epoch % 100 == 0:
            torch.save(model.state_dict(), os.path.join(save_path, 'final_%d_%d.pth'%(repeat, epoch)))
            opt.pretrain = 'final_%d_%d.pth'%(repeat, epoch)
            with torch.no_grad():
                adata = test_nano(opt, adata, hidden_dims=[512, 30],  n_epochs=opt.epochs, save_loss=True, 
                    lr=opt.lr, random_seed=opt.seed, save_path=sp, ncluster=ncluster, repeat=repeat)
            
            import anndata
            adata_pred = anndata.AnnData(cz.clone().detach().cpu().numpy(), obs=adata.obs)
            adata_pred.obs['merge_cell_type'] = adata.obs['merge_cell_type']
            sc.pp.neighbors(adata_pred, ncluster)
            
            def res_search(adata, ncluster, seed):
                start = 0; end =3
                while(start < end):
                    res = (start + end) / 2
            
                    random.seed(seed)                                                            
                    torch.manual_seed(seed)   
                    torch.cuda.manual_seed(seed)                                                      
                    torch.cuda.manual_seed_all(seed)                                             
                    np.random.seed(seed)                                                         
                    os.environ['PYTHONHASHSEED'] = str(seed)                                     
                    torch.backends.cudnn.deterministic = True                                    
                    torch.backends.cudnn.benchmark = False 
            
                    sc.tl.leiden(adata, random_state=seed, resolution=res)
                    count = len(set(adata.obs['leiden']))
                    print(count)
                    if count == ncluster:
                        print('find', res)
                        return res
                    if count > ncluster:
                        end = res
                    else:
                        start = res
                raise NotImplementedError()

            res = res_search(adata_pred, ncluster, seed)
            random.seed(seed)                                                            
            torch.manual_seed(seed)   
            torch.cuda.manual_seed(seed)                                                      
            torch.cuda.manual_seed_all(seed)                                             
            np.random.seed(seed)                                                         
            os.environ['PYTHONHASHSEED'] = str(seed)                                     
            torch.backends.cudnn.deterministic = True                                    
            torch.backends.cudnn.benchmark = False 
            sc.tl.leiden(adata_pred, resolution=res, key_added='leiden', random_state=seed)
            obs_df = adata_pred.obs.dropna()
            ARI = adjusted_rand_score(obs_df['leiden'], obs_df['cell_type'])
            print(ARI)

    torch.save(model.state_dict(), os.path.join(save_path, 'final_%d.pth'%(repeat)))
    return adata

def train_one_branch(adata, hidden_dims=[512, 30], n_epochs=1000, lr=0.001, key_added='STAGATE',
                gradient_clipping=5.,  weight_decay=0.0001, verbose=True, 
                random_seed=0, save_loss=False, save_reconstrction=False, 
                device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'),
                save_path='../checkpoint/trans_gene/', ncluster=7, repeat=1, use_img_loss=0, use_combine=1,
                X_train_idx=None, X_test_idx=None, lambda_1 = 1, lambda_2 = 1, lambda_3 = 1, branch='gene'):
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

    adata.X = sp.csr_matrix(adata.X)
    # adata.X = adata.X.A
    if 'highly_variable' in adata.var.columns:
        adata_Vars =  adata[:, adata.var['highly_variable']]
    else:
        adata_Vars = adata
    
    if verbose:
        print('Size of Input: ', adata_Vars.shape)
    if 'Spatial_Net' not in adata.uns.keys():
        raise ValueError("Spatial_Net is not existed! Run Cal_Spatial_Net first!")

    data, img = Transfer_img_Data(adata_Vars)
    if branch == 'img_img':
        use_img_loss = 1
    model = TransImg(hidden_dims=[data.x.shape[1], img.x.shape[1]] + hidden_dims, use_img_loss=use_img_loss).to(device)


    torch.save(model.state_dict(), os.path.join(save_path, 'init.pth'))
    data = data.to(device)
    img = img.to(device)

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
        
        model.train()
        optimizer.zero_grad()

        gz,iz,cz, gout,iout,cout = model(data.x, img.x, data.edge_index)

        if branch == 'gene':
            loss = F.mse_loss(data.x, gout)

        elif branch == 'img_gene':
            loss = F.mse_loss(data.x, iout)

        elif branch == 'img_img':
            loss = F.mse_loss(img.x, iout)
        
        elif branch == 'combine_only':
            loss = F.mse_loss(data.x, cout)
        else:
            raise NotImplementedError()
        
        # if use_img_loss:
        #     print('use img to build reconstrauction loss')
        #     iloss = F.mse_loss(img.x, iout)
        # else:
        #     iloss = F.mse_loss(data.x, iout)
        
        # if use_combine:
        #     closs = F.mse_loss(data.x, cout)
        #     loss = gloss * lambda_1 + iloss * lambda_2 + closs * lambda_3
        # else:
        #     loss = gloss * lambda_1 + iloss * lambda_2

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)
        optimizer.step()
    torch.save(model.state_dict(), os.path.join(save_path, 'final_%d_%s.pth'%(repeat, branch)))

    with torch.no_grad():
        model.load_state_dict(torch.load(os.path.join(save_path, 'final_%d_%s.pth'%(repeat, branch))))

        model.eval()
        gz,iz,cz, gout,iout,cout = model(data.x, img.x, data.edge_index)

        if branch == 'gene':
            loss = F.mse_loss(data.x, gout)
            predz = gz
            predout = gout

        elif branch == 'img_gene':
            loss = F.mse_loss(data.x, iout)
            predz = iz
            predout = iout

        elif branch == 'img_img':
            loss = F.mse_loss(img.x, iout)
            predz = iz
            predout = iout
        
        elif branch == 'combine_only':
            loss = F.mse_loss(data.x, cout)
            predz = cz
            predout = cout


        if not os.path.exists(save_path):
            os.makedirs(save_path)
    
        adata_Vars.obsm['pred'] = predz.to('cpu').detach().numpy().astype(np.float32)
        # output = predout.to('cpu').detach().numpy().astype(np.float32)
        # output[output < 0] = 0
        # adata_Vars.layers['recon'] = output
    plt.close('all')
    return adata_Vars

def train_img(adata, hidden_dims=[512, 30], n_epochs=1000, lr=0.001, key_added='STAGATE',
                gradient_clipping=5.,  weight_decay=0.0001, verbose=True, 
                random_seed=0, save_loss=False, save_reconstrction=False, 
                device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'),
                save_path='../checkpoint/trans_gene/', ncluster=7, repeat=1, use_img_loss=0, use_combine=1,
                X_train_idx=None, X_test_idx=None, lambda_1 = 1, lambda_2 = 1, lambda_3 = 1):
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

    adata.X = sp.csr_matrix(adata.X)
    # adata.X = adata.X.A
    if 'highly_variable' in adata.var.columns:
        adata_Vars =  adata[:, adata.var['highly_variable']]
    else:
        adata_Vars = adata
    
    if verbose:
        print('Size of Input: ', adata_Vars.shape)
    if 'Spatial_Net' not in adata.uns.keys():
        raise ValueError("Spatial_Net is not existed! Run Cal_Spatial_Net first!")

    data, img = Transfer_img_Data(adata_Vars)
    model = TransImg(hidden_dims=[data.x.shape[1], img.x.shape[1]] + hidden_dims, use_img_loss=use_img_loss).to(device)


    torch.save(model.state_dict(), os.path.join(save_path, 'init.pth'))
    data = data.to(device)
    img = img.to(device)

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
        
        model.train()
        optimizer.zero_grad()

        gz,iz,cz, gout,iout,cout = model(data.x, img.x, data.edge_index)

        gloss = F.mse_loss(data.x, gout)
        if use_img_loss:
            print('use img to build reconstrauction loss')
            iloss = F.mse_loss(img.x, iout)
        else:
            iloss = F.mse_loss(data.x, iout)
        
        if use_combine:
            closs = F.mse_loss(data.x, cout)
            loss = gloss * lambda_1 + iloss * lambda_2 + closs * lambda_3
        else:
            loss = gloss * lambda_1 + iloss * lambda_2

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)
        optimizer.step()
    torch.save(model.state_dict(), os.path.join(save_path, 'final_%d.pth'%(repeat)))

    with torch.no_grad():
        model.load_state_dict(torch.load(os.path.join(save_path, 'final_%d.pth'%(repeat))))

        model.eval()
        gz,iz,cz, gout,iout,cout = model(data.x, img.x, data.edge_index)

        gloss = F.mse_loss(data.x[data.val_mask], gout[data.val_mask])
        if use_img_loss:
            iloss = F.mse_loss(img.x[data.val_mask], iout[data.val_mask])
        else:
            iloss = F.mse_loss(data.x[data.val_mask], iout[data.val_mask])
        
        if use_combine:
            closs = F.mse_loss(data.x[data.val_mask], cout[data.val_mask])
            loss = gloss * lambda_1 + iloss * lambda_2 + closs * lambda_3
            loss = loss / (lambda_1 + lambda_2 + lambda_3)
        else:
            loss = gloss + iloss



        if not os.path.exists(save_path):
            os.makedirs(save_path)
        adata_Vars.obsm['pred'] = cz.clone().detach().cpu().numpy()
        sc.pp.neighbors(adata_Vars, use_rep='pred')
        sc.tl.umap(adata_Vars)
        plt.rcParams["figure.figsize"] = (3, 3)
        sc.settings.figdir = save_path
        ax = sc.pl.umap(adata_Vars, color=['Ground Truth'], show=False, title='combined latent variables')
        plt.savefig(os.path.join(save_path, 'umap_final.pdf'), bbox_inches='tight')
    
        adata_Vars.obsm['pred'] = cz.to('cpu').detach().numpy().astype(np.float32)
        output = cout.to('cpu').detach().numpy().astype(np.float32)
        output[output < 0] = 0
        adata_Vars.layers['recon'] = output
    plt.close('all')
    return adata_Vars, loss.item()


def train_10x_all(adatas, hidden_dims=[512, 30], n_epochs=1000, lr=0.001, key_added='STAGATE',
                gradient_clipping=5.,  weight_decay=0.0001, verbose=True, 
                random_seed=0, save_loss=False, save_reconstrction=False, 
                device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'),
                save_path='../checkpoint/trans_gene/', ncluster=7, repeat=1, use_img_loss=0, use_combine=1,
                lambda_1 = 1, lambda_2 = 1, lambda_3 = 1):
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

    adata_vars_list = []
    for adata in adatas:
        adata.X = sp.csr_matrix(adata.X)
        # adata.X = adata.X.A
        if 'highly_variable' in adata.var.columns:
            adata_Vars =  adata[:, adata.var['highly_variable']]
        else:
            adata_Vars = adata
        adata_vars_list.append(adata_Vars)

    if verbose:
        print('Size of Input: ', adata_Vars.shape)
    if 'Spatial_Net' not in adata.uns.keys():
        raise ValueError("Spatial_Net is not existed! Run Cal_Spatial_Net first!")

    datas = []
    imgs = []
    img_dim = 0
    gene_dim = 0
    for adata in adatas:
        data, img = Transfer_img_Data(adata_Vars)
        gene_dim = data.x.shape[1]
        img_dim = img.x.shape[1]
        data.x = torch.cat([data.x, img.x], dim=1)
        datas.append(data)
        # imgs.append(img)

    loader = DataLoader(datas, batch_size=1, shuffle=False)

    model = TransImg(hidden_dims=[gene_dim, img_dim] + hidden_dims, use_img_loss=use_img_loss).to(device)

    torch.save(model.state_dict(), os.path.join(save_path, 'init.pth'))
    data = data.to(device)
    img = img.to(device)

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
        model.train()
        for idx, batch in enumerate(loader):
            batch = batch.to(device)
            data = batch.x[:, :gene_dim]
            img = batch.x[:, gene_dim:]
            optimizer.zero_grad()
            gz,iz,cz, gout,iout,cout = model(data, img, batch.edge_index)
            gloss = F.mse_loss(data, gout)
            if use_img_loss:
                iloss = F.mse_loss(img, iout)
            else:
                iloss = F.mse_loss(data, iout)
        
            if use_combine:
                closs = F.mse_loss(data, cout)
                loss = gloss + iloss + closs
            else:
                loss = gloss + iloss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)
            optimizer.step()

    torch.save(model.state_dict(), os.path.join(save_path, 'final_%d.pth'%(repeat)))

    with torch.no_grad():
        model.load_state_dict(torch.load(os.path.join(save_path, 'final_%d.pth'%(repeat))))

        model.eval()
        for idx, batch in enumerate(loader):
            print(idx)
            batch = batch.to(device)
            data = batch.x[:, :gene_dim]
            img = batch.x[:, gene_dim:]
            gz,iz,cz, gout,iout,cout = model(data, img, batch.edge_index)
            # print(cz.shape)
            cell_shape = adata_vars_list[idx].shape[0]
            if cz.shape[0] != cell_shape:
                cz = cz[:cell_shape, ...]
                cout = cout[:cell_shape, ...]
                
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            adata_vars_list[idx].obsm['pred'] = cz.clone().detach().cpu().numpy()

    
            adata_vars_list[idx].obsm['pred'] = cz.to('cpu').detach().numpy().astype(np.float32)
            output = cout.to('cpu').detach().numpy().astype(np.float32)
            output[output < 0] = 0
            adata_vars_list[idx].layers['recon'] = output

    return adata_vars_list


def test_10x_all(adatas, hidden_dims=[512, 30], n_epochs=1000, lr=0.001, key_added='STAGATE',
                gradient_clipping=5.,  weight_decay=0.0001, verbose=True, 
                random_seed=0, save_loss=False, save_reconstrction=False, 
                device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'),
                save_path='../checkpoint/trans_gene/', ncluster=7, repeat=1, use_img_loss=0, use_combine=1,
                lambda_1 = 1, lambda_2 = 1, lambda_3 = 1):
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

    adata_vars_list = []
    for adata in adatas:
        adata.X = sp.csr_matrix(adata.X)
        # adata.X = adata.X.A
        if 'highly_variable' in adata.var.columns:
            adata_Vars =  adata[:, adata.var['highly_variable']]
        else:
            adata_Vars = adata
        adata_vars_list.append(adata_Vars)

    if verbose:
        print('Size of Input: ', adata_Vars.shape)
    if 'Spatial_Net' not in adata.uns.keys():
        raise ValueError("Spatial_Net is not existed! Run Cal_Spatial_Net first!")

    datas = []
    imgs = []
    img_dim = 0
    gene_dim = 0
    for adata in adatas:
        data, img = Transfer_img_Data(adata_Vars)
        gene_dim = data.x.shape[1]
        img_dim = img.x.shape[1]
        data.x = torch.cat([data.x, img.x], dim=1)
        datas.append(data)
        # imgs.append(img)

    loader = DataLoader(datas, batch_size=1, shuffle=False)

    model = TransImg(hidden_dims=[gene_dim, img_dim] + hidden_dims, use_img_loss=use_img_loss).to(device)

    # torch.save(model.state_dict(), os.path.join(save_path, 'init.pth'))
    # data = data.to(device)
    # img = img.to(device)

    # optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    # seed = random_seed
    # import random
    # random.seed(seed)
    # os.environ['PYTHONHASHSEED'] = str(seed)
    # np.random.seed(seed)
    # torch.manual_seed(seed)
    # torch.cuda.manual_seed(seed)
    # os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    
    # for epoch in tqdm(range(1, n_epochs+1)):
    #     model.train()
    #     for idx, batch in enumerate(loader):
    #         batch = batch.to(device)
    #         data = batch.x[:, :gene_dim]
    #         img = batch.x[:, gene_dim:]
    #         optimizer.zero_grad()
    #         gz,iz,cz, gout,iout,cout = model(data, img, batch.edge_index)
    #         gloss = F.mse_loss(data, gout)
    #         if use_img_loss:
    #             iloss = F.mse_loss(img, iout)
    #         else:
    #             iloss = F.mse_loss(data, iout)
        
    #         if use_combine:
    #             closs = F.mse_loss(data, cout)
    #             loss = gloss + iloss + closs
    #         else:
    #             loss = gloss + iloss
    #         loss.backward()
    #         torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)
    #         optimizer.step()

    # torch.save(model.state_dict(), os.path.join(save_path, 'final_%d.pth'%(repeat)))

    with torch.no_grad():
        model.load_state_dict(torch.load(os.path.join(save_path, 'final_%d.pth'%(repeat))))

        model.eval()
        for idx, batch in enumerate(loader):
            print(idx)
            batch = batch.to(device)
            data = batch.x[:, :gene_dim]
            img = batch.x[:, gene_dim:]
            gz,iz,cz, gout,iout,cout = model(data, img, batch.edge_index)
            # print(cz.shape)
            cell_shape = adata_vars_list[idx].shape[0]
            if cz.shape[0] != cell_shape:
                cz = cz[:cell_shape, ...]
                cout = cout[:cell_shape, ...]
                
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            adata_vars_list[idx].obsm['pred'] = cz.clone().detach().cpu().numpy()

    
            adata_vars_list[idx].obsm['pred'] = cz.to('cpu').detach().numpy().astype(np.float32)
            output = cout.to('cpu').detach().numpy().astype(np.float32)
            output[output < 0] = 0
            adata_vars_list[idx].layers['recon'] = output

    return adata_vars_list