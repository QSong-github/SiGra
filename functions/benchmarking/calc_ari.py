import pandas as pd
import numpy as np
from sklearn.metrics.cluster import adjusted_rand_score
from scipy.optimize import linear_sum_assignment
import os
import scanpy as sc

def _hungarian_match(flat_preds, flat_target, preds_k, target_k):
    num_samples = flat_preds.shape[0]
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

def match_cluster_to_cell(df_all, key):
    leiden_number = list(set(df_all[key]))
    dicts = {}
    for ln in leiden_number:
        # print(ln)
        ind = (df_all[key] == ln)
        temp = df_all[ind]
        # print(temp)
        df = temp['merge_cell_type'].value_counts()
        dicts[int(ln)] = df.index[0]
    return dicts

def get_stlearn(root='stlearn', num_fovs=30, key='cluster'):
    pred_all = []
    gt_all = []
    index = []
    for i in range(1, num_fovs + 1):
        csv = pd.read_csv(os.path.join(root, 'fov%d'%i, 'fov%d_cluster.csv'%(i)), header=0, index_col=0)
        index.extend(list(csv.index))
        pred = csv[key].astype('category').cat.codes
        pred = csv[key].astype(np.int8)
        csv[key] = pred
        gt = csv['merge_cell_type'].astype('category')
        gt_cat = gt.cat.codes
        gt_cat = gt_cat.astype(np.int8)

        dominate_dicts = match_cluster_to_cell(csv, key)
        # print(dominate_dicts)

        id2cell = {}
        for g,gc in zip(gt, gt_cat):
            if not gc in id2cell:
                id2cell[gc] = g
        # print('id2cell: ', id2cell)
        # id2cell = {5: 'myeloid', 3: 'lymphocyte', 2: 'fibroblast', 4: 'mast', 0: 'endothelial', 6: 'neutrophil', 1: 'epithelial', 7: 'tumors'}


        match = _hungarian_match(pred, gt_cat, len(set(pred)),len(set(gt_cat)))
        # print(match)
        pred2id = {}
        for outc, gtc in match:
            pred2id[outc] = gtc
        # print(pred2id)
        predcell = []
        for idx, p in enumerate(pred):
            if p in pred2id and pred2id[p] in id2cell:
                predcell.append(id2cell[pred2id[p]])
            else:
                # pred2id[p] in dominate_dicts:
                predcell.append(dominate_dicts[p])

        pred_all.extend(predcell)
        gt_all.extend(gt)
    ari = adjusted_rand_score(gt_all, pred_all)
    df = pd.DataFrame(index=index)
    df['pred'] = pred_all
    df['gt'] = gt_all
    df.to_csv('cell2domain/stlearn.csv')
    print('%s: ari:%.2f'%(root, ari))

def get_bayesspace(root='bayesspace', num_fovs=30, key='pred'):
    pred_all = []
    gt_all = []
    index = []
    for i in range(1, num_fovs+1):
        csv = pd.read_csv(os.path.join(root, 'fov%d_bayesSpace.csv'%(i)), header=0, index_col=0)
        index.extend(list(csv.index))
        pred = csv[key].astype('category').cat.codes
        pred = csv[key].astype(np.int8)
        csv['merge_cell_type'] = csv['gt']
        csv[key] = pred
        gt = csv['merge_cell_type'].astype('category')
        gt_cat = gt.cat.codes
        gt_cat = gt_cat.astype(np.int8)

        dominate_dicts = match_cluster_to_cell(csv, key)
        # print(dominate_dicts)

        id2cell = {}
        for g,gc in zip(gt, gt_cat):
            if not gc in id2cell:
                id2cell[gc] = g
        # print('id2cell: ', id2cell)
        # id2cell = {5: 'myeloid', 3: 'lymphocyte', 2: 'fibroblast', 4: 'mast', 0: 'endothelial', 6: 'neutrophil', 1: 'epithelial', 7: 'tumors'}


        match = _hungarian_match(pred, gt_cat, len(set(pred)),len(set(gt_cat)))
        # print(match)
        pred2id = {}
        for outc, gtc in match:
            pred2id[outc] = gtc
        # print(pred2id)
        predcell = []
        for idx, p in enumerate(pred):
            if p in pred2id and pred2id[p] in id2cell:
                predcell.append(id2cell[pred2id[p]])
            else:
                # pred2id[p] in dominate_dicts:
                predcell.append(dominate_dicts[p])

        pred_all.extend(predcell)
        gt_all.extend(gt)
    ari = adjusted_rand_score(gt_all, pred_all)
    df = pd.DataFrame(index=index)
    df['pred'] = pred_all
    df['gt'] = gt_all
    df.to_csv('bayesspace.csv')
    print('%s: ari:%.2f'%(root, ari))


def get_seurat(root='seurat', num_fovs=30, key='seurat_clusters', sep='\t', dataroot='spagcn/lung5-1'):
    pred_all = []
    gt_all = []
    index = []

    for i in range(1, num_fovs+1):
        csv = pd.read_csv(os.path.join(root, 'fov%d.csv'%(i)), header=0, index_col=0, sep=sep)
        index.extend(list(csv.index))
        adata = pd.read_csv(os.path.join(dataroot, 'fov%d.csv'%(i)), header=0, index_col=0)
        # merge_cell_type = adata.obs['merge_cell_type']
        csv['merge_cell_type'] = adata.loc[csv.index, 'merge_cell_type']
        pred = csv[key].astype('category').cat.codes
        pred = csv[key].astype(np.int8)
        csv[key] = pred
        gt = csv['merge_cell_type'].astype('category')
        gt_cat = gt.cat.codes
        gt_cat = gt_cat.astype(np.int8)

        dominate_dicts = match_cluster_to_cell(csv, key)
        # print(dominate_dicts)

        id2cell = {}
        for g,gc in zip(gt, gt_cat):
            if not gc in id2cell:
                id2cell[gc] = g
        # print('id2cell: ', id2cell)
        # id2cell = {5: 'myeloid', 3: 'lymphocyte', 2: 'fibroblast', 4: 'mast', 0: 'endothelial', 6: 'neutrophil', 1: 'epithelial', 7: 'tumors'}


        match = _hungarian_match(pred, gt_cat, len(set(pred)),len(set(gt_cat)))
        # print(match)
        pred2id = {}
        for outc, gtc in match:
            pred2id[outc] = gtc
        # print(pred2id)
        predcell = []
        for idx, p in enumerate(pred):
            if p in pred2id and pred2id[p] in id2cell:
                predcell.append(id2cell[pred2id[p]])
            else:
                # pred2id[p] in dominate_dicts:
                predcell.append(dominate_dicts[p])

        pred_all.extend(predcell)
        gt_all.extend(gt)
    ari = adjusted_rand_score(gt_all, pred_all)
    df = pd.DataFrame(index=index)
    df['pred'] = pred_all
    df['gt'] = gt_all
    df.to_csv('cell2domain/seurat.csv')
    print('%s: ari:%.2f'%(root, ari))

def get_fovs(root='scanpy', num_fovs=30, key='scanpy', sep=','):
    pred_all = []
    gt_all = []
    index = []

    for i in range(1, num_fovs+1):
        csv = pd.read_csv(os.path.join(root, 'fov%d.csv'%(i)), header=0, index_col=0, sep=sep)
        index.extend(list(csv.index))
        ind = csv['merge_cell_type'].isna()
        csv = csv[~ind]
        pred = csv[key].astype('category').cat.codes
        pred = csv[key].astype(np.int8)
        csv[key] = pred
        # pred = pred.astype(np.int8)
        gt = csv['merge_cell_type'].astype('category')
        gt_cat = gt.cat.codes
        gt_cat = gt_cat.astype(np.int8)

        dominate_dicts = match_cluster_to_cell(csv, key)
        # print(dominate_dicts)

        id2cell = {}
        for g,gc in zip(gt, gt_cat):
            if not gc in id2cell:
                id2cell[gc] = g
        # print('id2cell: ', id2cell)
        # id2cell = {5: 'myeloid', 3: 'lymphocyte', 2: 'fibroblast', 4: 'mast', 0: 'endothelial', 6: 'neutrophil', 1: 'epithelial', 7: 'tumors'}


        match = _hungarian_match(pred, gt_cat, len(set(pred)),len(set(gt_cat)))
        # print(match)
        pred2id = {}
        for outc, gtc in match:
            pred2id[outc] = gtc
        # print(pred2id)
        predcell = []
        for idx, p in enumerate(pred):
            if p in pred2id and pred2id[p] in id2cell:
                predcell.append(id2cell[pred2id[p]])
            else:
                # pred2id[p] in dominate_dicts:
                predcell.append(dominate_dicts[p])

        pred_all.extend(predcell)
        gt_all.extend(gt)
    ari = adjusted_rand_score(gt_all, pred_all)
    df = pd.DataFrame(index=index)
    df['pred'] = pred_all
    df['gt'] = gt_all
    df.to_csv('cell2domain/scanpy.csv')
    print('%s: ari:%.2f'%(root, ari))
        
# get_fovs(root='scanpy/lung5-1', num_fovs=30)
# get_fovs(root='scanpy/lung5-2', num_fovs=30)
# get_fovs(root='scanpy/lung5-3', num_fovs=30)
# get_fovs(root='scanpy/lung6', num_fovs=30)
# get_fovs(root='scanpy/lung9-1', num_fovs=20)
# get_fovs(root='scanpy/lung9-2', num_fovs=45)
# get_fovs(root='scanpy/lung12', num_fovs=28)
# get_fovs(root='scanpy/lung13', num_fovs=20)

# get_fovs(root='STAGATE/lung5-1', num_fovs=30, key='mclust')
# get_fovs(root='STAGATE/lung5-2', num_fovs=30, key='mclust')
# get_fovs(root='STAGATE/lung5-3', num_fovs=30, key='mclust')
# get_fovs(root='STAGATE/lung6', num_fovs=30, key='mclust')
# get_fovs(root='STAGATE/lung9-1', num_fovs=20, key='mclust')
# get_fovs(root='STAGATE/lung9-2', num_fovs=45, key='mclust')
# get_fovs(root='STAGATE/lung12', num_fovs=28, key='mclust')
# get_fovs(root='STAGATE/lung13', num_fovs=20, key='mclust')

# get_fovs(root='spagcn/lung5-1', num_fovs=30, key='refined_pred')
# get_fovs(root='spagcn/lung5-2', num_fovs=30, key='refined_pred')
# get_fovs(root='spagcn/lung5-3', num_fovs=30, key='refined_pred')
# get_fovs(root='spagcn/lung6', num_fovs=30, key='refined_pred')
# get_fovs(root='spagcn/lung9-1', num_fovs=20, key='refined_pred')
# get_fovs(root='spagcn/lung9-2', num_fovs=45, key='refined_pred')
# get_fovs(root='spagcn/lung12', num_fovs=28, key='refined_pred')
# get_fovs(root='spagcn/lung13', num_fovs=20, key='refined_pred')

# get_seurat(root='seurat/lung5-1', num_fovs=30, key='seurat_clusters', sep='\t', dataroot='spagcn/lung5-1')
# get_seurat(root='seurat/lung5-2', num_fovs=30, key='seurat_clusters', sep='\t', dataroot='spagcn/lung5-2')
# get_seurat(root='seurat/lung5-3', num_fovs=30, key='seurat_clusters',sep='\t', dataroot='spagcn/lung5-3')
# get_seurat(root='seurat/lung6', num_fovs=30, key='seurat_clusters', sep='\t', dataroot='spagcn/lung6')
# get_seurat(root='seurat/lung9-1', num_fovs=20, key='seurat_clusters', sep='\t', dataroot='spagcn/lung9-1')
# get_seurat(root='seurat/lung9-2', num_fovs=45, key='seurat_clusters', sep='\t', dataroot='spagcn/lung9-2')
# get_seurat(root='seurat/lung12', num_fovs=28, key='seurat_clusters', sep='\t', dataroot='spagcn/lung12')
# get_seurat(root='seurat/lung13', num_fovs=20, key='seurat_clusters', sep='\t', dataroot='spagcn/lung13')

# get_stlearn(root='stlearn/lung5-1', num_fovs=30)
# get_stlearn(root='stlearn/lung5-2', num_fovs=30)
# get_stlearn(root='stlearn/lung5-3', num_fovs=30)
# get_stlearn(root='stlearn/lung6', num_fovs=30)
# get_stlearn(root='stlearn/lung9-1', num_fovs=20)
# get_stlearn(root='stlearn/lung9-2', num_fovs=45)
# get_stlearn(root='stlearn/lung12', num_fovs=28)
# get_stlearn(root='stlearn/lung13', num_fovs=20)

# get_bayesspace(root='bayesspace/lung5-1', num_fovs=30)
# get_bayesspace(root='bayesspace/lung5-2', num_fovs=30)
# get_bayesspace(root='bayesspace/lung5-3', num_fovs=30)
# get_bayesspace(root='bayesspace/lung6', num_fovs=30)
# get_bayesspace(root='bayesspace/lung9-1', num_fovs=20)
# get_bayesspace(root='bayesspace/lung9-2', num_fovs=45)
# get_bayesspace(root='bayesspace/lung12', num_fovs=28)
# get_bayesspace(root='bayesspace/lung13', num_fovs=20)