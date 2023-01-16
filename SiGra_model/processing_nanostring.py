import pandas as pd
import numpy as np
import scanpy as sc
import anndata as AD
import cv2

import os

def gen_h5ad(id, img_id, fov):
    root = '../dataset/nanostring'
    # img_roots = os.listdir(id)
    # img_root = None
    # for root in img_roots:
    # 	if '.jpg' in root:
    # 		img_root = os.path.join(id, root)
    # 		break
    img_root = os.path.join(root, id, 'CellComposite_%s.jpg'%(img_id))
    print(img_root)
    img = cv2.imread(img_root)
    height, width, c = img.shape
    gene_expression = os.path.join(root, 'Lung9_Rep1_exprMat_file.csv')
    ge = pd.read_csv(gene_expression, delimiter=',')
    gene_f1 = ge[ge['fov'] == int(fov)]
    gene_f1 = gene_f1.drop(columns=['fov'])
    gene_f1 = gene_f1.set_index('cell_ID')
    idx = gene_f1.index

    annor = os.path.join(root, 'matched_annotation_all.csv')
    anno = pd.read_csv(annor)
    anno_f1 = anno[anno['fov'] == int(fov)]

    w, h = 60, 60
    # drop the cells at the edge
    # print(anno_f1.columns)
    # print(anno_f1['CenterX_local_px'])
    # print(anno_f1.shape)
    # exit(0)

    # for i in range(anno_f1.shape[0]):
    # 	print(anno_f1['CenterX_local_px'][i], anno_f1['CenterY_local_px'][i])
    # 	cx, cy = float(anno_f1['CenterX_local_px'][i]), float(anno_f1['CenterY_local_px'][i])
    # 	anno_f1['CenterY_local_px'][i] = height - float(anno_f1['CenterY_local_px'][i])

    # 	if cx - w < 0 or cx + w > width or cy - h < 0 or cy + h > height:
    # 		anno_f1['cell_type'][i] = np.nan
    # anno_f1 = anno_f1.set_index('cell_ID').reindex(idx)

    for i, row in anno_f1.iterrows():
        # print(anno_f1['CenterX_local_px'][i], anno_f1['CenterY_local_px'][i])
        cx, cy = float(anno_f1['CenterX_local_px'][i]), float(anno_f1['CenterY_local_px'][i])
        anno_f1['CenterY_local_px'][i] = height - float(anno_f1['CenterY_local_px'][i])

        if cx - w < 0 or cx + w > width or cy - h < 0 or cy + h > height:
            anno_f1['cell_type'][i] = np.nan
    anno_f1 = anno_f1.set_index('cell_ID').reindex(idx)



    gene_f1['cell_type'] = anno_f1['cell_type']
    gene_f1['niche'] = anno_f1['niche']
    gene_f1 = gene_f1.dropna(axis=0, how='any')
    # df = pd.DataFrame(gene_f1, columns=['cell_type', 'niche'])
    gene_f1 = gene_f1.drop(columns=['cell_type', 'niche'])

    adata = AD.AnnData(gene_f1)
    anno_f1.index = anno_f1.index.map(str)
    adata.obs['cell_type'] = anno_f1.loc[adata.obs_names, 'cell_type']
    adata.obs['niche'] = anno_f1.loc[adata.obs_names, 'niche']

    adata.obs['cx'] = anno_f1.loc[adata.obs_names, 'CenterX_local_px']
    adata.obs['cy'] = anno_f1.loc[adata.obs_names, 'CenterY_local_px']

    adata.obs['cx_g'] = anno_f1.loc[adata.obs_names, 'CenterX_global_px']
    adata.obs['cy_g'] = anno_f1.loc[adata.obs_names, 'CenterY_local_px']

    df = pd.DataFrame(index=adata.obs.index)
    df['cx'] = adata.obs['cx']
    df['cy'] = adata.obs['cy']
    arr = df.to_numpy()
    adata.obsm['spatial'] = arr

    df = pd.DataFrame(index=adata.obs.index)
    df['cx_g'] = adata.obs['cx_g']
    df['cy_g'] = adata.obs['cy_g']
    arr = df.to_numpy()
    adata.obsm['spatial_global'] = arr

    # df = pd.DataFrame(anno_f1, columns=['cell_type', 'niche'])
    # # anno_f1 = anno_f1.set_index('cell_ID').reindex(idx)
    # gene_f1['cell_type'] = anno_f1['cell_type']
    # gene_f1['niche'] = anno_f1['niche']
    # gene_f1 = gene_f1.dropna(axis=0, how='any')
    # # df = pd.DataFrame(gene_f1, columns=['cell_type', 'niche'])
    # gene_f1 = gene_f1.drop(columns=['cell_type', 'niche'])


    # adata = AD.AnnData(gene_f1)
    # anno_f1.index = anno_f1.index.map(str)
    # adata.obs['cell_type'] = anno_f1.loc[adata.obs_names, 'cell_type']
    # adata.obs['niche'] = anno_f1.loc[adata.obs_names, 'niche']
    # adata.obs['cx'] = anno_f1.loc[adata.obs_names, 'CenterX_local_px']
    # adata.obs['cy'] = anno_f1.loc[adata.obs_names, 'CenterY_local_px']

    dicts = {}

    dicts['T CD8 memory'] = 'lymphocyte'
    dicts['T CD8 naive'] = 'lymphocyte'
    dicts['T CD4 naive'] = 'lymphocyte'
    dicts['T CD4 memory'] = 'lymphocyte'
    dicts['Treg'] = 'lymphocyte'
    dicts['B-cell'] = 'lymphocyte'
    dicts['plasmablast'] = 'lymphocyte'
    dicts['NK'] = 'lymphocyte'


    dicts['monocyte'] = 'Mcell'
    dicts['macrophage'] = 'Mcell'
    dicts['mDC'] = 'Mcell'
    dicts['pDC'] = 'Mcell'


    dicts['tumors'] = 'tumors'


    dicts['epithelial'] = 'epithelial'

    dicts['mast'] = 'mast'
    dicts['endothelial'] = 'endothelial'

    dicts['fibroblast'] = 'fibroblast'

    dicts['neutrophil'] = 'neutrophil'
    adata.obs['merge_cell_type'] = np.zeros(adata.shape[0])
    for key, v in dicts.items():
        idx = (adata.obs['cell_type'] == key)
        adata.obs['merge_cell_type'][idx] = v
    adata.obs['merge_cell_type'] = adata.obs['merge_cell_type'].astype('category')

    adata.write(os.path.join(root, id, 'sampledata.h5ad'))

# if __name__ == '__main__':
def processing_nano():
    ids = [
        'fov1', 'fov2', 'fov3', 'fov4', 'fov5',
        'fov6', 'fov7', 'fov8', 'fov9', 'fov10',
        'fov11', 'fov12', 'fov13', 'fov14', 'fov15',
        'fov16', 'fov17', 'fov18', 'fov19', 'fov20'
    ]

    img_names = [
        'F001', 'F002', 'F003', 'F004', 'F005',
        'F006', 'F007', 'F008', 'F009', 'F010',
        'F011', 'F012', 'F013', 'F014', 'F015',
        'F016', 'F017', 'F018', 'F019', 'F020',
    ]

    fov = 1
    for id, imname in zip(ids, img_names):
        gen_h5ad(id, imname, fov)
        fov += 1
