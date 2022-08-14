import numpy as np
from skimage.io import imread
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import tifffile
import h5py
import os
from tqdm import tqdm
import scanpy as sc
import anndata as ad


def normal_images(image_root):
    image = tifffile.imread(image_root)
    return image.min(), image.max()

def check_images(name):
    if os.path.exists(os.path.join('cut_images', 'bound1', name+'_zIndex_3.tif')):
        return True
    else:
        return False

def build_ann(gene_df, meta_df, transformation_matrix, fov, z_index='zIndex_3'):
    h5file = 'cell_boundaries/feature_data_%d.hdf5'%(fov)
    if not os.path.exists(h5file):
        return
    cellBound = h5py.File(h5file)

    names = meta_df[meta_df.fov == fov].index

    selected_names = []
    for n in names.tolist():
        if check_images(n):
            selected_names.append(n)

    gene_sub = gene_df.loc[selected_names]
    adata = ad.AnnData(gene_sub, dtype='float64')

    globalx, globaly = [], []
    imgs = []

    for name in selected_names:
        temp = cellBound['featuredata'][str(name)][z_index]['p_0']['coordinates'][0]
        boundaryPolygon = np.ones((temp.shape[0], temp.shape[1] + 1))
        boundaryPolygon[:, :-1] = temp
        transformedBoundary = np.matmul(transformation_matrix, np.transpose(boundaryPolygon))[:-1]
        cy = (transformedBoundary[1].min() + transformedBoundary[1].max()) / 2
        cx = (transformedBoundary[0].min() + transformedBoundary[0].max()) / 2

        globaly.append(cy)
        globalx.append(cx)

    adata.obs['globalx'] = np.array(globalx)
    adata.obs['globaly'] = np.array(globaly)
    df = pd.DataFrame(index=adata.obs.index)
    df['gx'] = np.array(globalx)
    df['gy'] = np.array(globaly)
    arr = df.to_numpy()
    adata.obsm['spatial'] = arr

    if not os.path.exists('../dataset/merscope/data'):
        os.makedirs('../dataset/merscope/data')
    adata.write('../dataset/merscope/data/fov_%d.h5ad'%(fov))


def draw_all_fovs(image, transformation_matrix, fov, z_index='zIndex_3', btype='DAPI'):
    plt.close('all')
    h5file = '../dataset/merscope/cell_boundaries/feature_data_%d.hdf5'%(fov)
    if not os.path.exists(h5file):
        return
    cellBound = h5py.File(h5file)
    meta_cell = pd.read_csv('../dataset/merscope/Liver1Slice1_cell_metadata.csv', index_col=0)
    meta_cell = meta_cell[meta_cell.fov == fov]

    currentCells = []
    for inst_cell in meta_cell.index.tolist():
        temp = cellBound['featuredata'][str(inst_cell)][z_index]['p_0']['coordinates'][0]
        boundaryPolygon = np.ones((temp.shape[0], temp.shape[1] + 1))
        boundaryPolygon[:, :-1] = temp
        transformedBoundary = np.matmul(transformation_matrix, np.transpose(boundaryPolygon))[:-1]
        currentCells.append(transformedBoundary)

        cy = (transformedBoundary[1].min() + transformedBoundary[1].max()) / 2
        cx = (transformedBoundary[0].min() + transformedBoundary[0].max()) / 2
        w, h = 100, 100
        # print(inst_cell, cy-h, cy+h, cx-w, cx+w)
        if cy-h < 0 or cy+h >= image.shape[0] or cx-w < 0 or cx+w >= image.shape[1]:
            continue
        sub_img = image[int(cy-h):int(cy+h)+1, int(cx-w):int(cx+w)+1]
        cv2.imwrite(os.path.join('../dataset/merscope/cut_images/%s'%(btype), inst_cell+'_%s.tif')%(z_index), sub_img)

def save_image(root, transform_root, btype='DAPI'):
    print(btype)
    image = tifffile.imread(root)
    transformation_matrix = pd.read_csv(transform_root, header=None, sep=' ').values
    for i in tqdm(range(0, 1797)):
        draw_all_fovs(image, transformation_matrix, i, btype=btype)


def merge_fovs():
    adatas_lists = os.listdir('../dataset/merscope/data')
    adatas = []
    for name in adatas_lists:
        adata = sc.read(os.path.join('../dataset/merscope/data', name))
        # print(adata)
        adatas.append(adata)

    all_data = ad.concat(adatas)
    all_data.write('../dataset/merscope/allfovs.h5ad')

def split_fovs():
    txt = open('../dataset/merscope/fov_map.txt', 'w')
    all_data = sc.read('../dataset/merscope/allfovs.h5ad')
    gx = all_data.obs['globalx']
    gy = all_data.obs['globaly']

    gxmax = int(gx.max()) + 1
    gxmin = int(gx.min()) - 1

    gymax = int(gy.max()) + 1
    gymin = int(gy.min()) - 1

    xinterval = (gxmax - gxmin) // 10 + 1
    yinterval = (gxmax - gxmin) // 10 + 1

    xstart = gx.min()
    xend = xstart + xinterval

    fov = 0
    row = 0

    for xstart in range(gxmin, gxmax, xinterval):
        col = 0
        for ystart in range(gymin, gymax, yinterval):
            idx = (all_data.obs['globalx'] >= xstart) & (all_data.obs['globalx'] <= (xstart+xinterval)) \
            & (all_data.obs['globaly'] >= ystart) & (all_data.obs['globaly'] <= (ystart+xinterval))
            names = (all_data.obs.loc[idx]).index
            sub_adata = all_data[names]
            # print(fov, sub_adata.shape[0])
            plt.close('all')
            x, y = sub_adata.obs['globalx'], sub_adata.obs['globaly']
            plt.scatter(x, y, c='blue', s=1)
            plt.axis('off')
            plt.xlim(xstart, xstart+xinterval)
            plt.ylim(ystart+yinterval, ystart)
            plt.savefig('../dataset/merscope/fov_images/fov_%d_%d.png'%(row, col), bbox_inches='tight')
        
            if sub_adata.shape[0] > 100:
                sub_adata.write('../dataset/merscope/sample_data/fov_%d.h5ad'%(fov))
                txt.write('fov: %d, row: %d, col: %d\n'%(fov, row, col))
                fov += 1
            col += 1
        row += 1

def draw_spatial():
    combines = []
    for col in range(0, 10):
        imgs = []
        for row in range(0, 10):
            img = cv2.imread('../dataset/merscope/fov_images/fov_%d_%d.png'%(row, col))
            imgs.append(img)
        combine =  cv2.hconcat(imgs)
        combines.append(combine)

    final = cv2.vconcat(combines)
    cv2.imwrite('../dataset/merscope/combine.png', final)

# if __name__ == '__main__':
def processing_mscope():
    # step 1: cut images into image patchs
    # this may takes about 3 hours 
    save_image(root='../dataset/merscope/Liver1Slice1_images_mosaic_DAPI_z3.tif', transform_root='../dataset/merscope/Liver1Slice1_images_micron_to_mosaic_pixel_transform.csv', btype='DAPI')
    save_image(root='../dataset/merscope/Liver1Slice1_images_mosaic_Cellbound1_z3.tif', transform_root='../dataset/merscope/Liver1Slice1_images_micron_to_mosaic_pixel_transform.csv', btype='bound1')
    save_image(root='../dataset/merscope/Liver1Slice1_images_mosaic_Cellbound2_z3.tif', transform_root='../dataset/merscope/Liver1Slice1_images_micron_to_mosaic_pixel_transform.csv', btype='bound2')
    save_image(root='../dataset/merscope/Liver1Slice1_images_mosaic_Cellbound3_z3.tif', transform_root='../dataset/merscope/Liver1Slice1_images_micron_to_mosaic_pixel_transform.csv', btype='bound3')

    # step2: 
    # build anndata for default fovs
    df = pd.read_csv('../dataset/merscope/Liver1Slice1_cell_by_gene.csv', header=0, index_col=0)
    meta = pd.read_csv('../dataset/merscope/Liver1Slice1_cell_metadata.csv', header=0, index_col=0)
    transform_root='../dataset/merscope/Liver1Slice1_images_micron_to_mosaic_pixel_transform.csv'
    transformation_matrix = pd.read_csv(transform_root, header=None, sep=' ').values
    for i in tqdm(range(0, 1796)):
        build_ann(df, meta, transformation_matrix, i)

    # step3:
    # merge all the fovs into one anndata
    merge_fovs()

    # step4:
    # split the all h5ad files into larger fovs
    split_fovs()

    # step5:
    # draw spatials to check
    draw_spatial()