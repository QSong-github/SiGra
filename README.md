# SiGra: Single-cell spatial elucidation through image-augmented graph transformer

SiGra is built based on pytorch
Test on: Ubuntu 18.04, 2080TI GPU, Intel i9-9820, 3.30GHZ, 20 core, 64 GB, CUDA environment(cuda 11.2)

## Requirements
Required modules can be installed via requirements.txt under the project root
```
pip install -r requirements.txt
```

```
torchvision==0.11.1
matplotlib==2.1.1
torch==1.6.0
seaborn==0.10.0
tqdm==4.47.0
numpy==1.13.3
anndata==0.8.0
pandas==1.4.3
rpy2==3.5.2
scanpy==1.9.1
scipy==1.8.1
scikit_learn==1.1.1
torch_geometric==2.0.4
```
## Installation

Download SiGra:
```
git clone https://github.com/QSong-github/SiGra
```

## Dataset Setting
### NanoString CosMx SMI 
The dataset can be download [here](https://nanostring.com/products/cosmx-spatial-molecular-imager/ffpe-dataset/)
### Vizgen MERSCOPE 
The dataset can be download [here](https://info.vizgen.com/mouse-liver-access)
### 10x Visium 
The dataset can be download [here](https://github.com/LieberInstitute/HumanPilot/)

#### you can download our processed dataset [here](https://purdue0-my.sharepoint.com/:f:/g/personal/tang385_purdue_edu/EoJcJv8OZHRIhLyplj5r1PABW-UQfD1p1YU00gAdZNeK7A?e=K3Mmqg)  

## Data folder structure

```
├── requirement.txt
├── dataset
│   └── DLPFC
│        └── 151507
│              ├── filtered_feature_bc_matrix.h5
│              ├── metadata.tsv 
│              ├── sampledata.h5ad
│              └── spatial
│                     ├── tissue_positions_list.csv  
│                     ├── full_image.tif  
│                     ├── tissue_hires_image.png  
│                     ├── tissue_lowres_image.png
│   └── nanostring
│        └── Lung9_Rep1_exprMat_file.csv
│        └── matched_annotation_all.csv
│        └── fov1
│              ├── CellComposite_F001.jpg
│              ├── sampledata.h5ad
│        └── fov2
│              ├── CellComposite_F002.jpg
│              ├── sampledata.h5ad
│   └── merscope
│        └── Cell_boundaries
│        └── Cut Images
│        └── sample_data
│        └── processed_data



├── checkpoint
│   └── nanostring_final
│        ├── final.pth
│   └── merscope_all
│        ├── final.pth
│   └── 10x_final
│        └── 151507
│              ├── final.pth
```


## Tutorial for SiGra
1. Data processing: [here](https://github.com/QSong-github/SiGra/blob/main/Tutorials/SiGra_preprocess.ipynb)
2. Run SiGra: [here](https://github.com/QSong-github/SiGra/blob/main/Tutorials/SiGra_train.ipynb)
3. Output data visualization: [here](https://github.com/QSong-github/SiGra/blob/main/Tutorials/SiGra_visualize.ipynb)

## processing scripts
```
# go to /path/to/Sigra
# for NanoString CosMx dataset
python3 processing.py --dataset nanostring

# for Vizgen MERSCOPE dataset
python3 processing.py --dataset merscope

# for 10x Visium dataset
python3 processing.py --dataset 10x
```

## Reproduction instructions
go to /path/to/SiGra/SiGra_model

Download the datasets and [checkpoints](https://purdue0-my.sharepoint.com/:f:/g/personal/tang385_purdue_edu/Em6J9c_VogROtFRebPBSgmwBk8TH0jYu1OTWm9hhfNWJVA?e=Azhljx) and put in folders as above.

#### 1. for NanoString CosMx dataset
The results will be stored in "/path/siGra/results/nanostring/"
```
python3 train.py --test_only 1 --save_path ../checkpoint/nanostring_final/ --pretrain final.pth --dataset nanostring
```

#### 2. for Vizgen MERSCOPE dataset
The reuslts will be stored in /path/siGra/reuslts/merscope/
```
python3 train.py --test_only 1 --save_path ../checkpoint/merscope_final/ --pretrain final.pth --dataset merscope --root ../dataset/mouseLiver
```

#### 3. for 10x Visium dataset
The results will be stored in "/path/siGra/results/10x_final/"
```
python3 train.py --test_only 1 --save_path ../checkpoint/10x_final/ --id 151507 --ncluster 7 --dataset 10x --root ../dataset/DLPFC
```
And you can use the bash scripts to test all slices:
```
sh test_visium.sh
```


## Train from scratch

### Training tutorials

#### 1. for NanoString CosMx dataset
The hyperparameters were manually selected in individual datasets
```
python3 train.py --dataset nanostring --test_only 0 --save_path ../checkpoint/nanostring_train/ --seed 1234 --epochs 900 --lr 1e-3 
```

#### 2. for Vizgen MERSCOPE dataset
```
python3 train.py --dataset merscope --test_only 0 --save_path ../checkpoint/merscope_train/ --seed 1234 --epochs 1000 --lr 1e-3 --root ../dataset/mouseLiver
```


#### 3. for 10x Visium dataset
```
python3 train.py --dataset 10x --test_only 0 --save_path ../checkpoint/10x_train/ --seed 1234 --epochs 600 --lr 1e-3 --id 151507 --ncluster 7 --repeat 1 --root ../dataset/DLPFC
```
And you can use the bash scripts to train all slices:
```
sh train_visium.sh
```

