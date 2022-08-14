# SiGra: Image guided spatial transcriptomics clustering and denoising using graph transformer

## Installation
SiGra is implemented in the pytorch framework   
tested on: (Ubuntu 18.04, 2080TI GPU, Intel i9-9820, 3.30GHZ, 20 core, 64 GB)  
Please run scGIT on CUDA environment(cuda 11.2)  

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
Install requirements and SiGra:

```bash
python setup.py install
```

## Dataset Setting
### NanoString CosMx SMI 
The dataset can be download [here](https://nanostring.com/products/cosmx-spatial-molecular-imager/ffpe-dataset/)
### Vizgen MERSCOPE 
The dataset can be download [here](https://info.vizgen.com/mouse-liver-access)
### 10x Visium 
The dataset can be download [here](https://github.com/LieberInstitute/HumanPilot/)

### Data folder structure
SiGra
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


### Tutorial for SiGra
1. Data processing: [here](https://github.com/QSong-github/SiGra/blob/main/Tutorials/SiGra_preprocess.ipynb)
2. Run SiGra: [here](https://github.com/QSong-github/SiGra/blob/main/Tutorials/SiGra_train.ipynb)
3. Output data visualization: [here](https://github.com/QSong-github/SiGra/blob/main/Tutorials/SiGra_visualize.ipynb)

## Quick Start

### Download our intermeidate results and checkpoints for inference
download the [checkpoints](https://purdue0-my.sharepoint.com/:u:/g/personal/tang385_purdue_edu/EZnAbrQm59dPtGKtSgSUBDABGGW86kh3ur6zZ2e-hVFWXQ?e=MWlkwB) and put them into the above roots.

### Reproduction instructions
1. for NanoString CosMx dataset
The results will be stored in "/path/siGra/results/nanostring/"
```
python3 train_nanostring.py --test_only 1 --save_path ../checkpoint/nanostring_final/ --pretrain final.pth
```

2. for Vizgen MERSCOPE dataset
The reuslts will be stored in /path/siGra/reuslts/merscope/
```
python3 train_merscope.py --test_only 1 --save_path ../checkpoint/merscope_final/ --pretrain final.pth
```

3. for 10x Visium dataset
The results will be stored in "/path/siGra/results/10x_final/"
```
python3 -W ignore train_visium.py --test_only 1 --save_path ../checkpoint/10x_final/ --id 151507 --ncluster 7
```
or 
```
bash test_10x.sh
```

## Train from scratch

### Training tutorials

1. for NanoString CosMx dataset
The hyperparameters were manually selected in individual datasets
```
e.g.
python3 train.py --data nanostring --test_only 0 --save_path ../checkpoint/nanostring_train/ --seed 1234 --epochs 900 --lr 1e-3 
```

2. for Vizgen MERSCOPE dataset
```
e.g.
python3 train.py --data merscope --test_only 0 --save_path ../checkpoint/merscope_train/ --seed 1234 --epochs 1000 --lr 1e-3 
```


3. for 10x Visium dataset
```
e.g.
python3 train.py --data visium --test_only 0 --save_path ../checkpoint/10x_train/ --seed 1234 --epochs 600 --lr 1e-3 --id 151507 --ncluster 7 --repeat 1
```

