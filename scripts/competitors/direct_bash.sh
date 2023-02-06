# /usr/bin/bash
Rscript run_bayesspace.R ../dataset/nanostring 20 3 ./direct/bayesspace lung9-1
python3 run_scanpy.py --root ../dataset/nanostring --num_fov 20 --save_path ./direct/scanpy/lung9-1 --epochs 1000 --ncluster 3
Rscript run_seurat.R ../dataset/nanostring 20 3 ./direct/seurat lung9-1
python3 run_stlearn.py --root ../dataset/nanostring --num_fov 20 --save_path ./direct/stlearn/lung9-1 --epochs 1000 --ncluster 3
python3 run_spagcn.py --root ../dataset/nanostring --num_fov 20 --save_path ./direct/spagcn/lung9-1/ --epochs 1000 --ncluster 3
python3 run_stagate.py --root ../dataset/nanostring --num_fov 20 --save_path ./direct/STAGATE/lung9-1/ --epochs 1000 --ncluster 3 


