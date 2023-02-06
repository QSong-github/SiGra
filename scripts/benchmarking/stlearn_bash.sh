# /usr/bin/bash
python3 run_stlearn.py --root ../dataset/nanostring_Lung5_Rep1/ --num_fov 30 --save_path ./stlearn/lung5-1/ --epochs 1000
python3 run_stlearn.py --root ../dataset/nanostring_Lung5_Rep2/ --num_fov 30 --save_path ./stlearn/lung5-2/ --epochs 1000
python3 run_stlearn.py --root ../dataset/nanostring_Lung5_Rep3/ --num_fov 30 --save_path ./stlearn/lung5-3/ --epochs 1000
python3 run_stlearn.py --root ../dataset/nanostring_Lung6/ --num_fov 30 --save_path ./stlearn/lung6/ --epochs 1000
python3 run_stlearn.py --root ../dataset/nanostring --num_fov 20 --save_path ./stlearn/lung9-1/ --epochs 1000
python3 run_stlearn.py --root ../dataset/nanostring_Lung9_Rep2/ --num_fov 45 --save_path ./stlearn/lung9-2/ --epochs 1000
python3 run_stlearn.py --root ../dataset/nanostring_Lung12/ --num_fov 28 --save_path ./stlearn/lung12/ --epochs 1000
python3 run_stlearn.py --root ../dataset/nanostring_Lung13/ --num_fov 20 --save_path ./stlearn/lung13/ --epochs 1000