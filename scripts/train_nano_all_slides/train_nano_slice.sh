# /usr/bin/bash

python3 train_nano_single_slice.py --seed 1234 --epochs 1000 --num_fov 30 --root ../dataset/nanostring_Lung5_Rep1 --save_path ../checkpoint/nanostring_Lung5_Rep1
python3 train_nano_single_slice.py --seed 1234 --epochs 1000 --num_fov 30 --root ../dataset/nanostring_Lung5_Rep2 --save_path ../checkpoint/nanostring_Lung5_Rep2
python3 train_nano_single_slice.py --seed 1234 --epochs 1000 --num_fov 30 --root ../dataset/nanostring_Lung5_Rep3 --save_path ../checkpoint/nanostring_Lung5_Rep3
python3 train_nano_single_slice.py --seed 1234 --epochs 1000 --num_fov 30 --root ../dataset/nanostring_Lung6 --save_path ../checkpoint/nanostring_Lung6
python3 train_nano_single_slice.py --seed 1234 --epochs 1000 --num_fov 45 --root ../dataset/nanostring_Lung9_Rep2 --save_path ../checkpoint/nanostring_Lung9_Rep2
python3 train_nano_single_slice.py --seed 1234 --epochs 1000 --num_fov 28 --root ../dataset/nanostring_Lung12 --save_path ../checkpoint/nanostring_Lung12
python3 train_nano_single_slice.py --seed 1234 --epochs 1000 --num_fov 20 --root ../dataset/nanostring_Lung13 --save_path ../checkpoint/nanostring_Lung13
