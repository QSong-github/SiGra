# /usr/bin/bash

python3 train_nano_single_slice.py --test_only 1 --pretrain final_1000_0.pth --root ../dataset/nanostring_Lung5_Rep1 --save_path ../checkpoint/nanostring_Lung5_Rep1 --num_fov 30 --id lung5-1
python3 train_nano_single_slice.py --test_only 1 --pretrain final_1000_0.pth --root ../dataset/nanostring_Lung5_Rep2 --save_path ../checkpoint/nanostring_Lung5_Rep2 --num_fov 30 --id lung5-2
python3 train_nano_single_slice.py --test_only 1 --pretrain final_1000_0.pth --root ../dataset/nanostring_Lung5_Rep3 --save_path ../checkpoint/nanostring_Lung5_Rep3 --num_fov 30 --id lung5-3
python3 train_nano_single_slice.py --test_only 1 --pretrain final_1000_0.pth --root ../dataset/nanostring_Lung6 --save_path ../checkpoint/nanostring_Lung6 --num_fov 30 --id lung6
python3 train_nano_single_slice.py --test_only 1 --pretrain final_1000_0.pth --root ../dataset/nanostring_Lung9_Rep2 --save_path ../checkpoint/nanostring_Lung9_Rep2 --num_fov 45 --id lung9-2
python3 train_nano_single_slice.py --test_only 1 --pretrain final_1000_0.pth --root ../dataset/nanostring_Lung12 --save_path ../checkpoint/nanostring_Lung12 --num_fov 28 --id lung12
python3 train_nano_single_slice.py --test_only 1 --pretrain final_1000_0.pth --root ../dataset/nanostring_Lung13 --save_path ../checkpoint/nanostring_Lung13 --num_fov 20 --id lung13

python3 train_nano_single_slice.py --test_only 1 --pretrain final_600_0.pth --root ../dataset/nanostring_Lung9_Rep1 --save_path ../checkpoint/nanostring --num_fov 20 --id lung9-1
