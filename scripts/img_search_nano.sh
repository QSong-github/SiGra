# /usr/bin/bash
# rm -rf ../result/pruning_nano_parameter5.txt
# for dim1 in 128 256 512 1024
# do
#     for dim2 in 20 30 40 50
#     do 
#         python3 train_nano_single_slice.py --test_only 1 --h_dim1 $dim1 --h_dim2 $dim2 --num_fov 20 --save_path ../checkpoint/nanostring_Lung9_Rep1/ --root ../dataset/nanostring --id lung9-1 --root ../dataset/nanostring
#     done
# done
python3 train_nano_single_slice.py --test_only 1 --h_dim1 512 --h_dim2 30 --num_fov 20 --save_path ../checkpoint/nano_emb/mu5/ --root ../dataset/nanostring --id lung9-1 --root ../dataset/nanostring --suffix _mu_5
python3 train_nano_single_slice.py --test_only 1 --h_dim1 512 --h_dim2 30 --num_fov 20 --save_path ../checkpoint/nano_emb/mu10/ --root ../dataset/nanostring --id lung9-1 --root ../dataset/nanostring --suffix _mu_10
python3 train_nano_single_slice.py --test_only 1 --h_dim1 512 --h_dim2 30 --num_fov 20 --save_path ../checkpoint/nano_emb/mu20/ --root ../dataset/nanostring --id lung9-1 --root ../dataset/nanostring --suffix _mu_20
python3 train_nano_single_slice.py --test_only 1 --h_dim1 512 --h_dim2 30 --num_fov 20 --save_path ../checkpoint/nano_emb/mu30/ --root ../dataset/nanostring --id lung9-1 --root ../dataset/nanostring --suffix _mu_30
python3 train_nano_single_slice.py --test_only 1 --h_dim1 512 --h_dim2 30 --num_fov 20 --save_path ../checkpoint/nano_emb/mu40/ --root ../dataset/nanostring --id lung9-1 --root ../dataset/nanostring --suffix _mu_40
