# /usr/bin/bash
rm -rf ../result/pruning_nano_parameter4.txt
# for dim1 in 128 256 512 1024
# do
#     for dim2 in 20 30 40 50
#     do 
#         python3 train_nano_single_slice.py --test_only 1 --h_dim1 $dim1 --h_dim2 $dim2 --num_fov 20 --save_path ../checkpoint/nanostring_Lung9_Rep1/ --root ../dataset/nanostring --id lung9-1 --root ../dataset/nanostring
#     done
# done
python3 train_nano_single_slice.py --test_only 0 --h_dim1 128 --h_dim2 20 --num_fov 20 --save_path ../checkpoint/nano_emb/128_20/ --root ../dataset/nanostring --id lung9-1 --root ../dataset/nanostring
python3 train_nano_single_slice.py --test_only 0 --h_dim1 128 --h_dim2 30 --num_fov 20 --save_path ../checkpoint/nano_emb/128_30/ --root ../dataset/nanostring --id lung9-1 --root ../dataset/nanostring
python3 train_nano_single_slice.py --test_only 0 --h_dim1 128 --h_dim2 40 --num_fov 20 --save_path ../checkpoint/nano_emb/128_40/ --root ../dataset/nanostring --id lung9-1 --root ../dataset/nanostring
python3 train_nano_single_slice.py --test_only 0 --h_dim1 128 --h_dim2 50 --num_fov 20 --save_path ../checkpoint/nano_emb/128_50/ --root ../dataset/nanostring --id lung9-1 --root ../dataset/nanostring
python3 train_nano_single_slice.py --test_only 0 --h_dim1 256 --h_dim2 20 --num_fov 20 --save_path ../checkpoint/nano_emb/256_20/ --root ../dataset/nanostring --id lung9-1 --root ../dataset/nanostring
python3 train_nano_single_slice.py --test_only 0 --h_dim1 256 --h_dim2 30 --num_fov 20 --save_path ../checkpoint/nano_emb/256_30/ --root ../dataset/nanostring --id lung9-1 --root ../dataset/nanostring
python3 train_nano_single_slice.py --test_only 0 --h_dim1 256 --h_dim2 40 --num_fov 20 --save_path ../checkpoint/nano_emb/256_40/ --root ../dataset/nanostring --id lung9-1 --root ../dataset/nanostring
python3 train_nano_single_slice.py --test_only 0 --h_dim1 256 --h_dim2 50 --num_fov 20 --save_path ../checkpoint/nano_emb/256_50/ --root ../dataset/nanostring --id lung9-1 --root ../dataset/nanostring
python3 train_nano_single_slice.py --test_only 0 --h_dim1 512 --h_dim2 20 --num_fov 20 --save_path ../checkpoint/nano_emb/512_20/ --root ../dataset/nanostring --id lung9-1 --root ../dataset/nanostring
python3 train_nano_single_slice.py --test_only 0 --h_dim1 512 --h_dim2 30 --num_fov 20 --save_path ../checkpoint/nano_emb/512_30/ --root ../dataset/nanostring --id lung9-1 --root ../dataset/nanostring
python3 train_nano_single_slice.py --test_only 0 --h_dim1 512 --h_dim2 40 --num_fov 20 --save_path ../checkpoint/nano_emb/512_40/ --root ../dataset/nanostring --id lung9-1 --root ../dataset/nanostring
python3 train_nano_single_slice.py --test_only 0 --h_dim1 512 --h_dim2 50 --num_fov 20 --save_path ../checkpoint/nano_emb/512_50/ --root ../dataset/nanostring --id lung9-1 --root ../dataset/nanostring
python3 train_nano_single_slice.py --test_only 0 --h_dim1 1024 --h_dim2 20 --num_fov 20 --save_path ../checkpoint/nano_emb/1024_20/ --root ../dataset/nanostring --id lung9-1 --root ../dataset/nanostring
python3 train_nano_single_slice.py --test_only 0 --h_dim1 1024 --h_dim2 30 --num_fov 20 --save_path ../checkpoint/nano_emb/1024_30/ --root ../dataset/nanostring --id lung9-1 --root ../dataset/nanostring
python3 train_nano_single_slice.py --test_only 0 --h_dim1 1024 --h_dim2 40 --num_fov 20 --save_path ../checkpoint/nano_emb/1024_40/ --root ../dataset/nanostring --id lung9-1 --root ../dataset/nanostring
python3 train_nano_single_slice.py --test_only 0 --h_dim1 1024 --h_dim2 50 --num_fov 20 --save_path ../checkpoint/nano_emb/1024_50/ --root ../dataset/nanostring --id lung9-1 --root ../dataset/nanostring

