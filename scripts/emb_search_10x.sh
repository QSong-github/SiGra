# /usr/bin/bash
rm -rf ../result/pruning_10x_parameter.txt

for dim1 in 128 256 512 1024
do
    for dim2 in 20 30 40 50
    do 
        python3 train_10x_para.py --ratio 0.3 --epochs 600 --id 151676  --ncluster 7 --test_only 0 --save_path ../checkpoint/sigra_emb_search --h_dim1 $dim1 --h_dim2 $dim2
    done
done

