# /usr/bin/bash
rm -rf ../result/grid_search_10x_grid_search5.txt


# for c_weight in $(seq 0.5 0.1 1.0)
# do
#     for g_weight in $(seq 0.5 0.1 1.0)
#     do
#         for i_weight in $(seq 0.5 0.1 1.0)
#         do 
#             python3 train_10x_ratio.py --ratio 0.3 --epochs 600 --id 151507 --c_weight $c_weight --g_weight $g_weight --i_weight $i_weight
#         done
#     done
# done


for g_weight in $(seq 1.0 0.2 1.1)
do
    for i_weight in $(seq 1.0 0.2 1.0)
    do 
        python3 train_10x_ratio.py --ratio 0.3 --epochs 600 --id 151507 --c_weight 1.0 --g_weight $g_weight --i_weight $i_weight --ncluster 7 --test_only 1 --save_path ../checkpoint/sigra_final 
        python3 train_10x_ratio.py --ratio 0.3 --epochs 600 --id 151508 --c_weight 1.0 --g_weight $g_weight --i_weight $i_weight --ncluster 7 --test_only 1 --save_path ../checkpoint/sigra_final 
        python3 train_10x_ratio.py --ratio 0.3 --epochs 600 --id 151509 --c_weight 1.0 --g_weight $g_weight --i_weight $i_weight --ncluster 7 --test_only 1 --save_path ../checkpoint/sigra_final 
        python3 train_10x_ratio.py --ratio 0.3 --epochs 600 --id 151510 --c_weight 1.0 --g_weight $g_weight --i_weight $i_weight --ncluster 7 --test_only 1 --save_path ../checkpoint/sigra_final 
        python3 train_10x_ratio.py --ratio 0.3 --epochs 600 --id 151669 --c_weight 1.0 --g_weight $g_weight --i_weight $i_weight --ncluster 5 --test_only 1 --save_path ../checkpoint/sigra_final 
        python3 train_10x_ratio.py --ratio 0.3 --epochs 600 --id 151670 --c_weight 1.0 --g_weight $g_weight --i_weight $i_weight --ncluster 5 --test_only 1 --save_path ../checkpoint/sigra_final 
        python3 train_10x_ratio.py --ratio 0.3 --epochs 600 --id 151671 --c_weight 1.0 --g_weight $g_weight --i_weight $i_weight --ncluster 5 --test_only 1 --save_path ../checkpoint/sigra_final 
        python3 train_10x_ratio.py --ratio 0.3 --epochs 600 --id 151672 --c_weight 1.0 --g_weight $g_weight --i_weight $i_weight --ncluster 5 --test_only 1 --save_path ../checkpoint/sigra_final 
        python3 train_10x_ratio.py --ratio 0.3 --epochs 600 --id 151673 --c_weight 1.0 --g_weight $g_weight --i_weight $i_weight --ncluster 7 --test_only 1 --save_path ../checkpoint/sigra_final 
        python3 train_10x_ratio.py --ratio 0.3 --epochs 600 --id 151674 --c_weight 1.0 --g_weight $g_weight --i_weight $i_weight --ncluster 7 --test_only 1 --save_path ../checkpoint/sigra_final 
        python3 train_10x_ratio.py --ratio 0.3 --epochs 600 --id 151675 --c_weight 1.0 --g_weight $g_weight --i_weight $i_weight --ncluster 7 --test_only 1 --save_path ../checkpoint/sigra_final 
        python3 train_10x_ratio.py --ratio 0.3 --epochs 600 --id 151676 --c_weight 1.0 --g_weight $g_weight --i_weight $i_weight --ncluster 7 --test_only 1 --save_path ../checkpoint/sigra_final 
    done
done

