# /usr/bin/bash
rm -rf ../result/pruning_nano_grid_search.txt
for g_weight in $(seq 0.1 0.2 1.0)
do
    for i_weight in $(seq 0.1 0.2 1.0)
    do 
        python3 grid_search_nano.py --test_only 1 --g_weight $g_weight --i_weight $i_weight --num_fov 20 --save_path ../checkpoint/nano_grid/ --root ../dataset/nanostring --id lung9-1 --root ../dataset/nanostring 
    done
done

