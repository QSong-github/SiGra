lr=1e-3
epoch=600
seed=1234
repeat=1
sp='../checkpoint/transformer_ss'

python3 train.py --lr $lr --epochs $epoch --id 151676 --seed $seed --repeat $repeat --ncluster 7 --save_path $sp --dataset 10x --cluster_method mclust
python3 train.py --lr $lr --epochs $epoch --id 151675 --seed $seed --repeat $repeat --ncluster 7 --save_path $sp --dataset 10x --cluster_method mclust
python3 train.py --lr $lr --epochs $epoch --id 151674 --seed $seed --repeat $repeat --ncluster 7 --save_path $sp --dataset 10x --cluster_method mclust
python3 train.py --lr $lr --epochs $epoch --id 151673 --seed $seed --repeat $repeat --ncluster 7 --save_path $sp --dataset 10x --cluster_method mclust
python3 train.py --lr $lr --epochs $epoch --id 151672 --seed $seed --repeat $repeat --ncluster 5 --save_path $sp --dataset 10x --cluster_method mclust
python3 train.py --lr $lr --epochs $epoch --id 151671 --seed $seed --repeat $repeat --ncluster 5 --save_path $sp --dataset 10x --cluster_method mclust
python3 train.py --lr $lr --epochs $epoch --id 151670 --seed $seed --repeat $repeat --ncluster 5 --save_path $sp --dataset 10x --cluster_method mclust
python3 train.py --lr $lr --epochs $epoch --id 151669 --seed $seed --repeat $repeat --ncluster 5 --save_path $sp --dataset 10x --cluster_method mclust
python3 train.py --lr $lr --epochs $epoch --id 151510 --seed $seed --repeat $repeat --ncluster 7 --save_path $sp --dataset 10x --cluster_method mclust
python3 train.py --lr $lr --epochs $epoch --id 151509 --seed $seed --repeat $repeat --ncluster 7 --save_path $sp --dataset 10x --cluster_method mclust
python3 train.py --lr $lr --epochs $epoch --id 151508 --seed $seed --repeat $repeat --ncluster 7 --save_path $sp --dataset 10x --cluster_method mclust
python3 train.py --lr $lr --epochs $epoch --id 151507 --seed $seed --repeat $repeat --ncluster 7 --save_path $sp --dataset 10x --cluster_method mclust

