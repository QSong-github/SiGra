import argparse
from train_visium import train_10x
from train_nanostring import train_nano
from train_merscope import train_mscope

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='nanostring', help='should be nanostring, 10x or merscope')


    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--root', type=str, default='../dataset/nanostring')
    parser.add_argument('--epochs', type=int, default=2000)
    parser.add_argument('--id', type=str, default='fov1')
    parser.add_argument('--img_name', type=str, default='F001')
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--save_path', type=str, default='../checkpoint/nanostring_final')
    parser.add_argument('--ncluster', type=int, default=8)
    parser.add_argument('--repeat', type=int, default=1)
    parser.add_argument('--use_gray', type=float, default=0)
    parser.add_argument('--test_only', type=int, default=0)
    parser.add_argument('--pretrain', type=str, default='final.pth')
    parser.add_argument('--cluster_method', type=str, default='leiden', help='leiden or mclust')

    opt = parser.parse_args()


    if opt.dataset == 'nanostring':
        train_nano(opt)

    elif opt.dataset == '10x':
        train_10x(opt)

    elif opt.dataset == 'merscope':
        train_mscope(opt)