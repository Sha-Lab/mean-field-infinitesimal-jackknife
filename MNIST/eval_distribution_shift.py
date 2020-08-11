import torch
import numpy as np
import argparse
from pathlib import Path
import json
from ipdb import launch_ipdb_on_exception

from mnist_utils import gen_model_dir, load_mnist_data, \
    get_penultimate_features, viz_distribution_shift
from eval_in_domain import eval_indomain
import sys
sys.path.append('../')
from Utils.hess_utils import compute_hess, invert_hess

def parse_args(args=None):
    parser = argparse.ArgumentParser()
    ## rotation args
    parser.add_argument('--rotate_degs', type=int, default=0)

    ## uncertainty estimate args
    parser.add_argument('--temp_ens', type=float, default=0.001) # use the best temperatures found on MNIST heldout set
    parser.add_argument('--temp_act', type=float, default=2.0) # use the best temperatures found on MNIST heldout set
    parser.add_argument('--mf_approx', type=str, default='mf0')

    ## model training args, use the same values as in train.py file
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--model_path', type=str, default="model/")
    parser.add_argument('--model_str', type=str, default="mlp")
    parser.add_argument('--num_hiddens', nargs=2, type=int, default=[256, 256])
    parser.add_argument('--drop_rate', type=float, default=0.1)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--lr_decay', type=float, default=0.998)
    parser.add_argument('--n_epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--weight_decay', type=float, default=0)
    parser.add_argument('--num_features', type=int, default=784)
    parser.add_argument('--num_classes', type=int, default=10)

    args = parser.parse_args(args)
    return args

def main(args=None):
    args = parse_args(args)

    print('processing model: lr' + str(args.lr) + '_d' + str(args.lr_decay) + '_bs' + str(args.batch_size),
          'drop rate', args.drop_rate, 'seed', args.seed)

    # mfij related argument
    args.lambda0 = '3/(np.pi**2)'
    # Compute Hessian covariance matrix once
    _bs = args.batch_size
    args.batch_size = 5000
    train_dl, _, _ = load_mnist_data(args, train_shuffle=False)
    args.batch_size = _bs
    train_pen_features = get_penultimate_features(args, train_dl)
    hess = compute_hess(train_pen_features)
    args.cov = torch.Tensor(invert_hess(hess))
    del train_dl, train_pen_features

    shift_res = dict()
    for deg in np.arange(0, 181, 15):
        args.rotate_degs = deg
        if args.rotate_degs > 0:
            err, nll, ECE = eval_indomain(args, data_key='test_rotate')
        else:
            err, nll, ECE = eval_indomain(args, data_key='test')
        shift_res['rotate_' + str(deg)] = dict(err=err,
                                               nll=nll,
                                               ece=ECE)
    # save result
    model_dir = gen_model_dir(args)
    shift_fn = 'shift_Te' + str(args.temp_ens) + '_Ta' + str(args.temp_act) + '.json'
    with open(Path(model_dir, shift_fn), 'w') as fp:
        json.dump(shift_res, fp)

    # visualize ece across different degrees
    fig = viz_distribution_shift(shift_res)
    fig.savefig("figs/mnist_shift_ece.eps", dpi=fig.dpi, bbox_inches='tight', format='eps')

if __name__ == '__main__':
    with launch_ipdb_on_exception():
        main()
