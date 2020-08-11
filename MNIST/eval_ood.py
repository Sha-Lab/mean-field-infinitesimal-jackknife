import torch
import torch.nn.functional as F
import numpy as np
import argparse
from pathlib import Path
from ipdb import launch_ipdb_on_exception
import json
from mnist_utils import gen_model_dir, MLP, load_mnist_data, \
    load_notmnist, get_penultimate_features, viz_temperature_heatmap

import sys
sys.path.append('../')
from Utils.hess_utils import compute_hess, invert_hess
from Utils.inference_utils import model_feedforward
from Utils.metric_utils import ood_metric
from Utils.mfij_utils import mfij_predict

def eval_ood(args, data_key):
    model_dir = gen_model_dir(args)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # define model
    model = MLP(args.num_hiddens, drop_rate=0.0)
    model.to(device)
    model.eval()
    # load best model
    with open(Path(model_dir, "model"), 'rb') as f:
        params = torch.load(f)
        model.load_state_dict(params['model_weight'])

    ## OOD data is from NotMNIST datatset
    assert args.notset == 'notmnist'

    if data_key == 'test':
        _, _, eval_dl = load_mnist_data(args, train_shuffle=False)
        print('load test data')
        notmnist_dl = load_notmnist(args, heldout=False)
    elif data_key == 'heldout':
        _, eval_dl, _ = load_mnist_data(args, train_shuffle=False)
        print('load heldout data for OOD eval')
        notmnist_dl = load_notmnist(args, heldout=True)

    ## MLE predict
    in_mle_logits, _ = model_feedforward(model, eval_dl, device)
    in_mle_probs = F.softmax(in_mle_logits, dim=1).cpu().detach().numpy()

    out_mle_logits, _ = model_feedforward(model, notmnist_dl, device)
    out_mle_probs = F.softmax(out_mle_logits, dim=1).cpu().detach().numpy()
    del in_mle_logits, out_mle_logits

    # get penultimate layer features
    in_pen_features = get_penultimate_features(args, eval_dl)
    out_pen_features = get_penultimate_features(args, notmnist_dl)

    probs_dict = mfij_predict(args,
                              loaders={data_key: in_pen_features,
                                       args.notset: out_pen_features},
                              dimensions=[args.num_hiddens[-1], args.num_classes],
                              num_train=55000,
                              block_size=500)

    in_inf_probs = probs_dict[data_key]
    out_inf_probs = probs_dict[args.notset]

    # softmax OOD detection
    in_mle_stats = {'prob': np.amax(in_mle_probs, axis=1)}
    out_mle_stats = {'prob': np.amax(out_mle_probs, axis=1)}
    res_mle = ood_metric(in_mle_stats, out_mle_stats, stypes=['prob'], verbose=False)['prob']

    # mfij OOD detection
    in_inf_stats = {'prob': np.amax(in_inf_probs, axis=1)}
    out_inf_stats = {'prob': np.amax(out_inf_probs, axis=1)}
    res_inf = ood_metric(in_inf_stats, out_inf_stats, stypes=['prob'], verbose=False)['prob']

    print("& mle & {:6.3f} & {:6.3f} & {:6.3f}/{:6.3f} & {:6.3f} \\\\".format(
        res_mle['DTACC'] * 100, res_mle['AUROC'] * 100, res_mle['AUIN'] * 100,
        res_mle['AUOUT'] * 100, res_mle['TNR'] * 100, ))
    print(" ==================================== ")
    print("& mfij & {:6.3f} & {:6.3f} & {:6.3f}/{:6.3f} & {:6.3f} \\\\".format(
        res_inf['DTACC'] * 100, res_inf['AUROC'] * 100, res_inf['AUIN'] * 100,
        res_inf['AUOUT'] * 100, res_inf['TNR'] * 100, ))
    print(" ==================================== ")
    print("\n")
    return in_mle_probs, out_mle_probs, in_inf_probs, out_inf_probs, res_inf

def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--notset', type=str, default="notmnist")
    ## uncertainty estimate args
    parser.add_argument('--temp_ens', type=float, default=1.0)
    parser.add_argument('--temp_act', type=float, default=1.0)
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

    notmnist_res = dict()
    best_auc = 0
    ensemble_temperature_list = np.arange(-4, 3, 1, dtype=float)
    activation_temperature_list = np.array([0.001, 0.25, 0.5, 0.75, 1, 1.5, 2, 5])

    for i, ens_T in enumerate(ensemble_temperature_list):
        args.temp_ens = float(10**ens_T)
        for j, act_T in enumerate(activation_temperature_list):
            args.temp_act = act_T

            print('temperatures:', args.temp_ens, args.temp_act)
            _, _, _, _, res_mfij = eval_ood(args, data_key='heldout')
            notmnist_res[str(ens_T) + '_' + str(act_T)] = res_mfij
            if res_mfij['AUROC'] > best_auc:
                best_auc = res_mfij['AUROC']
                best_ts = [ens_T, act_T]

    notmnist_res['best_auc'] = best_auc
    notmnist_res['best_ts'] = best_ts
    print('eval best temperature on test', best_ts)
    args.temp_ens = float(10 ** best_ts[0])
    args.temp_act = best_ts[1]
    _, _, _, _, test_mfij = eval_ood(args, data_key='test')
    notmnist_res['best_test_ood'] = test_mfij

    # save result
    model_dir = gen_model_dir(args)
    tune_fn = args.notset + '_mfij.json'
    with open(Path(model_dir, tune_fn), 'w') as fp:
        json.dump(notmnist_res, fp)

    # visualize temperature heatmap
    fig = viz_temperature_heatmap(notmnist_res,
                                  ensemble_temperature_list=list(ensemble_temperature_list),
                                  activation_temperature_list=activation_temperature_list,
                                  ensemble_temperature_label=['1e-4', '1e-3', '1e-2', '1e-1', '1.', '1e1', '1e2'],
                                  task='ood')
    fig.savefig("figs/mnist_mf_auc_temp.eps", dpi=fig.dpi, bbox_inches='tight', format='eps')


if __name__ == '__main__':
    with launch_ipdb_on_exception():
        main()