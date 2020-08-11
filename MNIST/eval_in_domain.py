import torch
import torch.nn.functional as F
import numpy as np
import argparse
import json
from pathlib import Path
from ipdb import launch_ipdb_on_exception
from mnist_utils import gen_model_dir, load_mnist_data, load_mnist_shift, \
    MLP, get_penultimate_features, viz_temperature_heatmap

import sys
sys.path.append('../')
from Utils.hess_utils import compute_hess, invert_hess
from Utils.inference_utils import model_feedforward
from Utils.metric_utils import reliability_diagrams, nll
from Utils.mfij_utils import mfij_predict

def eval_indomain(args, data_key):
    model_dir = gen_model_dir(args)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # define model
    model = MLP(args.num_hiddens, drop_rate=0.0)
    model.to(device)
    # load best model
    with open(Path(model_dir, "model"), 'rb') as f:
        params = torch.load(f)
        model.load_state_dict(params['model_weight'])

    ## load data
    if data_key == 'heldout':
        _, eval_dl, _ = load_mnist_data(args, train_shuffle=False)
    elif data_key == 'test':
        _, _, eval_dl = load_mnist_data(args, train_shuffle=False)
    elif data_key == 'test_rotate':
        eval_dl = load_mnist_shift(args, data_key)

    ## MLE predict
    mle_logits, labels = model_feedforward(model, eval_dl, device)
    mle_probs = F.softmax(mle_logits, dim=1).cpu().detach().numpy()
    labels = labels.cpu().detach().numpy()
    mle_preds = np.argmax(mle_probs, axis=-1)
    del mle_logits

    # get penultimate layer features
    pen_features = get_penultimate_features(args, eval_dl)
    mfij_probs = mfij_predict(args,
                             loaders={data_key: pen_features},
                             dimensions=[args.num_hiddens[-1], args.num_classes],
                             num_train=55000)[data_key]
    mfij_preds = np.argmax(mfij_probs, axis=-1)

    # compute accuracy, NLL, Calibration
    mle_err = 1 - np.mean(mle_preds== labels)
    mfij_err = 1 - np.mean(mfij_preds == labels)

    mle_nll = np.mean(nll(labels, mle_probs))
    mfij_nll = np.mean(nll(labels, mfij_probs))

    fig, bin_confs, bin_accs, bin_percs = reliability_diagrams(mle_preds, labels, np.amax(mle_probs, axis=-1))
    mle_ECE = 100 * np.sum(np.array(bin_percs) * np.abs(np.array(bin_confs) - np.array(bin_accs)))

    fig, bin_confs, bin_accs, bin_percs = reliability_diagrams(mfij_preds, labels,
                                                               np.amax(mfij_probs, axis=-1))
    mfij_ECE = 100 * np.sum(np.array(bin_percs) * np.abs(np.array(bin_confs) - np.array(bin_accs)))

    ## print result
    print("& mle & {:.3g} & {:.4g} & {:.4g} \\\\".format(mle_err * 100, mle_nll, mle_ECE))

    print("& mfij & {:.3g} & {:.4g} & {:.4g} \\\\".format(mfij_err * 100, mfij_nll, mfij_ECE))

    print(" ==================================== ")
    print("\n")
    return mfij_err, mfij_nll, mfij_ECE

def parse_args(args=None):
    parser = argparse.ArgumentParser()
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

    tune_res = dict()
    # find the best ensemble and activation temperatures on heldout set
    ensemble_temperature_list = np.arange(-4, 3, 1, dtype=float)
    activation_temperature_list = np.array([0.001, 0.25, 0.5, 0.75, 1, 1.5, 2, 5])

    best_nll = np.inf

    for i, ens_T in enumerate(ensemble_temperature_list):
        args.temp_ens = float(10**ens_T)
        for j, act_T in enumerate(activation_temperature_list):
            args.temp_act = act_T
            print('temperatures:', args.temp_ens, args.temp_act)
            mfij_errs, mfij_nlls, mfij_ECEs = eval_indomain(args, data_key='heldout')
            tune_res[str(ens_T) + '_' + str(act_T)] = dict(err=mfij_errs,
                                                           nll=mfij_nlls,
                                                           ece=mfij_ECEs)
            if mfij_nlls < best_nll:
                best_nll = mfij_nlls
                nll_ts = [ens_T, act_T]
    tune_res['best_nll_temp'] = nll_ts
    tune_res['best_nll_heldout'] = best_nll

    print('eval best temperature on test', nll_ts)
    args.temp_ens, args.temp_act = nll_ts
    args.temp_ens = float(10 **args.temp_ens)
    test_errs, test_nlls, test_ECE = eval_indomain(args, data_key='test')
    tune_res['best_on_test'] = dict(err=test_errs,
                                     nll=test_nlls,
                                     ece=test_ECE)
    # save result
    model_dir = gen_model_dir(args)
    tune_fn = 'in_domain_mfij.json'
    with open(Path(model_dir, tune_fn), 'w') as fp:
        json.dump(tune_res, fp)

    # visualize temperature heatmap
    fig = viz_temperature_heatmap(tune_res,
                                  ensemble_temperature_list=list(ensemble_temperature_list),
                                  activation_temperature_list=activation_temperature_list,
                                  ensemble_temperature_label=['1e-4', '1e-3', '1e-2', '1e-1', '1.', '1e1', '1e2'],
                                  task='in_domain')
    fig.savefig("figs/mnist_mf_nll_temp.eps", dpi=fig.dpi, bbox_inches='tight', format='eps')

if __name__ == '__main__':
    with launch_ipdb_on_exception():
        main()
