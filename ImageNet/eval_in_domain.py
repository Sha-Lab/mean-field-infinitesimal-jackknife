import torch
import torchvision
import torch.nn.functional as F
import numpy as np
import argparse
from pathlib import Path
import json
from ipdb import launch_ipdb_on_exception

from imagenet_utils import gen_model_dir, load_imagenet, map_key_singlegpu, \
    get_penultimate_features, precompute_logits_cov

import sys
sys.path.append('../')
from Utils.hess_utils import compute_kron_approx_hess, invert_kron_approx_hess
from Utils.inference_utils import model_feedforward
from Utils.metric_utils import reliability_diagrams, nll
from Utils.mfij_utils import mfij_predict_kron_approx

def eval_indomain(args, data_key, print_mle=True):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    model_dir = gen_model_dir(args)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    ### dbg
    label_dict = np.load('data/label_dict.npz')
    labels = label_dict[data_key]
    ### dbg

    # define model
    model = torchvision.models.resnet50(pretrained=False)  # (args.seed==111)

    print('load model!')
    # load best model
    with open(Path(model_dir, "model_best"), 'rb') as f:
        params = torch.load(f, map_location=device)
        model.load_state_dict(map_key_singlegpu(params['model_weight']))
    model.to(device)
    if print_mle:
        # MLE predict
        # load data
        if data_key == 'heldout':
            _, eval_dl, _ = load_imagenet(args, train_shuffle=False)
        elif data_key == 'test':
            _, _, eval_dl = load_imagenet(args, train_shuffle=False)

        mle_logits, labels = model_feedforward(model=model, data_loader=eval_dl,
                                               device=device, is_eval=True)
        mle_probs = F.softmax(mle_logits, dim=1).cpu().detach().numpy()
        labels = labels.cpu().detach().numpy()
        mle_preds = np.argmax(mle_probs, axis=-1)
        del mle_logits

        mle_err = 1 - np.mean(mle_preds == labels)
        mle_nll = np.mean(nll(labels, mle_probs))
        fig, bin_confs, bin_accs, bin_percs = reliability_diagrams(mle_preds,
                                                                   labels,
                                                                   np.amax(mle_probs, axis=-1),
                                                                   bin_size=args.bin_size)
        mle_ECE = 100 * np.sum(np.array(bin_percs) * np.abs(np.array(bin_confs) - np.array(bin_accs)))
        print("& mle & {:6.5f} & {:6.5f} & {:6.5f} \\\\".format(mle_err * 100, mle_nll, mle_ECE))


    assert args.num_classes == args.cov_per_sample[data_key]['logits_mean'].shape[1]
    mfij_probs = mfij_predict_kron_approx(args,
                                          loaders={data_key: args.cov_per_sample[data_key]},
                                          block_size=500)[data_key]
    mfij_preds = np.argmax(mfij_probs, axis=-1)

    # compute accuracy, NLL, Calibration
    mfij_err = 1 - np.mean(mfij_preds == labels)
    mfij_nll = np.mean(nll(labels, mfij_probs))
    fig, bin_confs, bin_accs, bin_percs = reliability_diagrams(mfij_preds, labels,
                                                               np.amax(mfij_probs, axis=-1),
                                                               bin_size=args.bin_size)
    mfij_ECE = 100 * np.sum(np.array(bin_percs) * np.abs(np.array(bin_confs) - np.array(bin_accs)))

    ## print result
    print("& method & {:6.5f} & {:6.5f} & {:6.5f} \\\\".format(mfij_err * 100, mfij_nll, mfij_ECE))

    print(" ==================================== ")
    print("\n")
    return mfij_err, mfij_nll, mfij_ECE

def parse_args(args=None):
    parser = argparse.ArgumentParser()
    ## uncertainty estimate args
    parser.add_argument('--mf_approx', type=str, default='mf0')
    # #bins=15 when computing calibration on ImageNet, following paper
    # [On Calibration of Modern Neural Networks](https://arxiv.org/pdf/1706.04599.pdf)
    parser.add_argument('--bin_size', type=float, default=1/15.)
    parser.add_argument('--temp_ens', type=float, default=1.0)
    parser.add_argument('--temp_act', type=float, default=1.0)

    ## model training args, use the same values as in train.py file
    parser.add_argument('--model_path', type=str, default="model/")
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--num_workers', default=8, type=int, help='number of data loading workers (default: 4)')
    parser.add_argument('--num_classes', type=int, default=1000)

    args = parser.parse_args(args)
    return args

def main(args=None):
    args = parse_args(args)

    print('processing model: lr' + str(args.lr) + '_bs' + str(args.batch_size),
          'seed', args.seed)
    # mfij related argument
    args.lambda0 = '3/(np.pi**2)'

    # Compute approximate Hessian covariance matrix
    # change batch_size temporarily
    _bs = args.batch_size
    args.batch_size = 512
    train_dl, _, _ = load_imagenet(args, train_shuffle=False)
    args.batch_size = _bs

    train_pen_features = get_penultimate_features(args, train_dl)

    Hhh, Haa = compute_kron_approx_hess(train_pen_features)
    invH, invA = invert_kron_approx_hess(Hhh, Haa)
    num_train = train_pen_features['logits_mean'].shape[0]
    del train_pen_features # train_dl

    # Pre-compute logits covariance matrix per data point,
    # so that it can be reused for different temperatures
    args.cov_per_sample = dict()
    _, heldout_dl, test_dl = load_imagenet(args, train_shuffle=False)
    args.cov_per_sample['heldout'] = precompute_logits_cov(args,
                                                           heldout_dl,
                                                           invA,
                                                           invH,
                                                           num_train=num_train)

    args.cov_per_sample['test'] = precompute_logits_cov(args,
                                                        test_dl,
                                                        invA,
                                                        invH,
                                                        num_train=num_train)
    del heldout_dl, test_dl
    print('Precompute heldout and test per sample logits covariance done.')

    tune_res = dict()
    # find the best ensemble and activation temperatures on heldout set
    # ensemble_temperature_list = np.arange(-5, 1, 1, dtype=float)
    ensemble_temperature_list = np.arange(-7, -2, 0.5, dtype=float)
    activation_temperature_list = np.array([0.1, 0.25, 0.5, 0.75, 1, 1.5, 2, 5])
    best_nll = np.inf

    for i, ens_T in enumerate(ensemble_temperature_list):
        args.temp_ens = float(10 ** ens_T)
        for j, act_T in enumerate(activation_temperature_list):
            args.temp_act = act_T
            print('temperatures:', args.temp_ens, args.temp_act)

            mfij_errs, mfij_nlls, mfij_ECEs = eval_indomain(args, data_key='heldout', print_mle=False)
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
    args.temp_ens = float(10 ** args.temp_ens)
    test_errs, test_nlls, test_ECE = eval_indomain(args, data_key='test', print_mle=True)
    tune_res['best_on_test'] = dict(err=test_errs,
                                    nll=test_nlls,
                                    ece=test_ECE)

    # save result
    model_dir = gen_model_dir(args)
    tune_fn = 'in_domain_mfij.json'

    with open(Path(model_dir, tune_fn), 'w') as fp:
        json.dump(tune_res, fp)


if __name__ == '__main__':
    with launch_ipdb_on_exception():
        main()
