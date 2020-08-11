import torch
import numpy as np

def mfij_predict(args,
                 loaders,
                 dimensions,
                 num_train=45000,
                 block_size=1000):
    '''
    Compute mean-field infinitesimal jackknife predictions
    :param args: hyper-parameters
    :param loaders: a dictionary with key as the dataset name, and value as a dictionary.
                    The value dictionary contains 2 entries. One key 'features' has value of penultimate layer features,
                    and another key 'logits_mean' has value of logits as in standard softmax.
                    For example, loaders={'heldout': {'features': pen_features_heldout, 'logits_mean': logits_heldout},
                                          'test': {'features': pen_features_test, 'logits_mean': logits_test}}.
    :param dimensions: [softmax_input_dim, softmax_output_dim]
    :param num_train: number of training samples
    :param block_size: batch size for parallelization
    :return:
    a dictionary with keys the same as loaders keys, and values of mfij predictions.
    '''
    hatc = eval(args.lambda0)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # apply temp_ens
    U_cov = (args.cov / (args.temp_ens * num_train)).to(device)

    U_cov_4d = torch.reshape(U_cov, [dimensions[1], dimensions[0]+1,
                                     dimensions[1], dimensions[0]+1]).permute([0, 2, 1, 3])
    probs_mfij = dict()
    for data_key, per_sample_preacts in loaders.items():
        logits_mean = per_sample_preacts['logits_mean'].to(device)
        dimN, dimD = logits_mean.shape
        logits_cov = torch.zeros((dimN, dimD, dimD))
        for s_idx in range(0, dimN, block_size):
            logits_cov[s_idx:s_idx+block_size] = torch.einsum('nk,ijkl,nl->nij',
                                      per_sample_preacts['features'][s_idx:s_idx+block_size].to(device),
                                      U_cov_4d.to(device),
                                      per_sample_preacts['features'][s_idx:s_idx+block_size].to(device)).cpu().detach()
        logits_cov = logits_cov.to(device)
        probs_mfij[data_key] = batch_mf(args, logits_mean, logits_cov, hatc)
    return probs_mfij

def mfij_predict_kron_approx(args,
                            loaders,
                            block_size=1000):
    hatc = eval(args.lambda0)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    probs_mfij = dict()
    for data_key, per_sample_moments in loaders.items():
        dimN = per_sample_moments['logits_mean'].shape[0]
        print('num of data samples in', data_key, dimN)
        probs_mfij[data_key] = np.zeros((dimN, args.num_classes))
        logits_mat = per_sample_moments['logits_cov_mat'].to(device)
        for s_idx in range(0, dimN, block_size):
            logits_mean = per_sample_moments['logits_mean'][s_idx:s_idx + block_size].to(device)
            logits_scale = per_sample_moments['logits_cov_scale'][s_idx:s_idx + block_size].to(device)
            logits_cov = torch.einsum('n,ij->nij',
                                      logits_scale / args.temp_ens, # apply temp_ens
                                      logits_mat)
            probs_mfij[data_key][s_idx:s_idx + block_size] = batch_mf(args,
                                                                      logits_mean,
                                                                      logits_cov,
                                                                      hatc)
    return probs_mfij

def batch_mf(args,
             logits_mean,
             logits_cov,
             hatc):
    # apply temp_act
    if args.temp_act > 0:
        logits_mean /= args.temp_act
        # N x num_classes x num_classes
        logits_cov /= (args.temp_act ** 2)
    # N x num_classes x num_classes
    mukj = logits_mean.unsqueeze(2) - logits_mean.unsqueeze(1)
    if args.mf_approx == 'mf0':
        skj = torch.sqrt(1 + hatc * torch.diagonal(logits_cov, dim1=-2, dim2=-1)).unsqueeze(2)
    elif args.mf_approx == 'mf1':
        sigmak = torch.diagonal(logits_cov, dim1=-2, dim2=-1)
        sigmakj = sigmak.unsqueeze(2) + sigmak.unsqueeze(1)
        skj = torch.sqrt(1 + hatc * sigmakj)
        del sigmakj, sigmak
    elif args.mf_approx == 'mf2':
        sigmak = torch.diagonal(logits_cov, dim1=-2, dim2=-1)
        sigmakj = sigmak.unsqueeze(2) + sigmak.unsqueeze(1) - 2 * logits_cov
        skj = torch.sqrt(1 + hatc * sigmakj)
        del sigmakj, sigmak
    del logits_cov
    probs_unormalized = 1. / torch.sum(torch.exp(- mukj / skj), dim=-1)
    if args.temp_act > 0:
        # normalize
        probs = probs_unormalized / torch.sum(probs_unormalized, dim=1, keepdim=True)
    elif args.temp_act == 0:
        # do not normalize when args.temp_act == 0
        probs = probs_unormalized
    return probs.cpu().detach().numpy().astype(np.float64)