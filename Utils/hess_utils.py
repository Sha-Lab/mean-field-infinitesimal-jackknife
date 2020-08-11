import torch
import torch.nn.functional as F
import numpy as np
import scipy.linalg

def batch_hess_sm(batch_p, num, dim=10):
    diag_p = torch.zeros([num, dim, dim], device=batch_p.device)
    torch.einsum('njj->nj', diag_p)[...] = batch_p
    outer_p = torch.einsum('nj,nk->njk', batch_p, batch_p)
    return diag_p - outer_p

def batch_kron_sum(batch_Hhh, batch_a, kron_dim, use_cpu=False):
    outer_a = torch.einsum('nj,nk->njk', batch_a, batch_a)
    # kronecker product
    if use_cpu:
        batch_Hhh = batch_Hhh.cpu()
        outer_a = outer_a.cpu()
    kron_sum = torch.einsum('nik,njl->ijkl', batch_Hhh, outer_a).reshape([kron_dim, kron_dim])
    return kron_sum

def compute_hess(pen_features, batch_size=5000):
    # compute hessian of the softmax layer on the training set
    dims = [0] * 2
    num_train, dims[1] = pen_features['logits_mean'].shape
    dims[0] = pen_features['features'].shape[1] - 1

    print('num_train = {:d}, d0 = {:d}, d1 = {:d}'.format(num_train, dims[0], dims[1]))
    Hww = torch.zeros(((dims[0] + 1) * dims[1], (dims[0] + 1) * dims[1])).to(
        pen_features['features'].device)

    for sidx in range(0, num_train, batch_size):
        # ******* softmax layer *********
        a1 = F.softmax(pen_features['logits_mean'], dim=1)[sidx:sidx + batch_size]
        a0 = pen_features['features'][sidx:sidx + batch_size]
        # batch_size x dim3 x dim3
        Hhh = batch_hess_sm(a1, num=a1.shape[0], dim=dims[1])
        Hww += batch_kron_sum(Hhh, a0, kron_dim=dims[1]*(dims[0]+1))
        del a1

    Hww /= num_train
    print('hessian computation done.')
    return Hww.cpu().detach().numpy()

def invert_hess(Huu):
    U_val, U_vec = scipy.linalg.eigh(Huu)
    print('eig decomposition done.')

    # add 1-smallest eigen value
    U_valm = U_val + 1-np.amin(U_val)
    invU = np.matmul(np.matmul(U_vec,
                               np.diag(1. / U_valm)),
                     U_vec.T)
    print('inversion done.')
    return invU

def compute_kron_approx_hess(pen_features, batch_size=5000):
    # when Hessian is too big to fit into memory,
    # use Kronecker factored approximation.
    dims = [0] * 2
    num_train, dims[1] = pen_features['logits_mean'].shape
    dims[0] = pen_features['features'].shape[1] - 1

    print('num_train = {:d}, d0 = {:d}, d1 = {:d}'.format(num_train, dims[0], dims[1]))

    Hhh = torch.zeros((dims[1], dims[1]))
    Haa = torch.zeros((dims[0] + 1), (dims[0] + 1))
    for sidx in range(0, num_train, batch_size):
        # ******* softmax layer *********
        a1 = F.softmax(pen_features['logits_mean'][sidx:sidx + batch_size], dim=1)
        a0 = pen_features['features'][sidx:sidx + batch_size]
        # num_classes x num_classes
        Hhh += torch.sum(batch_hess_sm(a1, num=a1.shape[0], dim=dims[1]), dim=0)
        Haa += torch.einsum('nj,nk->jk', a0, a0)
        del a0, a1
        if sidx % 1e5 == 0:
            print(sidx, 'is done!')

    Hhh /= num_train
    Haa /= num_train
    return Hhh, Haa


def invert_kron_approx_hess(Hhh, Haa):
    # inversion for Kronecker factors
    H_val, H_vec = scipy.linalg.eigh(Hhh)
    print('Hhh eig decomposition done!')
    A_val, A_vec = scipy.linalg.eigh(Haa)
    print('Haa eig decomposition done!')

    A_valm = A_val + 1 - np.amin(A_val)
    invA = np.matmul(np.matmul(A_vec,
                               np.diag(1. / A_valm)),
                     A_vec.T)

    del A_valm, A_val, A_vec
    H_valm = H_val + 1 - np.amin(H_val)
    invH = np.matmul(np.matmul(H_vec,
                               np.diag(1. / H_valm)),
                     H_vec.T)
    del H_valm, H_val, H_vec
    print('inversion done.')
    return invH, invA