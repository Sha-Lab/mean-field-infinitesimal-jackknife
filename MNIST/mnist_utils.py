import torch
import torch.nn.functional as F
from torch.utils import data
from pathlib import Path
import numpy as np
import scipy.ndimage
import matplotlib.pyplot as plt
import seaborn as sns

class MLP(torch.nn.Module):
    def __init__(self, n_hiddens, n_feature=784, n_output=10, drop_rate=0.5):
        super(MLP, self).__init__()
        self.num_feature = n_feature
        self.num_layers = len(n_hiddens) + 1
        self.layer1 = torch.nn.Linear(n_feature, n_hiddens[0])
        self.layer2 = torch.nn.Linear(n_hiddens[0], n_hiddens[1])
        self.layer3 = torch.nn.Linear(n_hiddens[1], n_output)
        self.dropout2 = torch.nn.Dropout(p=drop_rate)
        self.dropout3 = torch.nn.Dropout(p=drop_rate)

    def forward(self, x, return_act=False):
        x = x.view(-1, self.num_feature)
        h1 = self.layer1(x)
        a1 = F.relu(h1)
        a1d = self.dropout2(a1)

        h2 = self.layer2(a1d)
        a2 = F.relu(h2)
        a2d = self.dropout3(a2)

        h3 = self.layer3(a2d)
        if return_act:
            batch_size = x.shape[0]
            device = x.device
            act = dict()
            act['hs'] = [None] * 1
            act['hs'][0] = h3

            act['as'] = [None] * (self.num_layers + 1)
            act['as'][0] = torch.cat((x, torch.ones(batch_size, 1, device=device)), dim=1)
            act['as'][1] = torch.cat((a1, torch.ones(batch_size, 1, device=device)), dim=1)
            act['as'][2] = torch.cat((a2, torch.ones(batch_size, 1, device=device)), dim=1)
            act['as'][3] = F.softmax(h3, dim=1)
            return act
        return h3


def gen_model_dir(args):
    model_dir = Path(args.model_path, args.model_str,
                     'hidden_' + '_'.join([str(h) for h in args.num_hiddens]) \
                     + '_lr' + str(args.lr) + '_d' + str(args.lr_decay) \
                     + '_bs' + str(args.batch_size) + '_do' + str(args.drop_rate),
                     str(args.seed))
    return model_dir


def load_mnist_data(args, train_shuffle=True):
    dataset = np.load('data/mnist_split.npz')
    train_ds = data.TensorDataset(torch.Tensor(dataset['train_data']),
                                  torch.tensor(dataset['train_labels'], dtype=torch.int64))
    train_dl = data.DataLoader(train_ds, batch_size=args.batch_size, shuffle=train_shuffle)

    valid_ds = data.TensorDataset(torch.Tensor(dataset['heldout_data']),
                                  torch.tensor(dataset['heldout_labels'], dtype=torch.int64))
    valid_dl = data.DataLoader(valid_ds, batch_size=args.batch_size)

    test_ds = data.TensorDataset(torch.Tensor(dataset['test_data']),
                                 torch.tensor(dataset['test_labels'], dtype=torch.int64))
    test_dl = data.DataLoader(test_ds, batch_size=args.batch_size)
    return train_dl, valid_dl, test_dl


def load_notmnist(args, heldout=True):
    dataset = np.load('data/data_notMNIST_small.npz')
    split_indices = np.load('data/notmnist_split_idx.npz')
    if heldout:
        indices = split_indices['heldout_idx']
    else:
        indices = split_indices['test_idx']
    test_ds = torch.utils.data.Subset(
        torch.utils.data.TensorDataset(torch.Tensor(dataset['data'].astype(np.float32) / np.float32(255)),
                                       torch.tensor(dataset['labels'], dtype=torch.int64)),
        indices)
    print('heldout is', heldout, 'there are', len(test_ds), 'samples')
    test_dl = torch.utils.data.DataLoader(test_ds, batch_size=args.batch_size)
    return test_dl


def _crop_center(images, size):
    height, width = images.shape[1:3]
    i0 = height // 2 - size // 2
    j0 = width // 2 - size // 2
    return images[:, i0:i0 + size, j0:j0 + size]


def load_mnist_shift(args, data_key):
    dataset = np.load('data/mnist_split.npz')
    data_src = data_key.split('_')[0]
    images = dataset[data_src + '_data'].astype(np.float32)

    assert args.rotate_degs > 0
    images = scipy.ndimage.rotate(images, args.rotate_degs, axes=[-1, -2], order=1)
    test_data = _crop_center(images, 28)
    test_ds = data.TensorDataset(torch.Tensor(test_data),
                                 torch.tensor(dataset[data_src + '_labels'], dtype=torch.int64))
    test_dl = data.DataLoader(test_ds, batch_size=args.batch_size)
    return test_dl


def get_penultimate_features(args, data_dl):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # define model
    model = MLP(args.num_hiddens,
                n_feature=args.num_features,
                n_output=args.num_classes,
                drop_rate=0)
    model.to(device)
    model.eval()
    model_dir = gen_model_dir(args)

    # load model weight
    with open(Path(model_dir, "model"), 'rb') as f:
        params = torch.load(f, map_location=device)
        model.load_state_dict(params['model_weight'])

    # change batch_size temporarily
    _bs = args.batch_size
    args.batch_size = 5000
    args.batch_size = _bs

    # activations
    logits_mean = []
    features = []
    model.eval()
    with torch.no_grad():
        for xb, _ in data_dl:
            act_batch = model(xb.to(device), return_act=True)
            logits_mean.extend(act_batch['hs'][0])
            features.extend(act_batch['as'][2])
    # cast as numpy array
    logits_mean = torch.stack(logits_mean).detach()
    features = torch.stack(features).detach()
    rts = dict(features=features, logits_mean=logits_mean)
    return rts


def viz_temperature_heatmap(res_dict,
                            ensemble_temperature_list,
                            activation_temperature_list,
                            ensemble_temperature_label,
                            task='in_domain'):
    if task == 'in_domain':
        viz_field = 'nll'
    elif task == 'ood':
        viz_field = 'AUROC'
    vals = np.zeros((len(ensemble_temperature_list),
                     len(activation_temperature_list)))

    for i, e in enumerate(ensemble_temperature_list):
        for j, a in enumerate(activation_temperature_list):
            vals[i, j] = res_dict[str(e) + '_' + str(float(a))][viz_field]

    fig, ax = plt.subplots(figsize=(7, 5))
    sns.heatmap(vals, annot=True, fmt=".2f", cmap='coolwarm', square=True, ax=ax,
                vmin=vals.min(), vmax=vals.max(),
                cbar=False,
                xticklabels=activation_temperature_list,
                yticklabels=ensemble_temperature_label)

    ax.set_xlabel('activation temperature', fontsize=12)
    ax.set_ylabel('ensemble temperature', fontsize=12)
    return fig

def viz_distribution_shift(res_dict):
    rot_degs = np.arange(0, 181, 15)
    fig = plt.figure(figsize=(8, 6))

    sns.set(style="whitegrid", palette='colorblind', font_scale=1.2)

    rot_ece = [res_dict['rotate_' + str(d)]['ece'] for d in rot_degs]
    plt.plot(rot_degs, rot_ece, 'r*-', linewidth=3.5, markersize=11)
    plt.xticks(rot_degs)
    plt.xlabel('Rotation ($^\circ$)', fontsize=20)
    plt.ylabel('ECE (%)', fontsize=20)
    plt.xlim(0, 180)
    plt.tick_params(axis='x', labelrotation=30)
    plt.legend(['mf-ij'])
    return fig