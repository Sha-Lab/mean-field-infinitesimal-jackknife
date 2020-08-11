import torch
import torchvision
import torchvision.transforms as transforms
from pathlib import Path
from collections import OrderedDict
import numpy as np

def gen_model_dir(args):
    model_dir = Path(args.model_path, 'resnet50',
                         'lr' + str(args.lr) + '_bs' + str(args.batch_size),
                         str(args.seed))
    return model_dir

def load_imagenet(args, train_shuffle=True):
    traindir = Path('data', 'train')
    heldoutdir = Path('data', 'heldout')
    testdir = Path('data', 'test')

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    if train_shuffle:
        train_dataset = torchvision.datasets.ImageFolder(
            traindir,
            transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]))
    else:
        train_dataset = torchvision.datasets.ImageFolder(
            traindir,
            transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ]))


    train_dl = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=train_shuffle,
        num_workers=args.num_workers, pin_memory=True)

    valid_dl = torch.utils.data.DataLoader(
        torchvision.datasets.ImageFolder(heldoutdir, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True)
    test_dl = torch.utils.data.DataLoader(
        torchvision.datasets.ImageFolder(testdir, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True)
    return train_dl, valid_dl, test_dl

def map_key_singlegpu(state_dict):
    '''
    remove `module.` in the keys of state_dict of models trained on multiple gpus.
    :param state_dict:
    :return: new_state_dict with modified keys
    '''
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:]  # remove `module.`
        new_state_dict[name] = v
    return new_state_dict

def get_penultimate_features(args, data_dl):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # define model
    model = torchvision.models.resnet50(pretrained=False)
    model.to(device)
    model_dir = gen_model_dir(args)
    print('load model!')
    with open(Path(model_dir, "model_best"), 'rb') as f:
        params = torch.load(f)
        model.load_state_dict(map_key_singlegpu(params['model_weight']))

    model.eval()
    def avgpool_hook(module, input, output):
        # output: B x (D+1)
        fts = output.cpu().detach()
        features.append(
            torch.cat((torch.flatten(fts, 1),
                       torch.ones(fts.shape[0], 1, device=fts.device)),
                      dim=1)
        )
    model.avgpool.register_forward_hook(avgpool_hook)

    logits_mean = []
    features = []
    labels = []
    model.eval()
    with torch.no_grad():
        for idx, (xb, yb) in enumerate(data_dl):
            out = model(xb.to(device))
            logits_mean.append(out.cpu().detach())
            labels.append(yb.cpu().detach())
            if idx % 500 == 0:
                print('pen features', idx, 'done!')

    # concat
    logits_mean = torch.cat(logits_mean, dim=0).detach()
    features = torch.cat(features, dim=0).detach()
    labels = torch.cat(labels, dim=0).detach()
    rts = dict(features=features, logits_mean=logits_mean, labels=labels)
    return rts

def precompute_logits_cov(args,
                          data_dl,
                          invA,
                          invH,
                          block_size=1000,
                          num_train=1281167):
    # use kronecker factor structure to compute per data point logits covariance
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    invA = torch.Tensor(invA).to(device)
    invH = torch.Tensor(invH / num_train).to(device)

    pen_features = get_penultimate_features(args, data_dl)

    dimN, dimD = pen_features['logits_mean'].shape
    print('number of data samples', dimN)
    logits_cov_scale = torch.zeros((dimN,))
    for s_idx in range(0, dimN, block_size):
        logits_cov_scale[s_idx:s_idx + block_size] = torch.einsum('nk,kl,nl->n',
                                                                  pen_features['features'][
                                                                  s_idx:s_idx + block_size].to(device),
                                                                  invA,
                                                                  pen_features['features'][
                                                                  s_idx:s_idx + block_size].to(device))
        if s_idx % 10000 == 0:
            print(s_idx, 'is done!')

    del pen_features['features']
    pen_features['logits_cov_scale'] = logits_cov_scale
    pen_features['logits_cov_mat'] = invH
    return pen_features