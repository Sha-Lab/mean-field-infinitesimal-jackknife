import torch
import numpy as np
from pathlib import Path
import json
import argparse
from ipdb import launch_ipdb_on_exception
from mnist_utils import gen_model_dir, MLP, load_mnist_data

def eval_acc(data_loader, model, device):
    model.eval()
    total = 0
    correct = 0
    with torch.no_grad():
        for images, labels in data_loader:
            outputs = model(images.to(device))
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels.to(device)).sum().item()
    return correct / total

def train(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # load data
    train_dl, valid_dl, test_dl = load_mnist_data(args)

    # define model
    if args.model_str == 'mlp':
        model = MLP(args.num_hiddens, drop_rate=args.drop_rate)

    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    # and a learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.lr_decay)

    model_dir = gen_model_dir(args)
    model_dir.mkdir(parents=True, exist_ok=True)

    loss_fn = torch.nn.CrossEntropyLoss()
    min_err = 1
    for epoch in range(args.n_epochs):
        model.train()
        loss_train = 0
        for data, target in train_dl:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = loss_fn(output, target)
            loss.backward()
            optimizer.step()

            loss_train += loss.item()
        loss_train /= len(train_dl)
        lr_scheduler.step()

        # eval heldout
        model.eval()
        with torch.no_grad():
            err_heldout = 1 - eval_acc(valid_dl, model, device=device)

        print('Train Epoch: {}, Train Loss: {:.6f}, Heldout Err: {:.6f}'.format(
            epoch, loss_train, err_heldout))

        if err_heldout < min_err:
            min_err = err_heldout
            # err_heldout = 1 - eval_acc(valid_dl, model, device=device)
            loss_heldout = sum(loss_fn(
                model(xb.to(device)), yb.to(device)) for xb, yb in valid_dl).item() / len(valid_dl)
            # save model
            with open(Path(model_dir, "model"), 'wb') as f:
                torch.save({
                    'model_weight': model.state_dict(),
                    'epoch': epoch,
                    'loss_train': loss_train,
                    'loss_heldout': loss_heldout,
                    'err_heldout': err_heldout,
                },
                    f,
                )
            print(
            'New best! epoch: {}, learning rate: {:.4g}, train loss: {:.4f}, val err: {:.2f}.'.format(
                epoch, lr_scheduler.get_last_lr()[0], loss_train, err_heldout * 100))

    # load best model
    with open(Path(model_dir, "model"), 'rb') as f:
        params = torch.load(f)
        model.load_state_dict(params['model_weight'])
    # test
    model.eval()
    err_test = 1 - eval_acc(test_dl, model, device=device)
    print('epoch: {}, val error: {:.4f}, test error: {:.4f}'.format(
        params["epoch"], params["err_heldout"] * 100, err_test * 100))

    with open(Path(model_dir, "res.json"), 'w') as fp:
        json.dump({
            'epoch': params["epoch"],
            'loss_train': params["loss_train"],
            'loss_heldout': params["loss_heldout"],
            'err_heldout': params["err_heldout"],
            'err_test': err_test,
        },
            fp)

def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default="model/")
    parser.add_argument('--model_str', type=str, default="mlp")
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--num_hiddens', nargs='+', type=int, default=[256, 256])
    parser.add_argument('--drop_rate', type=float, default=0.1)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--lr_decay', type=float, default=0.998)
    parser.add_argument('--n_epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--weight_decay', type=float, default=0)
    args = parser.parse_args(args)
    return args

def main(args=None):
    args = parse_args(args)
    train(args)

if __name__ == '__main__':
    with launch_ipdb_on_exception():
        main()
