'''Train ImageNet with PyTorch.'''
import torch
import torchvision
import numpy as np
import time
import shutil
from pathlib import Path
import argparse
import json
from ipdb import launch_ipdb_on_exception
from train_utils import adjust_learning_rate, accuracy, validate, \
    AverageMeter, ProgressMeter
from imagenet_utils import gen_model_dir, load_imagenet
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

def train(args):
    args.print_freq = 100
    args.gpu = None
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # load data
    train_dl, valid_dl, test_dl = load_imagenet(args)
    # define model
    model = torchvision.models.resnet50(pretrained=False)
    # multiple gpus
    model = torch.nn.DataParallel(model).cuda()
    loss_fn = torch.nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=0.9,
                                weight_decay=args.weight_decay)

    model_dir = gen_model_dir(args)
    model_dir.mkdir(parents=True, exist_ok=True)

    torch.backends.cudnn.benchmark = True
    best_acc1 = 0
    for epoch in range(args.n_epochs):
        adjust_learning_rate(optimizer, epoch, args)
        batch_time = AverageMeter('Time', ':6.3f')
        data_time = AverageMeter('Data', ':6.3f')
        losses = AverageMeter('Loss', ':.4e')
        top1 = AverageMeter('Acc@1', ':6.2f')
        top5 = AverageMeter('Acc@5', ':6.2f')
        progress = ProgressMeter(
            len(train_dl),
            [batch_time, data_time, losses, top1, top5],
            prefix="Epoch: [{}]".format(epoch))

        # switch to train mode
        model.train()
        end = time.time()
        for batch_idx, (images, target) in enumerate(train_dl):
            # measure data loading time
            data_time.update(time.time() - end)

            # if args.gpu is not None:
            images = images.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)

            # compute output
            output = model(images)
            loss = loss_fn(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            # compute gradient and do SGD step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if batch_idx % args.print_freq == 0:
                progress.display(batch_idx)

        # evaluate on validation set
        acc1 = validate(valid_dl, model, loss_fn, args)

        # remember best acc@1 and save checkpoint
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)
        torch.save(
            {
                'epoch': epoch + 1,
                'model_weight': model.state_dict(),
                'heldout_best_acc1': best_acc1,
                'optimizer': optimizer.state_dict(),
            }, Path(model_dir, 'model'))
        if is_best:
            shutil.copyfile(Path(model_dir, 'model'), Path(model_dir, 'model_best'))

    # load best model
    with open(Path(model_dir, "model_best"), 'rb') as f:
        params = torch.load(f)
        model.load_state_dict(params['model_weight'])

    # test
    model.eval()
    # evaluate on test set
    acc_test = validate(test_dl, model, loss_fn, args)

    print('epoch: {}, val acc: {:.4f}, test acc: {:.4f}'.format(
        params["epoch"], params["heldout_best_acc1"], acc_test))

    with open(Path(model_dir, "res.json"), 'w') as fp:
        json.dump({
            'epoch': params["epoch"],
            'heldout_best_acc1': params["heldout_best_acc1"].item(),
            'test_best_acc1': acc_test.item(),
        },
            fp)

def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_workers', default=4, type=int, help='number of data loading workers (default: 4)')
    parser.add_argument('--model_path', type=str, default="model/")
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--n_epochs', type=int, default=90)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    args = parser.parse_args(args)
    return args

def main(args=None):
    args = parse_args(args)
    train(args)

if __name__ == '__main__':
    with launch_ipdb_on_exception():
        main()
