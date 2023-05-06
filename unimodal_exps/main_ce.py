from __future__ import print_function

import os
import sys
import argparse
import time
import math
import random
import numpy as np

import tensorboard_logger as tb_logger
import torch
import torch.backends.cudnn as cudnn
from torchvision import transforms, datasets

from util import AverageMeter
from util import adjust_learning_rate, warmup_learning_rate, accuracy
from util import set_optimizer, save_model
from util import My_iNaturalist, CIFAR10_LT, CIFAR100_LT, ImageNet_LT
from networks.resnet import SupCEResNet
from torch.utils.data.sampler import SubsetRandomSampler

from sampler import TriDataSampler
from avalanche.benchmarks.datasets import INATURALIST2018
from ce_isogclr import CE_iSogCLR_Loss


def parse_option():
    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--print_freq', type=int, default=10,
                        help='print frequency')
    parser.add_argument('--save_freq', type=int, default=50,
                        help='save frequency')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='batch_size')
    parser.add_argument('--num_workers', type=int, default=16,
                        help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=200,
                        help='number of training epochs')

    # optimization
    parser.add_argument('--learning_rate', type=float, default=0.03,
                        help='learning rate')
    parser.add_argument('--lr_decay_epochs', type=str, default='350,400,450',
                        help='where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1,
                        help='decay rate for learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='momentum')

    # model dataset
    parser.add_argument('--data_folder', type=str, default='./datasets')
    parser.add_argument('--model', type=str, default='resnet50')
    parser.add_argument('--dataset', type=str, default='cifar10',
                        choices=['cifar10', 'cifar100', 'imagenet-lt', 'iNat'], help='dataset')
    parser.add_argument('--size', type=int, default=224)

    # other setting
    parser.add_argument('--cosine', action='store_true',
                        help='using cosine annealing')
    parser.add_argument('--syncBN', action='store_true',
                        help='using synchronized batch normalization')
    parser.add_argument('--warm', action='store_true',
                        help='warm-up for large batch training')
    parser.add_argument('--trial', type=str, default='0',
                        help='id for recording multiple runs')

    parser.add_argument('--use_amp', action='store_true')

    parser.add_argument('--use_trisampler', action='store_true')
    parser.add_argument('--multi_task', default=16, type=int)
    parser.add_argument('--pos_num', default=10, type=int)
    parser.add_argument('--ce_isogclr_obj', action='store_true')
    parser.add_argument('--tau', default=0.5, type=float)
    parser.add_argument('--alpha', default=1.0, type=float)

    opt = parser.parse_args()

    # set the path according to the environment
    opt.model_path = './save/SupCon/{}_models'.format(opt.dataset)
    opt.tb_path = './save/SupCon/{}_tensorboard'.format(opt.dataset)

    iterations = opt.lr_decay_epochs.split(',')
    opt.lr_decay_epochs = list([])
    for it in iterations:
        opt.lr_decay_epochs.append(int(it))

    opt.model_name = 'SupCE_{}_{}_lr_{}_decay_{}_bsz_{}_trial_{}'.\
        format(opt.dataset, opt.model, opt.learning_rate, opt.weight_decay,
               opt.batch_size, opt.trial)

    if opt.cosine:
        opt.model_name = '{}_cosine'.format(opt.model_name)

    # warm-up for large-batch training,
    if opt.batch_size > 256:
        opt.warm = True
    if opt.warm:
        opt.model_name = '{}_warm'.format(opt.model_name)
        opt.warmup_from = 0.01
        opt.warm_epochs = 10
        if opt.cosine:
            eta_min = opt.learning_rate * (opt.lr_decay_rate ** 3)
            opt.warmup_to = eta_min + (opt.learning_rate - eta_min) * (
                    1 + math.cos(math.pi * opt.warm_epochs / opt.epochs)) / 2
        else:
            opt.warmup_to = opt.learning_rate

    opt.tb_folder = os.path.join(opt.tb_path, opt.model_name)
    if not os.path.isdir(opt.tb_folder):
        os.makedirs(opt.tb_folder)

    opt.save_folder = os.path.join(opt.model_path, opt.model_name)
    if not os.path.isdir(opt.save_folder):
        os.makedirs(opt.save_folder)

    if opt.dataset == 'cifar10':
        opt.n_cls = 10
    elif opt.dataset == 'cifar100':
        opt.n_cls = 100
    elif opt.dataset == 'imagenet-lt':
        opt.n_cls = 1000
    elif opt.dataset == 'iNat':
        opt.n_cls = 8138
    else:
        raise ValueError('dataset not supported: {}'.format(opt.dataset))

    return opt


def set_loader(opt):
    if opt.dataset in ['cifar10', 'cifar10-lt']:
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2023, 0.1994, 0.2010)
    elif opt.dataset in ['cifar100', 'cifar100-lt']:
        mean = (0.5071, 0.4867, 0.4408)
        std = (0.2675, 0.2565, 0.2761)
    elif opt.dataset == 'svhn':
        mean = (0.438, 0.444, 0.473)
        std = (0.1751, 0.1771, 0.1744)
    elif 'imagenet' in opt.dataset or 'iNat' in opt.dataset:
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
    else:
        raise ValueError('dataset not supported: {}'.format(opt.dataset))
    normalize = transforms.Normalize(mean=mean, std=std)

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(size=opt.size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])

    val_transform = transforms.Compose([
        transforms.Resize(opt.size),
        transforms.CenterCrop(opt.size),
        transforms.ToTensor(),
        normalize,
    ])

    if opt.dataset == 'cifar10':
        train_dataset = datasets.CIFAR10(root=opt.data_folder,
                                         transform=train_transform,
                                         download=True)
        val_dataset = datasets.CIFAR10(root=opt.data_folder,
                                       train=False,
                                       transform=val_transform, download=True)
    elif opt.dataset == 'cifar100':
        train_dataset = datasets.CIFAR100(root=opt.data_folder,
                                          transform=train_transform,
                                          download=True)
        val_dataset = datasets.CIFAR100(root=opt.data_folder,
                                        train=False,
                                        transform=val_transform, download=True)
    elif opt.dataset == 'svhn':
        train_dataset = datasets.SVHN(root=opt.data_folder, split='train', transform=train_transform, download=True)
        val_dataset = datasets.SVHN(root=opt.data_folder, split='test', transform=val_transform, download=True)

    elif opt.dataset == 'iNat':
        train_dataset = INATURALIST2018(root=opt.data_folder, transform=train_transform, target_transform=int)
        val_dataset = INATURALIST2018(root=opt.data_folder, transform=val_transform, target_transform=int, split='val')

        print("iNat training samples:", len(train_dataset))
        print("iNat test samples:", len(val_dataset))

        """
        train_dataset = My_iNaturalist(root=opt.data_folder, 
                                        content_file=os.path.join(opt.data_folder, 'Inat_dataset_splits/Inaturalist_train_set1.txt'), 
                                        transform=train_transform,
                                        use_fine_label=opt.fine_label_iNat)
        val_dataset = My_iNaturalist(root=opt.data_folder, 
                                        content_file=os.path.join(opt.data_folder, 'Inat_dataset_splits/Inaturalist_test_set1.txt'), 
                                        transform=val_transform,
                                        use_fine_label=opt.fine_label_iNat)
        """

    elif opt.dataset in ['imagenet', 'imagenet100']:
        train_dataset = datasets.ImageFolder(root=opt.data_folder+'train',
											transform=train_transform)
        val_dataset = datasets.ImageFolder(root=opt.data_folder+'val',
										   transform=val_transform)

    elif opt.dataset == 'cifar10-lt':
        train_dataset = CIFAR10_LT(root=opt.data_folder, train=True, download=True,
                                   transform=train_transform, return_idx=False)

        val_dataset = datasets.CIFAR10(root=opt.data_folder, train=False,
                                         transform=val_transform, download=True)
        print('cifar10-lt:', train_dataset.get_cls_num_list())

    elif opt.dataset == 'cifar100-lt':
        train_dataset = CIFAR100_LT(root=opt.data_folder, train=True, download=True,
                                   transform=train_transform, return_idx=False)

        val_dataset = datasets.CIFAR100(root=opt.data_folder, train=False,
                                         transform=val_transform, download=True)
        print('cifar100-lt:', train_dataset.get_cls_num_list())

    elif opt.dataset == 'imagenet-lt':
        train_dataset = ImageNet_LT(root=opt.data_folder, txt='./datasets/ImageNet_LT/ImageNet_LT_train.txt',
                                        transform=train_transform, return_idx=opt.ce_isogclr_obj)

        val_dataset = datasets.ImageFolder(root=os.path.join(opt.data_folder, 'val'), 
                                           transform=val_transform)
        print("imagenet-lt:", len(train_dataset))
    else:
        assert 0, opt.dataset + ' not implemented !'

    # try TriSampler in libauc
    labels_list = np.array(train_dataset.targets)
    labels_one_hot = np.zeros((labels_list.size, labels_list.max()+1))
    labels_one_hot[np.arange(labels_list.size), labels_list] = 1   # shape: [115846, 1000]
    
    if opt.use_trisampler:
        train_sampler = TriDataSampler(labels=labels_one_hot, batchSize=opt.batch_size, posNum=opt.pos_num, multi_task=opt.multi_task) # 16 tasks, 10 pos + 10 neg, batchsize should be 16*(10+10)=320
    else:
        train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=opt.batch_size, shuffle=(train_sampler is None),
        num_workers=opt.num_workers, pin_memory=True, sampler=train_sampler, prefetch_factor=12)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=2048, shuffle=False,
        num_workers=8, pin_memory=True)

    return train_loader, val_loader, len(train_dataset)



def set_model(opt, num_samples):
    model = SupCEResNet(name=opt.model, num_classes=opt.n_cls)

    if opt.ce_isogclr_obj:
        criterion = CE_iSogCLR_Loss(multi_task=opt.multi_task, num_pos=opt.pos_num, N=num_samples, tau=opt.tau, alpha=opt.alpha)
    else:
        criterion = torch.nn.CrossEntropyLoss()

    # enable synchronized Batch Normalization
    if opt.syncBN:
        model = apex.parallel.convert_syncbn_model(model)

    if torch.cuda.is_available():
        if torch.cuda.device_count() > 1:
            model = torch.nn.DataParallel(model)
        model = model.cuda()
        criterion = criterion.cuda()
        cudnn.benchmark = True

    return model, criterion


def train(train_loader, model, criterion, optimizer, epoch, scaler, opt):
    """one epoch training"""
    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    end = time.time()
    for idx, (samples, labels) in enumerate(train_loader):
        data_time.update(time.time() - end)

        if opt.ce_isogclr_obj:
            index, images = samples
            index = index.cuda(non_blocking=True)
        else:
            images = samples

        images = images.cuda(non_blocking=True)
        labels = labels.cuda(non_blocking=True)
        bsz = labels.shape[0]

        # warm-up learning rate
        warmup_learning_rate(opt, epoch, idx, len(train_loader), optimizer)

        optimizer.zero_grad()

        if opt.use_amp:
            with torch.cuda.amp.autocast():
                # compute loss
                output = model(images)

                if not opt.ce_isogclr_obj: # ce only
                    loss = criterion(output, labels)
                else:
                    loss = criterion(output, labels, index)

            # update metric
            losses.update(loss.item(), bsz)
            acc1, acc5 = accuracy(output, labels, topk=(1, 5))
            top1.update(acc1[0], bsz)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        else:
            # compute loss
            output = model(images)
            
            if not opt.ce_isogclr_obj: # ce only
                loss = criterion(output, labels)
            else:
                loss = criterion(output, labels, index)

            # update metric
            losses.update(loss.item(), bsz)
            acc1, acc5 = accuracy(output, labels, topk=(1, 5))
            top1.update(acc1[0], bsz)

            # SGD
            loss.backward()
            optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # print info
        if (idx + 1) % opt.print_freq == 0:
            print('Train: [{0}][{1}/{2}]\t'
                  'BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'DT {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'loss {loss.val:.3f} ({loss.avg:.3f})\t'
                  'Acc@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                   epoch, idx + 1, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses, top1=top1))
            sys.stdout.flush()

    return losses.avg, top1.avg


def validate(val_loader, model, criterion, opt):
    """validation"""
    model.eval()

    batch_time = AverageMeter()
    top1 = AverageMeter()

    with torch.no_grad():
        end = time.time()
        for idx, (images, labels) in enumerate(val_loader):
            images = images.float().cuda()
            labels = labels.cuda()
            bsz = labels.shape[0]

            # forward
            output = model(images)

            # update metric
            acc1, acc5 = accuracy(output, labels, topk=(1, 5))
            top1.update(acc1[0], bsz)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if idx % opt.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Acc@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                       idx, len(val_loader), batch_time=batch_time,
                       top1=top1))

    print(' * Acc@1 {top1.avg:.3f}'.format(top1=top1))
    return top1.avg


def main():
    best_acc = 0
    opt = parse_option()

    # fix the seed for reproducibility
    torch.manual_seed(opt.seed)
    np.random.seed(opt.seed)
    random.seed(opt.seed)
    cudnn.benchmark = True

    if opt.use_amp:
        scaler = torch.cuda.amp.GradScaler()
    else:
        scaler = None

    # build data loader
    train_loader, val_loader, num_samples = set_loader(opt)

    # build model and criterion
    model, criterion = set_model(opt, num_samples)

    print("compiling the model...")
    model = torch.compile(model)
    print("done.")

    # build optimizer
    optimizer = set_optimizer(opt, model)

    # tensorboard
    logger = tb_logger.Logger(logdir=opt.tb_folder, flush_secs=2)

    # training routine
    for epoch in range(1, opt.epochs + 1):
        adjust_learning_rate(opt, optimizer, epoch)

        # train for one epoch
        time1 = time.time()
        loss, train_acc = train(train_loader, model, criterion, optimizer, epoch, scaler, opt)
        time2 = time.time()
        print('epoch {}, total time {:.2f}'.format(epoch, time2 - time1))

        # tensorboard logger
        logger.log_value('train_loss', loss, epoch)
        logger.log_value('train_acc', train_acc, epoch)
        logger.log_value('learning_rate', optimizer.param_groups[0]['lr'], epoch)

        # evaluation
        val_acc = validate(val_loader, model, criterion, opt)
        logger.log_value('val_acc', val_acc, epoch)

        if val_acc > best_acc:
            best_acc = val_acc

        if epoch % opt.save_freq == 0:
            save_file = os.path.join(
                opt.save_folder, 'ckpt_epoch_{epoch}.pth'.format(epoch=epoch))
            save_model(model, optimizer, opt, epoch, save_file)

    # save the last model
    save_file = os.path.join(
        opt.save_folder, 'last.pth')
    save_model(model, optimizer, opt, opt.epochs, save_file)

    print('best accuracy: {:.2f}'.format(best_acc))


if __name__ == '__main__':
    main()
