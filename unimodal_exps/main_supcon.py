from __future__ import print_function

import os
import sys
import argparse
import time
import math
import json
import numpy as np
import pickle

import tensorboard_logger as tb_logger
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from torchvision import transforms, datasets

from util import TwoCropTransform, AverageMeter
from util import adjust_learning_rate, warmup_learning_rate
from util import set_optimizer, save_model
from util import IndexImageFolder, Index_CIFAR10, Index_CIFAR100, Index_SVHN, CIFAR10_LT, CIFAR100_LT, ImageNet_LT, Index_iNat

from networks.resnet import SupConResNet
from losses import SupConLoss, BatchIndependentLoss, SogCLR_Loss, SogCLR_DRO_Loss, SogCLR_DRO_Loss_v2, spectral_cl_loss, flatclr_loss
from losses import BarlowTwins, VICReg, Projector, simco_loss, tau_simclr, hcl_loss

from new_losses import TempGenerator, TempGenerator_v2, iSogCLR_New_v1_Loss


from PIL import ImageFilter
import random

from signal import signal, SIGPIPE, SIG_DFL, SIG_IGN
signal(SIGPIPE, SIG_IGN)



class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x


def parse_option():
    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--print_freq', type=int, default=20,
                        help='print frequency')
    parser.add_argument('--save_freq', type=int, default=200,
                        help='save frequency')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='batch_size')
    parser.add_argument('--num_workers', type=int, default=20,
                        help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=200,
                        help='number of training epochs')

    # optimization
    parser.add_argument('--learning_rate', type=float, default=0.03,
                        help='learning rate')
    parser.add_argument('--lr_decay_epochs', type=str, default='160,190',
                        help='where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1,
                        help='decay rate for learning rate')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='momentum')

    # model dataset
    parser.add_argument('--model', type=str, default='resnet18')
    parser.add_argument('--feat_dim', type=int, default=128)
    parser.add_argument('--dataset', type=str, default='cifar10',
                        choices=['cifar10', 'cifar100', 'cifar10-lt', 'cifar100-lt', 'svhn', 'imagenet100', 'imagenet-lt', 'imagenet','iNat'], help='dataset')
    parser.add_argument('--mean', type=str, help='mean of dataset in path in form of str tuple')
    parser.add_argument('--std', type=str, help='std of dataset in path in form of str tuple')
    parser.add_argument('--data_folder', type=str, default=None, help='path to custom dataset')
    parser.add_argument('--size', type=int, default=32, help='parameter for RandomResizedCrop')

    # method
    parser.add_argument('--method', type=str, default='SimCLR',
                        choices=['SupCon', 'SimCLR', 'SogCLR', 'SpecCLR', 'FlatCLR', 'MoCo', 'SiMo', 'BarlowTwins', 'VICReg', 'SimCo','TaU_simclr','HCL'], help='choose method')

    parser.add_argument('--vicreg_coeff', type=float, default=25.0)
    parser.add_argument('--barlowtwins_lambd', type=float, default=0.0051)
    parser.add_argument('--specclr_mu', type=float, default=1.0)

    # temperature
    parser.add_argument('--temp', type=float, default=0.07,
                        help='temperature for loss function')

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

    # for BI mod
    parser.add_argument('--BI_mod', action='store_true',
                        help='enable Batch-Independent mod')
    parser.add_argument('--BI_gamma', default=0.9, type=float,
                        help='gamma for BI mod')

    parser.add_argument('--desc', type=str, default='')

    # for DRO mod
    parser.add_argument('--DRO_mod', action='store_true')
    parser.add_argument('--add_surrogate_func', action='store_true')
    parser.add_argument('--new_imp_type', default='', type=str) # 'v1' or 'v2' or None ('')
    parser.add_argument('--v1_c', default=1.0, type=float)
    parser.add_argument('--DRO_gamma', default=0.8, type=float)
    parser.add_argument('--DRO_rho', default=6.0, type=float)
    parser.add_argument('--DRO_beta_u', default=0.9, type=float)
    parser.add_argument('--DRO_tau_init', default=0.1, type=float)
    parser.add_argument('--DRO_eta_init', default=0.001, type=float)
    parser.add_argument('--DRO_eta_min', default=0.0001, type=float)
    parser.add_argument('--DRO_eta_sched', type=str, default='cosine', choices=['cosine','exp','const'])
    parser.add_argument('--DRO_eta_exp_gamma', default=0.6, type=float)

    opt = parser.parse_args()

    opt.weight_decay = 5e-4
    if 'imagenet' in opt.dataset or 'iNat' in opt.dataset:
        opt.weight_decay = 1e-4

    # set the path according to the environment
    if opt.data_folder is None:
        opt.data_folder = './datasets/'
    opt.model_path = './save/{}/models'.format(opt.desc)
    opt.tb_path = './save/{}/tensorboard'.format(opt.desc)

    iterations = opt.lr_decay_epochs.split(',')
    opt.lr_decay_epochs = list([])
    for it in iterations:
        opt.lr_decay_epochs.append(int(it))

    opt.model_name = '{}_{}_{}_lr_{}_decay_{}_bsz_{}_temp_{}_trial_{}'.\
        format(opt.method, opt.dataset, opt.model, opt.learning_rate,
               opt.weight_decay, opt.batch_size, opt.temp, opt.trial)

    if opt.cosine:
        opt.model_name = '{}_cosine'.format(opt.model_name)

    if opt.BI_mod:
        opt.model_name = '{}_BI_gamma_{}'.format(opt.model_name, opt.BI_gamma)

    if opt.DRO_mod:
        opt.model_name = '{}_DRO_tau_{}_gamma_{}_beta_u_{}_rho_{}_eta_{}-{}_sched_{}-{}'.format(opt.model_name, opt.DRO_tau_init, opt.DRO_gamma, 
                   opt.DRO_beta_u, opt.DRO_rho, opt.DRO_eta_init, opt.DRO_eta_min, opt.DRO_eta_sched, opt.DRO_eta_exp_gamma)

    # warm-up for large-batch training
    if opt.batch_size > 255:
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

    opt.model_name = '{}_desc_{}'.format(opt.model_name, opt.desc)

    opt.tb_folder = os.path.join(opt.tb_path, opt.model_name)
    if not os.path.isdir(opt.tb_folder):
        os.makedirs(opt.tb_folder)

    opt.save_folder = os.path.join(opt.model_path, opt.model_name)
    if not os.path.isdir(opt.save_folder):
        os.makedirs(opt.save_folder)

    opt.validate = True

    return opt


def set_loader(opt):
    # construct data loader
    if opt.dataset == 'cifar10' or opt.dataset == 'cifar10-lt':
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2023, 0.1994, 0.2010)
    elif opt.dataset == 'cifar100' or opt.dataset == 'cifar100-lt':
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
        transforms.RandomResizedCrop(opt.size, scale=(0.2, 1.0)),
        transforms.RandomApply([
          transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])

    if opt.dataset == 'cifar10':
        if not (opt.DRO_mod or opt.BI_mod):
            train_dataset = datasets.CIFAR10(root=opt.data_folder, train=True,
                                             transform=TwoCropTransform(train_transform),
                                             download=True)
        else:
            train_dataset = Index_CIFAR10(root=opt.data_folder, train=True,
                                             transform=TwoCropTransform(train_transform))

        val_dataset = datasets.CIFAR10(root=opt.data_folder, train=False,
                                         transform=TwoCropTransform(train_transform),
                                         download=True)
    elif opt.dataset == 'cifar100':
        if not (opt.DRO_mod or opt.BI_mod):
            train_dataset = datasets.CIFAR100(root=opt.data_folder, train=True,
                                              transform=TwoCropTransform(train_transform),
                                              download=True)
        else:
            train_dataset = Index_CIFAR100(root=opt.data_folder, train=True,
                                             transform=TwoCropTransform(train_transform))

        val_dataset = datasets.CIFAR100(root=opt.data_folder, train=False,
                                          transform=TwoCropTransform(train_transform),
                                          download=True)
    elif opt.dataset == 'svhn':
        if not (opt.DRO_mod or opt.BI_mod):
            train_dataset = datasets.SVHN(root=opt.data_folder, split='train',
                                              transform=TwoCropTransform(train_transform),
                                              download=True)
        else:
            train_dataset = Index_SVHN(root=opt.data_folder, split='train',
                                              transform=TwoCropTransform(train_transform))

        val_dataset = datasets.SVHN(root=opt.data_folder, split='test',
                                          transform=TwoCropTransform(train_transform),
                                          download=True)
    elif opt.dataset == 'imagenet-lt':
        return_idx = False
        if opt.DRO_mod or opt.BI_mod:
            return_idx = True

        opt.validate = False

        train_dataset = ImageNet_LT(root=opt.data_folder, txt='./datasets/ImageNet_LT/ImageNet_LT_train.txt',
                                    transform=TwoCropTransform(train_transform), return_idx=return_idx)

        val_dataset = datasets.ImageFolder(root=opt.data_folder+'val', 
                                           transform=TwoCropTransform(train_transform))

    elif opt.dataset == 'cifar10-lt':
        return_idx = False
        if opt.DRO_mod or opt.BI_mod:
            return_idx = True

        train_dataset = CIFAR10_LT(root=opt.data_folder, train=True, download=True,
                                   transform=TwoCropTransform(train_transform), return_idx=return_idx)

        val_dataset = datasets.CIFAR10(root=opt.data_folder, train=False,
                                         transform=TwoCropTransform(train_transform),
                                         download=True)
        print('cifar10-lt:', train_dataset.get_cls_num_list())

    elif opt.dataset == 'cifar100-lt':
        return_idx = False
        if opt.DRO_mod or opt.BI_mod:
            return_idx = True

        train_dataset = CIFAR100_LT(root=opt.data_folder, train=True, download=True,
                                    transform=TwoCropTransform(train_transform), return_idx=return_idx)
        
        val_dataset = datasets.CIFAR100(root=opt.data_folder, train=False,
                                         transform=TwoCropTransform(train_transform),
                                         download=True)
        print('cifar100-lt:', train_dataset.get_cls_num_list())

    elif 'imagenet' in opt.dataset:
        if opt.dataset == 'imagenet':
            opt.validate = False

        if opt.BI_mod or opt.DRO_mod:
            train_dataset = IndexImageFolder(root=opt.data_folder+'train',
                                             transform=TwoCropTransform(train_transform))
        else:
            train_dataset = datasets.ImageFolder(root=opt.data_folder+'train',
                                                 transform=TwoCropTransform(train_transform))

        val_dataset = datasets.ImageFolder(root=opt.data_folder+'val', 
                                           transform=TwoCropTransform(train_transform))

    elif 'iNat' in opt.dataset:
        opt.validate = False

        if opt.BI_mod or opt.DRO_mod:
            train_dataset = IndexImageFolder(root=opt.data_folder,
                                             transform=TwoCropTransform(train_transform))
        else:
            train_dataset = datasets.ImageFolder(root=opt.data_folder,
                                                 transform=TwoCropTransform(train_transform))

        val_dataset = datasets.ImageFolder(root=opt.data_folder,
                                            transform=TwoCropTransform(train_transform))
        
    else:
        raise ValueError(opt.dataset)

    print("datasets length:", len(train_dataset))

    train_sampler = None
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=opt.batch_size, shuffle=(train_sampler is None),
        num_workers=opt.num_workers, pin_memory=True, sampler=train_sampler, drop_last=True, prefetch_factor=4)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=512, shuffle=False, num_workers=8, pin_memory=True)

    return train_loader, len(train_dataset), val_loader


def set_model(opt):
    if opt.method == 'MoCo':
        model = MoCo(base_encoder_name='resnet50', dim=opt.feat_dim)
    elif opt.method == 'SiMo':
        model = SiMo(bsz=opt.batch_size, base_encoder_name='resnet50', T=opt.temp, dim=opt.feat_dim)
    else:
        model = SupConResNet(name=opt.model, feat_dim=opt.feat_dim, add_one_entry=opt.method=='TaU_simclr')

    if opt.method == 'MoCo':
        if opt.DRO_mod:
            criterion = SogCLR_DRO_M_Loss(gamma=opt.DRO_gamma, rho=opt.DRO_rho, tau_init=opt.DRO_tau_init, eta_init=opt.DRO_eta_init, beta_u=opt.DRO_beta_u)
        else:
            criterion = torch.nn.CrossEntropyLoss()
    elif opt.method == 'SiMo':
        criterion = None
    elif opt.method == 'BarlowTwins':
        #criterion = BarlowTwins(dim_size=opt.feat_dim, batch_size=opt.batch_size, lambd=opt.barlowtwins_lambd)
        criterion = BarlowTwins(dim_size=8192, batch_size=opt.batch_size, lambd=opt.barlowtwins_lambd)          # using Projector
    elif opt.method == 'VICReg':
        #criterion = VICReg(dim_size=opt.feat_dim, batch_size=opt.batch_size, sim_coeff=opt.vicreg_coeff, std_coeff=opt.vicreg_coeff)
        criterion = VICReg(dim_size=8192, batch_size=opt.batch_size, sim_coeff=opt.vicreg_coeff, std_coeff=opt.vicreg_coeff)   # using Projector
    elif opt.method == 'SogCLR':
        criterion = SogCLR_Loss(gamma=opt.BI_gamma, temperature=opt.temp)
    else:
        if opt.BI_mod:
            criterion = BatchIndependentLoss(gamma=opt.BI_gamma, temperature=opt.temp)
        elif opt.DRO_mod:
            if len(opt.new_imp_type) > 0:
                if opt.new_imp_type == 'v1':
                    criterion = iSogCLR_New_v1_Loss(gamma=opt.DRO_gamma, rho=opt.DRO_rho, tau_init=opt.DRO_tau_init, eta_init=opt.DRO_eta_init, 
                                                    bsz=opt.batch_size, beta_u=opt.DRO_beta_u, c=opt.v1_c)
                elif opt.new_imp_type == 'v2':
                    pass
                else:
                    assert 0
            else:
                if not opt.add_surrogate_func:
                    criterion = SogCLR_DRO_Loss(gamma=opt.DRO_gamma, rho=opt.DRO_rho, tau_init=opt.DRO_tau_init, eta_init=opt.DRO_eta_init, bsz=opt.batch_size,
                                        beta_u=opt.DRO_beta_u, eta_min=opt.DRO_eta_min, eta_sched=opt.DRO_eta_sched, eta_exp_gamma=opt.DRO_eta_exp_gamma)
                else:
                    criterion = SogCLR_DRO_Loss_v2(gamma=opt.DRO_gamma, rho=opt.DRO_rho, tau_init=opt.DRO_tau_init, eta_init=opt.DRO_eta_init, bsz=opt.batch_size,
                                        beta_u=opt.DRO_beta_u, eta_min=opt.DRO_eta_min, eta_sched=opt.DRO_eta_sched, eta_exp_gamma=opt.DRO_eta_exp_gamma)
        else:
            criterion = SupConLoss(batch_size=opt.batch_size, temperature=opt.temp)

    # for Barlow Twins and VICReg, use the Projector!
    #if opt.method in ['BarlowTwins', 'VICReg']:

    # all methods use the same projector
    projector = Projector(net_type=opt.model, add_one_entry=opt.method=='TaU_simclr')
    model.head = projector

    # add the temperature generator into the model
    temp_generator = TempGenerator_v2(feature_dim=opt.feat_dim)
    model.temp_generator = temp_generator

    # enable synchronized Batch Normalization
    if opt.syncBN:
        model = apex.parallel.convert_syncbn_model(model)

    if torch.cuda.is_available():
        if torch.cuda.device_count() > 1:
            model.encoder = torch.nn.DataParallel(model.encoder)
        model = model.cuda()
        if criterion is not None:
            criterion = criterion.cuda()
        cudnn.benchmark = True

    return model, criterion


def train(train_loader, model, criterion, optimizer, epoch, max_epoch, opt, scaler=None):
    """one epoch training"""
    model.train()
    model.temp_generator.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    end = time.time()
    for idx, (items, labels) in enumerate(train_loader):

        data_time.update(time.time() - end)

        if opt.BI_mod or opt.DRO_mod:
            index, images = items
        else:
            images = items

        images = torch.cat([images[0], images[1]], dim=0)
        if torch.cuda.is_available():
            images = images.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)
        bsz = labels.shape[0]

        # warm-up learning rate
        warmup_learning_rate(opt, epoch, idx, len(train_loader), optimizer)

        optimizer.zero_grad()

        # compute loss
        if opt.use_amp:
            with torch.cuda.amp.autocast():

                if opt.method == 'MoCo':
                    img_q, img_k = torch.split(images, [bsz, bsz], dim=0)
                    output, target = model(img_q, img_k)
                    if opt.DRO_mod:
                        loss, avg_tau, eta_cur, grad_tau, b = criterion(index, output)
                    else:
                        output /= opt.temp
                        loss = criterion(output, target)
                elif opt.method == 'SiMo':
                    img_q, img_k = torch.split(images, [bsz, bsz], dim=0)
                    loss = model(img_q, img_k)
                else:
                    encoder_out, raw_features = model(images)
                    norm_features = F.normalize(raw_features, dim=1)
                    f1, f2 = torch.split(norm_features, [bsz, bsz], dim=0)
                    features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
                    torch.cuda.empty_cache()

                    if opt.method == 'FlatCLR':
                        logits, labels = flatclr_loss(norm_features, bsz, temperature=opt.temp)
                        v = torch.logsumexp(logits, dim=1, keepdim=True)
                        loss_vec = torch.exp(v - v.detach())
                        
                        assert loss_vec.shape == (len(logits),1)
                        dummy_logits = torch.cat([torch.zeros(logits.size(0), 1).cuda(), logits], 1)
                        loss = loss_vec.mean() - 1 + torch.nn.CrossEntropyLoss()(dummy_logits, labels).detach()

                    elif opt.method == 'SpecCLR':
                        x, y = torch.split(raw_features, [bsz, bsz], dim=0)
                        loss, _ = spectral_cl_loss(z1=x, z2=y, mu=opt.specclr_mu)
                        loss = loss.mean()

                    elif opt.method == 'SimCo':
                        x, y = torch.split(raw_features, [bsz, bsz], dim=0)
                        loss = simco_loss(x, y)

                    elif opt.method == 'TaU_simclr':
                        x, y = torch.split(raw_features, [bsz, bsz], dim=0)
                        loc1, temp1 = x[:, :-1], x[:, -1].unsqueeze(1)
                        loc2, temp2 = y[:, :-1], y[:, -1].unsqueeze(1)
                        loss = tau_simclr(loc1, temp1, loc2, temp2)

                    elif opt.method == 'BarlowTwins':
                        x, y = torch.split(raw_features, [bsz, bsz], dim=0)
                        loss = criterion(z1=x, z2=y)

                    elif opt.method == 'VICReg':
                        x, y = torch.split(raw_features, [bsz, bsz], dim=0)
                        loss = criterion(x=x, y=y)

                    elif opt.method == 'SogCLR':
                        loss = criterion(index, f1, f2)

                    elif opt.method == 'HCL':
                        if opt.dataset == 'cifar10':
                            tau_plus = 0.1 
                        elif opt.dataset == 'cifar100':
                            tau_plus = 0.05
                        else:
                            assert 0, opt.dataset + " does not supports HCL"

                        loss = hcl_loss(f1, f2, tau_plus, batch_size=opt.batch_size)

                    elif opt.method == 'SupCon':
                        if opt.BI_mod:
                            loss = criterion(index, features, labels)
                        else:
                            loss = criterion(features, labels)
                    elif opt.method == 'SimCLR':
                        if opt.BI_mod:
                            loss = criterion(index, features)
                        elif opt.DRO_mod:
                            if opt.new_imp_type == 'v1':
                                t1, t2 = torch.split(encoder_out, [bsz, bsz], dim=0)
                                tau, att_weights = model.temp_generator(t1.detach(), t2.detach())
                                loss, avg_tau, eta_cur, grad_tau, b = criterion(index, features, tau, att_weights)
                            else:
                                loss, avg_tau, eta_cur, grad_tau, b = criterion(index, features, epoch, max_epoch)
                        else:
                            loss = criterion(features)
                    else:
                        raise ValueError('contrastive method not supported: {}'.
                                         format(opt.method))

            # update metric
            losses.update(loss.item(), bsz)

            # SGD
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            if opt.method == 'MoCo':
                img_q, img_k = torch.split(images, [bsz, bsz], dim=0)
                output, target = model(img_q, img_k)
                if opt.DRO_mod:
                    loss, avg_tau, eta_cur, grad_tau, b = criterion(index, output)
                else:
                    output /= opt.temp
                    loss = criterion(output, target)
            elif opt.method == 'SiMo':
                img_q, img_k = torch.split(images, [bsz, bsz], dim=0)
                loss = model(img_q, img_k)
            else:
                encoder_out, raw_features = model(images)
                norm_features = F.normalize(raw_features, dim=1)
                f1, f2 = torch.split(norm_features, [bsz, bsz], dim=0)
                features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
                torch.cuda.empty_cache()

                if opt.method == 'FlatCLR':
                    logits, labels = flatclr_loss(norm_features, bsz, temperature=opt.temp)
                    v = torch.logsumexp(logits, dim=1, keepdim=True)
                    loss_vec = torch.exp(v - v.detach())
                    
                    assert loss_vec.shape == (len(logits),1)
                    dummy_logits = torch.cat([torch.zeros(logits.size(0), 1).cuda(), logits], 1)
                    loss = loss_vec.mean() - 1 + torch.nn.CrossEntropyLoss()(dummy_logits, labels).detach()

                elif opt.method == 'SpecCLR':
                    x, y = torch.split(raw_features, [bsz, bsz], dim=0)
                    loss, _ = spectral_cl_loss(z1=x, z2=y, mu=opt.specclr_mu)
                    loss = loss.mean()

                elif opt.method == 'SimCo':
                    x, y = torch.split(raw_features, [bsz, bsz], dim=0)
                    loss = simco_loss(x, y)

                elif opt.method == 'TaU_simclr':
                    x, y = torch.split(raw_features, [bsz, bsz], dim=0)
                    loc1, temp1 = x[:, :-1], x[:, -1].unsqueeze(1)
                    loc2, temp2 = y[:, :-1], y[:, -1].unsqueeze(1)
                    loss = tau_simclr(loc1, temp1, loc2, temp2)

                elif opt.method == 'BarlowTwins':
                    x, y = torch.split(raw_features, [bsz, bsz], dim=0)
                    loss = criterion(z1=x, z2=y)

                elif opt.method == 'VICReg':
                    x, y = torch.split(raw_features, [bsz, bsz], dim=0)
                    loss = criterion(x=x, y=y)

                elif opt.method == 'SogCLR':
                    loss = criterion(index, f1, f2)

                elif opt.method == 'HCL':
                    if opt.dataset == 'cifar10':
                        tau_plus = 0.1 
                    elif opt.dataset == 'cifar100':
                        tau_plus = 0.05
                    else:
                        assert 0, opt.dataset + " does not supports HCL"

                    loss = hcl_loss(f1, f2, tau_plus, batch_size=opt.batch_size)

                elif opt.method == 'SupCon':
                    if opt.BI_mod:
                        loss = criterion(index, features, labels)
                    else:
                        loss = criterion(features, labels)
                elif opt.method == 'SimCLR':
                    if opt.BI_mod:
                        loss = criterion(index, features)
                    elif opt.DRO_mod:
                        if opt.new_imp_type == 'v1':
                            t1, t2 = torch.split(encoder_out, [bsz, bsz], dim=0)
                            tau, att_weights = model.temp_generator(t1.detach(), t2.detach())
                            loss, avg_tau, eta_cur, grad_tau, b = criterion(index, features, tau, att_weights)
                        else:
                            loss, avg_tau, eta_cur, grad_tau, b = criterion(index, features, epoch, max_epoch)
                    else:
                        loss = criterion(features)
                else:
                    raise ValueError('contrastive method not supported: {}'.
                                     format(opt.method))

            # update metric
            losses.update(loss.item(), bsz)

            # SGD
            loss.backward()
            optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # print info
        if (idx + 1) % opt.print_freq == 0:
            info_str = 'Train: [{0}][{1}/{2}] ' \
                       'BT {batch_time.val:.3f} ({batch_time.avg:.3f}) ' \
                       'DT {data_time.val:.3f} ({data_time.avg:.3f}) ' \
                       'loss {loss.val:.3f} ({loss.avg:.3f}) '.format(
                        epoch, idx + 1, len(train_loader), batch_time=batch_time,
                        data_time=data_time, loss=losses)
            if opt.DRO_mod:
                info_str += 'avg_tau {avg_tau:.6f} eta_cur {eta_cur:.6f} grad_tau {grad_tau:.6f} b {b:.6f}'.format(avg_tau=avg_tau, eta_cur=eta_cur, grad_tau=grad_tau, b=b)

            print(info_str)
            sys.stdout.flush()

    return losses.avg


def validate(val_loader, model):
    """compute contrastice accuracy"""
    model.eval()
    top1 = AverageMeter()

    with torch.no_grad():
        for images, _ in val_loader:

            images = torch.cat([images[0], images[1]], dim=0)
            if torch.cuda.is_available():
                images = images.cuda(non_blocking=True)
            bsz = images.shape[0] // 2

            features = F.normalize(model.encoder(images), dim=1)
            f1, f2 = torch.split(features, [bsz, bsz], dim=0)

            logits = torch.einsum('i d, j d -> i j', f1, f2)

            _, preds = torch.topk(logits, k=1, dim=1)
            labels = torch.arange(bsz)

            acc = labels.eq(preds.squeeze().cpu()).float().sum() / bsz
            top1.update(acc.item(), bsz)

    print("Contrastive accuracy on val set:", top1.avg)


def main():
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
    train_loader, train_data_len, val_loader = set_loader(opt)

    # build model and criterion
    model, criterion = set_model(opt)

    # build optimizer
    optimizer = set_optimizer(opt, model)

    # tensorboard
    logger = tb_logger.Logger(logdir=opt.tb_folder, flush_secs=2)

    # training routine
    for epoch in range(1, opt.epochs + 1):
        lr = adjust_learning_rate(opt, optimizer, epoch)

        # train for one epoch
        time1 = time.time()
        loss = train(train_loader, model, criterion, optimizer, epoch, opt.epochs, opt, scaler)
        time2 = time.time()
        print('epoch {}, lr {:.4f}, total time {:.2f}'.format(epoch, lr, time2 - time1))

        if opt.validate:
            validate(val_loader, model)

        # tensorboard logger
        logger.log_value('loss', loss, epoch)
        logger.log_value('learning_rate', optimizer.param_groups[0]['lr'], epoch)

        if epoch % opt.save_freq == 0:
            save_file = os.path.join(
                opt.save_folder, 'ckpt_epoch_{epoch}.pth'.format(epoch=epoch))
            save_model(model, optimizer, opt, epoch, save_file)

        if opt.DRO_mod and epoch % 10 == 0 and opt.new_imp_type != 'v1':
            print("saving tau...")
            tau = criterion.tau.clone().cpu().numpy()[:train_data_len]
            with open(os.path.join(opt.save_folder, 'tau_'+str(epoch)+'.pkl'), 'wb') as f:
                pickle.dump(tau, f, protocol=pickle.HIGHEST_PROTOCOL)

    # save the last model
    save_file = os.path.join(
        opt.save_folder, 'last.pth')
    save_model(model, optimizer, opt, opt.epochs, save_file)

    # save the config
    json.dump(opt.__dict__, open(os.path.join(opt.save_folder, 'args.json'), 'w'), indent=2) 

    # save a model in the working directory
    save_model(model, optimizer, opt, opt.epochs, opt.desc+'.pth')


if __name__ == '__main__':
    main()
