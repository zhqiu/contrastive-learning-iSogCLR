from __future__ import print_function

import math
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import Dataset
from torch.optim.optimizer import Optimizer, required

from torchvision.datasets.folder import ImageFolder
from torchvision import transforms, datasets

import os
from PIL import Image

from lars import LARS

class TwoCropTransform:
    """Create two crops of the same image"""
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, x):
        return [self.transform(x), self.transform(x)]


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def adjust_learning_rate(args, optimizer, epoch):
    lr = args.learning_rate
    if args.cosine:
        eta_min = lr * (args.lr_decay_rate ** 3)
        lr = eta_min + (lr - eta_min) * (
                1 + math.cos(math.pi * epoch / args.epochs)) / 2
    else:
        steps = np.sum(epoch > np.asarray(args.lr_decay_epochs))
        if steps > 0:
            lr = lr * (args.lr_decay_rate ** steps)

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return lr


def warmup_learning_rate(args, epoch, batch_id, total_batches, optimizer):
    if args.warm and epoch <= args.warm_epochs:
        p = (batch_id + (epoch - 1) * total_batches) / \
            (args.warm_epochs * total_batches)
        lr = args.warmup_from + p * (args.warmup_to - args.warmup_from)

        for param_group in optimizer.param_groups:
            param_group['lr'] = lr


def set_optimizer(opt, model, model_2=None, opt_type='LARS'):
    if model_2 is not None:
        params = list(model.parameters()) + list(model_2.parameters())
    else:
        params = list(model.parameters())
    if opt_type == 'SGD':
        optimizer = optim.SGD(params, lr=opt.learning_rate, momentum=opt.momentum, weight_decay=opt.weight_decay)
    elif opt_type == 'LARS':
        optimizer = LARS(params, lr=opt.learning_rate, momentum=opt.momentum, weight_decay=opt.weight_decay)
    else:
        assert 0

    return optimizer


def save_model(model, optimizer, opt, epoch, save_file):
    print('==> Saving...')
    state = {
        'opt': opt,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch,
    }
    torch.save(state, save_file)
    del state


# for SogCLR and SogCLR_DRO
class IndexImageFolder(ImageFolder):
    def __init__(self, root, transform):
        super().__init__(root, transform=transform)

    def __getitem__(self, index):
        sample, target = super().__getitem__(index)

        return (index, sample), target


class ImageNet_LT(Dataset):
    def __init__(self, root, txt, transform=None, return_idx=False):
        self.img_path = []
        self.targets = []
        self.transform = transform
        self.return_idx = return_idx

        with open(txt) as f:
            for line in f:
                self.img_path.append(os.path.join(root, line.split(' ')[0]))
                self.targets.append(int(line.split(' ')[1]))

    def __len__(self):
        return len(self.img_path)

    def __getitem__(self, index):
        path = self.img_path[index]
        label = self.targets[index]

        with open(path, 'rb') as f:
            sample = Image.open(f).convert('RGB')

        if self.transform is not None:
            sample = self.transform(sample)

        if self.return_idx:
            return (index, sample), label
        else:
            return sample, label


class My_iNaturalist(Dataset):
    def __init__(self, root, content_file, transform, use_fine_label=False):
        self.root = root
        self.img_path = []
        self.labels = []
        self.transform = transform
        self.label_map = {'Actinopterygii':0, 'Amphibia':1, 'Animalia':2,
                          'Arachnida':3, 'Aves':4, 'Bacteria':5, 'Chromista':6,
                          'Fungi':7, 'Insecta':8, 'Mammalia':9, 'Mollusca':10,
                          'Plantae':11, 'Protozoa':12, 'Reptilia':13}

        self.use_fine_label = use_fine_label
        if use_fine_label:
            self.fine_label_map = {}
            self.fine_label_id = 0

        with open(content_file, 'r') as f:
            FileLines = f.readlines()
            FileLines = [x.strip() for x in FileLines]
            for entry in FileLines:
                self.img_path.append(entry)

                if use_fine_label:
                    fine_label_name = '_'.join(entry.split('/')[1:3])
                    if fine_label_name not in self.fine_label_map:
                        self.fine_label_map[fine_label_name] = self.fine_label_id
                        self.fine_label_id += 1
                    assert fine_label_name in self.fine_label_map
                    self.labels.append(self.fine_label_map[fine_label_name])

                else:  # just use coarse label
                    super_name = entry.split('/')[1]
                    assert super_name in self.label_map
                    self.labels.append(self.label_map[super_name])

    def __len__(self):
        return len(self.labels)

    def get_fine_label_nums(self):
        assert self.use_fine_label
        return self.fine_label_id

    def __getitem__(self, index):
        path = self.img_path[index]
        label = self.labels[index]

        with open(os.path.join(self.root, path), 'rb') as f:
            sample = Image.open(f).convert('RGB')

        if self.transform is not None:
            sample = self.transform(sample)

        return sample, label


class Index_iNat(datasets.INaturalist):
    def __init__(self, root, transform, version='2018', target_type='full'):
        super().__init__(root, transform=transform, version=version, target_type=target_type)

    def __getitem__(self, index):
        sample, target = super().__getitem__(index)

        return (index, sample), target



class Index_CIFAR10(datasets.CIFAR10):
    def __init__(self, root, train, transform):
        super().__init__(root, train=train, transform=transform)

    def __getitem__(self, index):
        sample, target = super().__getitem__(index)

        return (index, sample), target


class Index_CIFAR100(datasets.CIFAR100):
    def __init__(self, root, train, transform):
        super().__init__(root, train=train, transform=transform)

    def __getitem__(self, index):
        sample, target = super().__getitem__(index)

        return (index, sample), target


class Index_SVHN(datasets.SVHN):
    def __init__(self, root, split, transform):
        super().__init__(root, split=split, transform=transform)

    def __getitem__(self, index):
        sample, target = super().__getitem__(index)

        return (index, sample), target


class CIFAR10_LT(datasets.CIFAR10):
    cls_num = 10

    def __init__(self, root, imb_type='exp', imb_factor=0.01, rand_number=0, train=True,
                 transform=None, target_transform=None, download=True, return_idx=False):
        super(CIFAR10_LT, self).__init__(root, train, transform, target_transform, download)
        np.random.seed(rand_number)
        img_num_list = self.get_img_num_per_cls(self.cls_num, imb_type, imb_factor)
        self.gen_imbalanced_data(img_num_list)
        self.return_idx = return_idx

    def get_img_num_per_cls(self, cls_num, imb_type, imb_factor):
        img_max = len(self.data) / cls_num
        img_num_per_cls = []

        if imb_type == 'exp':
            for cls_idx in range(cls_num):
                num = img_max * (imb_factor ** (cls_idx / (cls_num - 1.0)))
                img_num_per_cls.append(int(num))
        elif imb_type == 'step':
            for cls_idx in range(cls_num // 2):
                img_num_per_cls.append(int(img_max))
            for cls_idx in range(cls_num // 2):
                img_num_per_cls.append(int(img_max * imb_factor))
        else:
            img_num_per_cls.extend([int(img_max)] * cls_num)

        return img_num_per_cls

    def gen_imbalanced_data(self, img_num_per_cls):
        new_data = []
        new_targets = []
        targets_np = np.array(self.targets, dtype=np.int64)
        classes = np.unique(targets_np)

        self.num_per_cls_dict = dict()
        for the_class, the_img_num in zip(classes, img_num_per_cls):
            self.num_per_cls_dict[the_class] = the_img_num
            idx = np.where(targets_np == the_class)[0]
            np.random.shuffle(idx)
            selec_idx = idx[:the_img_num]
            new_data.append(self.data[selec_idx, ...])
            new_targets.extend([the_class, ] * the_img_num)
        new_data = np.vstack(new_data)
        self.data = new_data
        self.targets = new_targets
        
    def get_cls_num_list(self):
        cls_num_list = []
        for i in range(self.cls_num):
            cls_num_list.append(self.num_per_cls_dict[i])
        return cls_num_list

    def __getitem__(self, index):
        sample, target = super().__getitem__(index)

        if self.return_idx:
            return (index, sample), target
        else:
            return sample, target


class CIFAR100_LT(CIFAR10_LT):
    base_folder = 'cifar-100-python'
    url = "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
    filename = "cifar-100-python.tar.gz"
    tgz_md5 = 'eb9058c3a382ffc7106e4002c42a8d85'
    train_list = [
        ['train', '16019d7e3df5f24257cddd939b257f8d'],
    ]

    test_list = [
        ['test', 'f0ef6b0ae62326f3e7ffdfab6717acfc'],
    ]
    meta = {
        'filename': 'meta',
        'key': 'fine_label_names',
        'md5': '7973b15100ade9c7d40fb424638fde48',
    }
    cls_num = 100

