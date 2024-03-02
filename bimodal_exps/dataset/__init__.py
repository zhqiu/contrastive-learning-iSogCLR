import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image

from dataset.caption_dataset import re_train_dataset, re_eval_dataset
from dataset.randaugment import RandomAugment


def create_train_dataset(dataset, args):
    
    normalize = transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
     
    train_transform = transforms.Compose([                        
            transforms.RandomResizedCrop(args.image_res, scale=(0.5, 1.0), interpolation=Image.BICUBIC),
            transforms.RandomHorizontalFlip(),
            RandomAugment(2, 7, isPIL=True, augs=['Identity','AutoContrast','Equalize','Brightness','Sharpness',
                                              'ShearX', 'ShearY', 'TranslateX', 'TranslateY', 'Rotate']),     
            transforms.ToTensor(),
            normalize,
        ]) 
    
    if dataset=='re':          
        train_dataset = re_train_dataset([args.train_file], train_transform, args.train_image_root)          
        return train_dataset

    else:
        assert 0, dataset + " is not supported."


def create_val_dataset(dataset, args, val_file, test_file, val_image_root):
    
    normalize = transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))

    test_transform = transforms.Compose([
        transforms.Resize((args.image_res,args.image_res), interpolation=Image.BICUBIC),
        transforms.ToTensor(),
        normalize,
        ])   
    
    if dataset=='re':
        val_dataset = re_eval_dataset(val_file, test_transform, val_image_root)  
        test_dataset = re_eval_dataset(test_file, test_transform, val_image_root)                
        return val_dataset, test_dataset

    else:
        assert 0, dataset + " is not supported."


def create_sampler(datasets, shuffles, num_tasks, global_rank):
    samplers = []
    for dataset,shuffle in zip(datasets, shuffles):
        sampler = torch.utils.data.DistributedSampler(dataset, num_replicas=num_tasks, rank=global_rank, shuffle=shuffle)
        samplers.append(sampler)
    return samplers


def create_train_loader(dataset, sampler, batch_size, num_workers, collate_fn):
    return DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=True,
                      sampler=sampler, shuffle=(sampler is None), collate_fn=collate_fn, drop_last=True, prefetch_factor=4)              


def create_val_loader(datasets, samplers, batch_size, num_workers, collate_fns):
    loaders = []
    for dataset, sampler, bs, n_worker, collate_fn in zip(datasets, samplers, batch_size, num_workers, collate_fns):
        shuffle = False
        drop_last = False
        loader = DataLoader(
            dataset,
            batch_size=bs,
            num_workers=n_worker,
            pin_memory=True,
            sampler=sampler,
            shuffle=shuffle,
            collate_fn=collate_fn,
            drop_last=drop_last,
        )              
        loaders.append(loader)
    return loaders  
