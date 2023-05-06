# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import torch
import torch.nn as nn

import torchvision.models as models


model_dict_imagenet = {  # these models are for imagenet datasets (imagenet-100 and imagenet-1K)
    'resnet18': [models.__dict__['resnet18'], 512],
    'resnet34': [models.__dict__['resnet34'], 512],
    'resnet50': [models.__dict__['resnet50'], 2048],
    'resnet101': [models.__dict__['resnet101'], 2048],
}


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class MoCo(nn.Module):
    """
    Only for ImageNet-100 and ImageNet-1K experiments!

    Build a MoCo model with: a query encoder, a key encoder, and a queue
    https://arxiv.org/abs/1911.05722
    """
    def __init__(self, base_encoder_name='resnet50', dim=128, K=65536, m=0.999, mlp=True, simclr=False):
        """
        dim: feature dimension (default: 128)
        K: queue size; number of negative keys (default: 65536)
        m: moco momentum of updating key encoder (default: 0.999)
        """
        super(MoCo, self).__init__()

        self.K = K
        self.m = m

        # create the encoders
        # num_classes is the output fc dimension
        model_encoder, dim_in = model_dict_imagenet[base_encoder_name]
        model_encoder_k, _ = model_dict_imagenet[base_encoder_name]

        self.encoder = model_encoder()
        self.encoder_k = model_encoder_k()

        self.encoder.fc = Identity()
        self.encoder_k.fc = Identity()

        if mlp:  # hack: brute-force replacement
            self.head = nn.Sequential(nn.Linear(dim_in, dim), nn.ReLU(inplace=True), nn.Linear(dim, dim))
            self.head_k = nn.Sequential(nn.Linear(dim_in, dim), nn.ReLU(inplace=True), nn.Linear(dim, dim))

        for param, param_k in zip(self.encoder.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

        for param, param_k in zip(self.head.parameters(), self.head_k.parameters()):
            param_k.data.copy_(param.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

        self.simclr = simclr

        # create the queue
        self.register_buffer("queue", torch.randn(dim, K))
        self.queue = nn.functional.normalize(self.queue, dim=0)

        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))


    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param, param_k in zip(self.encoder.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param.data * (1. - self.m)

        for param, param_k in zip(self.head.parameters(), self.head_k.parameters()):
            param_k.data = param_k.data * self.m + param.data * (1. - self.m)


    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)
        assert self.K % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue[:, ptr:ptr + batch_size] = keys.T
        ptr = (ptr + batch_size) % self.K  # move pointer

        self.queue_ptr[0] = ptr


    def forward(self, im_q, im_k):
        """
        Input:
            im_q: a batch of query images
            im_k: a batch of key images
        Output:
            logits, targets
        """

        # compute query features
        q = self.head(self.encoder(im_q))  # queries: NxC
        q = nn.functional.normalize(q, dim=1)

        # compute key features
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()  # update the key encoder

            k = self.head_k(self.encoder_k(im_k))  # keys: NxC
            k = nn.functional.normalize(k, dim=1)

        # compute logits
        # Einstein sum is more intuitive
        # positive logits: Nx1
        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
        # negative logits: NxK
        l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])

        if self.simclr:
            # use simclr-style negative samples: NxN
            l_neg_simclr = torch.einsum('nc,mc->nm', [q, q])
            l_neg_simclr *= (1 - torch.eye(q.shape[0], q.shape[0])).cuda()

            # logits: Nx(1+(K+N))
            logits = torch.cat([l_pos, l_neg, l_neg_simclr], dim=1)

        else:
            # logits: Nx(1+K)
            logits = torch.cat([l_pos, l_neg], dim=1)

        # labels: positive key indicators
        labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()

        # dequeue and enqueue
        self._dequeue_and_enqueue(k)

        return logits, labels


"""
    https://github.com/poodarchu/SelfSup/blob/master/examples/simo/simo.res50.scratch.imagenet.224size.256bs.200e/simo.py
"""
class SiMo(nn.Module):
    """
    Only for ImageNet-100 and ImageNet-1K experiments!

    Build a MoCo model with: a query encoder, a key encoder, and a queue
    https://arxiv.org/abs/1911.05722
    """
    def __init__(self, bsz=256, base_encoder_name='resnet50', dim=128, m=0.999, mlp=True, T=0.1, alpha=256):
        """
        dim: feature dimension (default: 128)
        m: moco momentum of updating key encoder (default: 0.999)
        """
        super(SiMo, self).__init__()

        self.m = m
        self.T = T
        self.alpha = alpha
        self.bsz = bsz

        # create the encoders
        # num_classes is the output fc dimension
        model_encoder, dim_in = model_dict_imagenet[base_encoder_name]
        model_encoder_k, _ = model_dict_imagenet[base_encoder_name]

        self.encoder = model_encoder()
        self.encoder_k = model_encoder_k()

        self.encoder.fc = Identity()
        self.encoder_k.fc = Identity()

        if mlp:  # hack: brute-force replacement
            self.head = nn.Sequential(nn.Linear(dim_in, dim), nn.ReLU(inplace=True), nn.Linear(dim, dim))
            self.head_k = nn.Sequential(nn.Linear(dim_in, dim), nn.ReLU(inplace=True), nn.Linear(dim, dim))

        for param, param_k in zip(self.encoder.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

        for param, param_k in zip(self.head.parameters(), self.head_k.parameters()):
            param_k.data.copy_(param.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

        self.similarity_f = nn.CosineSimilarity(dim=2)

        pos_mask_i, neg_mask_i = self.mask_correlated_samples(world_size=1, device_size=self.bsz)

        self.pos_mask_i = pos_mask_i.cuda()
        self.neg_mask_i = neg_mask_i.cuda()


    def mask_correlated_samples(self, world_size, device_size):
        batch_size = world_size * device_size

        neg_mask_i = torch.ones((batch_size, batch_size), dtype=bool)
        for rank in range(world_size):
            for idx in range(device_size):
                neg_mask_i[device_size * rank + idx, device_size * rank + idx] = 0  # i
        pos_mask_i = neg_mask_i.clone()

        return ~pos_mask_i, neg_mask_i


    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param, param_k in zip(self.encoder.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param.data * (1. - self.m)

        for param, param_k in zip(self.head.parameters(), self.head_k.parameters()):
            param_k.data = param_k.data * self.m + param.data * (1. - self.m)


    def forward(self, im_q, im_k):
        """
        Input:
            im_q: a batch of query images
            im_k: a batch of key images
        Output:
            loss
        """

        # compute query features
        q = self.head(self.encoder(im_q))  # queries: NxC
        q = nn.functional.normalize(q, dim=1)

        # compute key features
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()  # update the key encoder

            k = self.head_k(self.encoder_k(im_k))  # keys: NxC
            k = nn.functional.normalize(k, dim=1)

        neg_perm = torch.randperm(self.bsz - 1)

        zi_large, zj_large = q, k

        sim_i_large = self.similarity_f(
            zi_large.unsqueeze(1), zj_large.unsqueeze(0)) / self.T

        positive_samples_i = sim_i_large[self.pos_mask_i].reshape(self.bsz, 1)
        negative_samples_i = sim_i_large[self.neg_mask_i].reshape(self.bsz, -1)[:, neg_perm]

        labels_i = torch.zeros(self.bsz).cuda().long()
        logits_i = torch.cat((positive_samples_i, negative_samples_i), dim=1)

        loss_i = torch.log(
            torch.exp(positive_samples_i) +
            # self.alpha / negative_samples_i.shape[1] *  # uncomment this when negatives != bs
            torch.exp(negative_samples_i).sum(dim=-1, keepdim=True)
        ) - positive_samples_i

        loss = loss_i.mean()

        return loss


class SogCLR_DRO_M_Loss(nn.Module):
    def __init__(self, N=15000000, gamma=0.8, tau_init=0.07, tau_min=0.05, tau_max=1.0, rho=6.0,
                    eta_init=0.03, eta_min=1e-4, beta_u=0.9, eta_sched='const', eta_exp_gamma=0.8):
        super(SogCLR_DRO_M_Loss, self).__init__()
        self.gamma = gamma
        self.tau_init = tau_init
        self.tau_min = tau_min
        self.tau_max = tau_max
        self.rho = rho
        self.eta_init = eta_init
        self.eta_min = eta_min
        self.beta_u = beta_u
        self.eta_sched = eta_sched
        self.eta_exp_gamma = eta_exp_gamma
        self.s = torch.zeros(N).cuda()
        self.tau = torch.ones(N).cuda() * self.tau_init
        self.u = torch.zeros(N).cuda()
        self.b = torch.zeros(N).cuda()
        self.eps = 0.0
        self.grad_clip = 3.0

    def forward(self, index, logits):
        #Compute SogCLR_DRO loss for MoCo

        #Args:
        #    index: index of each sample [bsz].  e.g. [512]
        #    logits: [bsz, 1+K]
        #Returns:
        #    A loss scalar.
        pos_logits = logits[:, 0][:, None] # [bsz, 1]

        tau = self.tau[index][:, None] # [bsz, 1]

        logits_diff = logits - pos_logits # [bsz, 1+k]

        logits_diff_temp = (logits_diff / tau).clone().detach_() # [bsz, 1+K]
        exp_logits_diff_temp = torch.exp(logits_diff_temp) 

        g = torch.mean(exp_logits_diff_temp, dim=1, keepdim=True) # [bsz, 1]

        self.s[index] = (1.0-self.gamma) * self.s[index] + self.gamma * g.squeeze()

        weights = exp_logits_diff_temp / self.s[index][:, None]

        loss = torch.mean(weights * logits_diff, dim=1, keepdim=True)

        # gradient of tau
        grad_tau = torch.log(self.s[index]) + self.rho - torch.mean(weights * logits_diff_temp, dim=1, keepdim=False)
        grad_tau = grad_tau.clamp_(min=-self.grad_clip, max=self.grad_clip)
        
        self.u[index] = (1.0-self.beta_u) * self.u[index] + self.beta_u * grad_tau

        if self.eta_sched == 'cosine':
            eta_cur = self.eta_min + (self.eta_init - self.eta_min) * math.cos(math.pi * (epoch / max_epoch) / 2)
        elif self.eta_sched == 'exp':
            eta_cur = (self.eta_init - self.eta_min) * self.eta_exp_gamma ** (epoch-1) + self.eta_min
        elif self.eta_sched == 'const':
            eta_cur = self.eta_init
        else:
            assert 0, self.eta_sched + " is not supported."

        self.tau[index] = (self.tau[index] - eta_cur * self.u[index]).clamp_(min=self.tau_min, max=self.tau_max)
        
        avg_tau = tau.mean()
        
        return loss.mean(), avg_tau, eta_cur, grad_tau.mean().item(), 0.0 #old_b.mean().item()

