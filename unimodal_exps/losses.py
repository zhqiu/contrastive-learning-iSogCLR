"""
Author: Yonglong Tian (yonglong@mit.edu)
Date: May 07, 2020
"""
from __future__ import print_function

import math

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, batch_size, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.batch_size = batch_size
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

        self.contrast_count = 2
        if self.contrast_mode == 'one':
            self.anchor_count = 1
        elif self.contrast_mode == 'all':
            self.anchor_count = self.contrast_count

        self.mask = torch.eye(self.batch_size, dtype=torch.float32).repeat(self.anchor_count, self.contrast_count).cuda()
        self.logits_mask = torch.scatter(                                      
                            torch.ones_like(self.mask), 1,
                            torch.arange(self.batch_size * self.anchor_count).view(-1, 1).cuda(), 0)
        self.mask = self.mask * self.logits_mask

    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf

        Args:
            features: hidden vector of shape [bsz, n_views, dim].  e.g. [512, 2, 128]  
            labels: ground truth of shape [bsz].  e.g. [512]
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')

        elif labels is None and mask is None:
            #mask = torch.eye(batch_size, dtype=torch.float32).to(device) 
            pass

        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)          
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]                                 
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0) # shape: [batch_size * 2, dim]
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(                                   # shape: [batch_size * 2, batch_size * 2]
            torch.matmul(anchor_feature, contrast_feature.T),              # inner product matrix between any two samples
            self.temperature)

        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        #mask = mask.repeat(anchor_count, contrast_count)                   
        # mask-out self-contrast cases
        #logits_mask = torch.scatter(                                       
        #    torch.ones_like(mask),
        #    1,
        #    torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
        #    0
        #)
        #mask = mask * logits_mask                                         

        # compute log_prob
        exp_logits = torch.exp(logits) * self.logits_mask                     
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))    

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (self.mask * log_prob).sum(1) / self.mask.sum(1)       

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss


class BatchIndependentLoss(nn.Module):
    def __init__(self, gamma, temperature=0.07, contrast_mode='all', base_temperature=0.07):
        super(BatchIndependentLoss, self).__init__()
        self.gamma = gamma
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature             
        self.u = torch.zeros(15000000).view(-1, 1).cuda()

    def forward(self, index, features, labels=None, mask=None):
        """Compute Batch-Independent SGD-NTX loss

        Args:
            index: index of each sample [bsz].  e.g. [512]
            features: hidden vector of shape [bsz, n_views, dim].  e.g. [512, 2, 128]  
            labels: ground truth of shape [bsz].  e.g. [512]
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')

        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device) 

        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)          # shape: [batch_size, batch_size]

        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]                                 
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0) # shape: [batch_size * 2, dim]
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(                                   # shape: [batch_size * 2, batch_size * 2]
            torch.matmul(anchor_feature, contrast_feature.T),              # inner product matrix between any two samples
            self.temperature)

        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)                   
        # mask-out self-contrast cases
        logits_mask = torch.scatter(                                       
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask                                          

        # 正负样本对 的 mask
        pos_mask = mask.float()
        neg_mask = 1.0 - pos_mask

        # 计算 p
        bsz = index.size(0)
        neg_exp_logits = (torch.exp(logits) * neg_mask).detach_()
        self.u[index] = (1 - self.gamma) * self.u[index] + self.gamma * torch.sum(neg_exp_logits, dim=1, keepdim=True)[:bsz]
        p_neg_pairs = neg_exp_logits / self.u[index].repeat(anchor_count, 1)

        # new way to compute log_prob
        log_prob = logits - (p_neg_pairs * logits * neg_mask).sum(1, keepdim=True)

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss


"""
    follow: https://github.com/Optimization-AI/SogCLR
"""
class SogCLR_Loss(nn.Module):
    def __init__(self, N=15000000, gamma=0.8, temperature=0.1):
        super(SogCLR_Loss, self).__init__()
        self.gamma = gamma
        self.u = torch.zeros(N).cuda()
        self.LARGE_NUM = 1e9
        self.T = temperature

    def forward(self, index, hidden1, hidden2):
        batch_size = hidden1.shape[0]

        labels = F.one_hot(torch.arange(batch_size, dtype=torch.long), batch_size * 2).cuda() # [B, 2*B]
        masks  = F.one_hot(torch.arange(batch_size, dtype=torch.long), batch_size).cuda()     # [B, B]

        logits_aa = torch.matmul(hidden1, hidden1.T)
        logits_aa = logits_aa - masks * self.LARGE_NUM
        logits_bb = torch.matmul(hidden2, hidden2.T)
        logits_bb = logits_bb - masks * self.LARGE_NUM
        logits_ab = torch.matmul(hidden1, hidden2.T)
        logits_ba = torch.matmul(hidden2, hidden1.T)

        #  SogCLR
        neg_mask = 1-labels
        logits_ab_aa = torch.cat([logits_ab, logits_aa], 1) 
        logits_ba_bb = torch.cat([logits_ba, logits_bb], 1) 
      
        neg_logits1 = torch.exp(logits_ab_aa / self.T) * neg_mask   #(B, 2B)
        neg_logits2 = torch.exp(logits_ba_bb / self.T) * neg_mask

        u1 = (1 - self.gamma) * self.u[index] + self.gamma * torch.sum(neg_logits1, dim=1, keepdim=False) / (2*(batch_size-1))
        u2 = (1 - self.gamma) * self.u[index] + self.gamma * torch.sum(neg_logits2, dim=1, keepdim=False) / (2*(batch_size-1))
        
        self.u[index] = u1.detach()+ u2.detach()

        p_neg_weights1 = (neg_logits1 / u1[:, None]).detach()
        p_neg_weights2 = (neg_logits2 / u2[:, None]).detach()

        def softmax_cross_entropy_with_logits(labels, logits, weights):
            expsum_neg_logits = torch.sum(weights*logits, dim=1, keepdim=True)/(2*(batch_size-1))
            normalized_logits = logits - expsum_neg_logits
            return -torch.sum(labels * normalized_logits, dim=1)

        loss_a = softmax_cross_entropy_with_logits(labels, logits_ab_aa, p_neg_weights1)
        loss_b = softmax_cross_entropy_with_logits(labels, logits_ba_bb, p_neg_weights2)
        loss = (loss_a + loss_b).mean()

        return loss


class SogCLR_DRO_Loss(nn.Module):
    def __init__(self, N=15000000, gamma=0.8, tau_init=0.07, tau_min=0.05, tau_max=1.0, rho=6.0, bsz=256,
                    eta_init=0.001, eta_min=1e-4, beta_u=0.9, eta_sched='const', eta_exp_gamma=0.8):
        super(SogCLR_DRO_Loss, self).__init__()
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
        self.eps = 1e-8
        self.grad_clip = 3.0

        self.mask_neg = (1.0 - torch.eye(bsz)).repeat(2,2).cuda()
        self.num_neg = 2 * bsz - 2

        #self.mask_neg = (1.0 - torch.eye(bsz * 2)).cuda()
        #self.num_neg = 2 * bsz - 1


    def forward(self, index, features, epoch, max_epoch):
        #Compute SogCLR_DRO loss

        #Args:
        #    index: index of each sample [bsz].  e.g. [512]
        #    features: hidden vector of shape [bsz, n_views, dim].  e.g. [512, 2, 128]
        #Returns:
        #    A loss scalar.
        
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0) # shape: [bsz * 2, dim]
        bsz = contrast_feature.shape[0] // 2

        sim = torch.einsum('i d, j d -> i j', contrast_feature, contrast_feature) # [bsz * 2, bsz * 2]

        pos_sim = torch.cat([torch.diagonal(sim, offset=bsz), torch.diagonal(sim, offset=-bsz)])[:, None] # [bsz * 2, 1]

        tau = self.tau[index].repeat(2)

        sim_d_temps = (sim / tau[:, None]).clone().detach_()
 
        exp_sim_d_temps = torch.exp(sim_d_temps) * self.mask_neg

        g = torch.sum(exp_sim_d_temps, dim=1, keepdim=True) / self.num_neg

        if epoch == 0:
            s1 = g.squeeze()[:bsz]
            s2 = g.squeeze()[bsz:]
        else:
            s1 = (1.0-self.gamma) * self.s[index] + self.gamma * g.squeeze()[:bsz]
            s2 = (1.0-self.gamma) * self.s[index] + self.gamma * g.squeeze()[bsz:]

        self.s[index] = (s1 + s2) / 2.0

        weights_1 = exp_sim_d_temps[:bsz, :] / s1[:, None]
        loss_1 = torch.sum(weights_1 * sim[:bsz, :], dim=1, keepdim=True) / self.num_neg - pos_sim[:bsz, :]

        weights_2 = exp_sim_d_temps[bsz:, :] / s2[:, None]
        loss_2 = torch.sum(weights_2 * sim[bsz:, :], dim=1, keepdim=True) / self.num_neg - pos_sim[bsz:, :]

        loss = loss_1 + loss_2

        # gradient of tau
        grad_tau_1 = torch.log(s1) + self.rho - torch.sum(weights_1 * sim_d_temps[:bsz, :], dim=1, keepdim=False) / self.num_neg
        grad_tau_2 = torch.log(s2) + self.rho - torch.sum(weights_2 * sim_d_temps[bsz:, :], dim=1, keepdim=False) / self.num_neg 

        grad_tau = ((grad_tau_1 + grad_tau_2) / 2.0).clamp_(min=-self.grad_clip, max=self.grad_clip)
        
        self.u[index] = (1.0-self.beta_u) * self.u[index] + self.beta_u * grad_tau

        self.tau[index] = (self.tau[index] - self.eta_init * self.u[index]).clamp_(min=self.tau_min, max=self.tau_max)
        
        avg_tau = tau.mean()
        
        return loss.mean(), avg_tau, self.eta_init, grad_tau.mean().item(), 0.0 #old_b.mean().item()



# using surrogate function (sqh)
class SogCLR_DRO_Loss_v2(nn.Module):
    def __init__(self, N=15000000, gamma=0.8, tau_init=0.07, tau_min=0.05, tau_max=1.0, rho=6.0, bsz=256,
                    eta_init=0.001, eta_min=1e-4, beta_u=0.9, eta_sched='const', eta_exp_gamma=0.8):
        super(SogCLR_DRO_Loss_v2, self).__init__()
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

        #self.mask_neg = (1.0 - torch.eye(bsz)).repeat(2,2).cuda()
        #self.num_neg = 2 * bsz - 2

        self.mask_neg = (1.0 - torch.eye(bsz * 2)).cuda()
        self.num_neg = 2 * bsz - 1

    def _sqh(self, x, c=0.8): # squared hinge surrogate function
        return torch.max(torch.zeros_like(x), x + c) ** 2


    def forward(self, index, features, epoch, max_epoch):
        #Compute SogCLR_DRO loss

        #Args:
        #    index: index of each sample [bsz].  e.g. [512]
        #    features: hidden vector of shape [bsz, n_views, dim].  e.g. [512, 2, 128]
        #Returns:
        #    A loss scalar.
        
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0) # shape: [bsz * 2, dim]
        bsz = contrast_feature.shape[0] // 2

        sim = torch.einsum('i d, j d -> i j', contrast_feature, contrast_feature) # [bsz * 2, bsz * 2]

        pos_sim = torch.cat([torch.diagonal(sim, offset=bsz), torch.diagonal(sim, offset=-bsz)])[:, None] # [bsz * 2, 1]

        h_i = sim - pos_sim
        l_i = self._sqh(h_i)

        tau = self.tau[index].repeat(2)

        l_d_tau = (l_i / tau).clone().detach_()

        exp_l_d_tau = torch.exp(l_d_tau) * self.mask_neg

        g = torch.sum(exp_l_d_tau, dim=1, keepdim=True) / self.num_neg

        s1 = (1.0-self.gamma) * self.s[index] + self.gamma * g.squeeze()[:bsz]
        s2 = (1.0-self.gamma) * self.s[index] + self.gamma * g.squeeze()[bsz:]

        self.s[index] = (s1 + s2) / 2.0

        weights_1 = exp_l_d_tau[:bsz, :] / (s1[:, None] + self.eps)
        loss_1 = torch.sum(weights_1 * l_i[:bsz, :], dim=1, keepdim=True) / self.num_neg 

        weights_2 = exp_l_d_tau[bsz:, :] / (s2[:, None] + self.eps)
        loss_2 = torch.sum(weights_2 * l_i[bsz:, :], dim=1, keepdim=True) / self.num_neg

        loss = loss_1 + loss_2

        # gradient of tau
        grad_tau_1 = torch.log(s1) + self.b[index] + self.rho - torch.sum(weights_1 * l_d_tau[:bsz, :], dim=1, keepdim=False) / self.num_neg
        grad_tau_2 = torch.log(s2) + self.b[index] + self.rho - torch.sum(weights_2 * l_d_tau[bsz:, :], dim=1, keepdim=False) / self.num_neg

        grad_tau = ((grad_tau_1 + grad_tau_2) / 2.0).clamp_(min=-self.grad_clip, max=self.grad_clip)
        
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




"""
    Spectral Contrastive Learning
    https://github.com/jhaochenz/spectral_contrastive_learning
"""

def spectral_cl_loss(z1, z2, mu=1.0):
    mask1 = (torch.norm(z1, p=2, dim=1) < np.sqrt(mu)).float().unsqueeze(1)
    mask2 = (torch.norm(z2, p=2, dim=1) < np.sqrt(mu)).float().unsqueeze(1)
    z1 = mask1 * z1 + (1-mask1) * F.normalize(z1, dim=1) * np.sqrt(mu)
    z2 = mask2 * z2 + (1-mask2) * F.normalize(z2, dim=1) * np.sqrt(mu)
    loss_part1 = -2 * torch.mean(z1 * z2) * z1.shape[1]
    square_term = torch.matmul(z1, z2.T) ** 2
    loss_part2 = torch.mean(torch.triu(square_term, diagonal=1) + torch.tril(square_term, diagonal=-1)) * \
                 z1.shape[0] / (z1.shape[0] - 1)
    return (loss_part1 + loss_part2) / mu, {"part1": loss_part1 / mu, "part2": loss_part2 / mu}


"""
    FlatCLR
    https://github.com/Junya-Chen/FlatCLR
"""
def flatclr_loss(features, batch_size, temperature=0.1, n_views=2):
        labels = torch.cat([torch.arange(batch_size) for i in range(n_views)], dim=0)
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        labels = labels.cuda()
        features = F.normalize(features, dim=1)

        similarity_matrix = torch.matmul(features, features.T)

        # discard the main diagonal from both: labels and similarities matrix
        mask = torch.eye(labels.shape[0], dtype=torch.bool).cuda()
        labels = labels[~mask].view(labels.shape[0], -1)
        similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)

        # select and combine multiple positives
        positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)

        # select only the negatives the negatives
        negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

        logits = torch.cat([positives, negatives], dim=1)

        labels = torch.zeros(positives.shape[0], dtype=torch.long).cuda()

        logits = (negatives - positives) / temperature

        return logits, labels


"""
    Barlow Twins
    https://github.com/facebookresearch/barlowtwins/blob/main.py
"""
def off_diagonal(x):
    # return a flattened view of the off-diagonal elements of a square matrix
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


def Projector(net_type, add_one_entry=False):
    mlp = "8192-8192-8192"
    embedding = 512 if net_type == 'resnet18' else 2048

    mlp_spec = f"{embedding}-{mlp}"
    layers = []
    f = list(map(int, mlp_spec.split("-")))

    if add_one_entry:
        f[-1] += 1

    for i in range(len(f) - 2):
        layers.append(nn.Linear(f[i], f[i + 1]))
        layers.append(nn.BatchNorm1d(f[i + 1]))
        layers.append(nn.ReLU(True))
    layers.append(nn.Linear(f[-2], f[-1], bias=False))
    return nn.Sequential(*layers)


class BarlowTwins(nn.Module):
    def __init__(self, dim_size, batch_size, lambd=0.0051):
        super().__init__()

        self.lambd = lambd
        self.batch_size = batch_size
        self.bn = nn.BatchNorm1d(dim_size, affine=False)

    def forward(self, z1, z2):
        # empirical cross-correlation matrix
        c = self.bn(z1).T @ self.bn(z2)

        # sum the cross-correlation matrix between all gpus
        c.div_(self.batch_size)

        on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
        off_diag = off_diagonal(c).pow_(2).sum()
        loss = on_diag + self.lambd * off_diag

        return loss


"""
    VICReg
    https://github.com/facebookresearch/vicreg/blob/main/main_vicreg.py
"""
class VICReg(nn.Module):
    def __init__(self, dim_size, batch_size, sim_coeff=25.0, std_coeff=25.0, cov_coeff=1.0):
        super().__init__()

        self.batch_size = batch_size
        self.dim_size = dim_size
        self.sim_coeff = sim_coeff
        self.std_coeff = std_coeff
        self.cov_coeff = cov_coeff


    def forward(self, x, y):
        repr_loss = F.mse_loss(x, y) # invariance term

        x = x - x.mean(dim=0)
        y = y - y.mean(dim=0)

        std_x = torch.sqrt(x.var(dim=0) + 0.0001)
        std_y = torch.sqrt(y.var(dim=0) + 0.0001)
        std_loss = torch.mean(F.relu(1 - std_x)) / 2 + torch.mean(F.relu(1 - std_y)) / 2  # variance term

        cov_x = (x.T @ x) / (self.batch_size - 1)
        cov_y = (y.T @ y) / (self.batch_size - 1)
        cov_loss = off_diagonal(cov_x).pow_(2).sum().div(
            self.dim_size
        ) + off_diagonal(cov_y).pow_(2).sum().div(self.dim_size)  # covariance term

        loss = (
            self.sim_coeff * repr_loss
            + self.std_coeff * std_loss
            + self.cov_coeff * cov_loss
        )

        return loss


"""
    SimCo (Dual Temperature Loss)
    https://arxiv.org/pdf/2203.17248.pdf
"""
def simco_loss(query, key, intra_temperature=0.1, inter_temperature=1.0):
    """
        N: batch size
        D: the dimension of representation vector
        Args:
        query (torch.Tensor): NxD Tensor containing
        projected features from view 1.
        key (torch.Tensor): NxD Tensor containing
        projected features from view 2.
        intra_temperature (float): temperature factor
        for the intra component.
        inter_temperature (float): temperature factor
        for the inter component.
        Returns:
        torch.Tensor: SimCo loss.
    """

    # normalize query and key
    query = F.normalize(query, dim=-1)
    key = F.normalize(key, dim=-1)

    # calculate logits
    logits = query @ key.T

    # intra awareness
    logits_intra = logits / intra_temperature
    prob_intra = F.softmax(logits_intra, dim=1)

    # inter awareness
    logits_inter = logits / inter_temperature
    prob_inter = F.softmax(logits_inter, dim=1)

    # inter awareness changing factor
    mask = torch.ones(prob_inter.size()).fill_diagonal_(0).cuda()

    weight_alpha = (prob_intra * mask).sum(-1)
    weight_beta = (prob_inter * mask).sum(-1)
    inter_intra = weight_beta / weight_alpha

    # loss calculation
    log_softmax = F.log_softmax(logits, dim=-1)
    log_softmax_diag = log_softmax.diag()
    loss = -inter_intra.detach() * log_softmax_diag

    return loss.mean()


"""
    TaU
    https://github.com/mhw32/temperature-as-uncertainty-public/blob/d6c6f05dc217b6169f31ba25385cb4bcdd28ab6a/src/objectives/simclr.py
"""
def tau_simclr_build_mask(mb, simclr=False): # Either building the SimCLR mask or the new mask
    if simclr:
        m = torch.eye(mb).cuda().bool()
    else:
        m = torch.eye(mb // 2).cuda().bool()
        m = torch.cat([m, m], dim=1)
        m = torch.cat([m, m], dim=0)
    return m


def tau_simclr(loc1, temp1, loc2, temp2, t=0.07, eps=1e-6, simclr_mask=False):
    loc1 = F.normalize(loc1)
    loc2 = F.normalize(loc2)
    temp1 = torch.sigmoid(temp1)
    temp2 = torch.sigmoid(temp2)

    # out: [2 * batch_size, dim]
    loc = torch.cat([loc1, loc2], dim=0)
    temp = torch.cat([temp1, temp2], dim=0)
    n_samples = loc.size(0)

    # cov and sim: [2 * batch_size, 2 * batch_size]
    # neg: [2 * batch_size]
    cov = torch.mm(loc, loc.t().contiguous())
    var = temp.repeat(1, n_samples)
    sim = torch.exp((cov * var) / t)

    # NOTE: this mask is a little different than SimCLR's mask
    mask = ~tau_simclr_build_mask(n_samples, simclr=simclr_mask)
    neg = sim.masked_select(mask).view(n_samples, -1).sum(dim=-1)

     # Positive similarity, pos becomes [2 * batch_size]
    pos = torch.sum(loc1 * loc2, dim=-1)
    pos = pos * (temp1 / t).squeeze(-1)
    pos = torch.exp(pos)
    pos = torch.cat([pos, pos], dim=0)

    loss = -torch.log(pos / (neg + eps)).mean()

    return loss


"""
    HCL: Contrastive Learning with Hard Negative Samples
    https://github.com/joshr17/HCL
"""
def get_negative_mask(batch_size):
    negative_mask = torch.ones((batch_size, 2 * batch_size), dtype=bool)
    for i in range(batch_size):
        negative_mask[i, i] = 0
        negative_mask[i, i + batch_size] = 0

    negative_mask = torch.cat((negative_mask, negative_mask), 0)
    return negative_mask


def hcl_loss(out_1,out_2,tau_plus,batch_size,beta=0.5,estimator='hard',temperature=0.5):
        # neg score
        out = torch.cat([out_1, out_2], dim=0)
        neg = torch.exp(torch.mm(out, out.t().contiguous()) / temperature)
        old_neg = neg.clone()
        mask = get_negative_mask(batch_size).cuda()
        neg = neg.masked_select(mask).view(2 * batch_size, -1)

        # pos score
        pos = torch.exp(torch.sum(out_1 * out_2, dim=-1) / temperature)
        pos = torch.cat([pos, pos], dim=0)
        
        # negative samples similarity scoring
        if estimator=='hard':
            N = batch_size * 2 - 2
            imp = (beta* neg.log()).exp()
            reweight_neg = (imp*neg).sum(dim = -1) / imp.mean(dim = -1)
            Ng = (-tau_plus * N * pos + reweight_neg) / (1 - tau_plus)
            # constrain (optional)
            Ng = torch.clamp(Ng, min = N * np.e**(-1 / temperature))
        elif estimator=='easy':
            Ng = neg.sum(dim=-1)
        else:
            raise Exception('Invalid estimator selected. Please use any of [hard, easy]')
            
        # contrastive loss
        loss = (- torch.log(pos / (pos + Ng) )).mean()

        return loss

