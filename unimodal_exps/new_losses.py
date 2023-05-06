"""
    iSogCLR_New_v1: using an additional neural network to compute the temperature 
                    for each input

    iSogCLR_New_v2: applying DRO techniques on all positive pairs based on iSogCLR
"""

from __future__ import print_function

import math

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


# very similar to the projection head
class TempGenerator(torch.nn.Module):
    def __init__(self, feature_dim, tau_min=0.05, tau_max=1.0):
        super(TempGenerator, self).__init__()

        self.feature_dim = feature_dim
        self.tau_min, self.tau_max = tau_min, tau_max
        self.temp_generator = nn.Sequential(
                                    nn.Linear(self.feature_dim, 8),
                                    nn.Sigmoid(),
                                    nn.Linear(8, 1)
                                )

    def forward(self, x_1, x_2):
        x_1 = F.normalize(x_1, dim=1)
        x_2 = F.normalize(x_2, dim=1)
        tau_1 = torch.sigmoid(self.temp_generator(x_1))   # [bsz, dim] -> [bsz, 1]
        tau_2 = torch.sigmoid(self.temp_generator(x_2))
        tau = (tau_1 + tau_2) / 2.0
        return torch.clamp(tau, min=self.tau_min, max=self.tau_max).squeeze()



# mimic the closed-form solution of tau
class TempGenerator_v2(torch.nn.Module):
    def __init__(self, feature_dim, M=16, tau_min=0.05, tau_max=1.0, attn_type='bilinear', hidden_dim=128):
        super(TempGenerator_v2, self).__init__()

        assert attn_type in ['dot_product', 'scaled_dot_product', 'additive', 'bilinear']
        self.attn_type = attn_type
        self.hidden_dim = hidden_dim  # only for additive attention

        self.feature_dim = feature_dim
        self.M = M                                       # the number of prototypes
        self.tau_min, self.tau_max = tau_min, tau_max
        
        self.mapping = nn.Linear(self.feature_dim, self.hidden_dim)
        self.prototypes = nn.Linear(self.hidden_dim, self.M, bias=False)

        #self.linear = nn.Linear(1, 1, bias=False)
    
        self.linear = nn.Linear(self.M, 1)

        self.linear_ab = nn.Linear(1, 1)

        self.pos_scaler = nn.Parameter(torch.rand([]).uniform_(-0.1,0.1))

        # weights for attention
        if self.attn_type == 'additive':
            self.attn_W = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
            self.attn_U = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
            self.attn_v = nn.Linear(self.hidden_dim, 1)

        elif self.attn_type == 'bilinear':
            self.attn_W = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
            self.attn_scaler = nn.Parameter(torch.rand([]).uniform_(-0.1,0.1))

        elif self.attn_type == 'scaled_dot_product':
            self.attn_scaler = nn.Parameter(torch.rand([]).uniform_(-0.1,0.1))


    def _att_func(self, query, key):
        # assume: query [bsz, dim], key [M, dim], return [bsz, M]
        bsz = query.shape[0]

        if self.attn_type == 'dot_product':
            att_scores = torch.einsum('n d, m d -> n m', query, key)

        elif self.attn_type == 'scaled_dot_product':
            att_scores = torch.einsum('n d, m d -> n m', query, key) / self.attn_scaler #np.sqrt(self.feature_dim)

        elif self.attn_type == 'additive':
            query_proj = self.attn_W(query).unsqueeze(1)        # [bsz, 1, hdim]
            query_proj = torch.tile(query_proj, (1, self.M, 1)) # [bsz, M, hdim]
            key_proj = self.attn_U(key).unsqueeze(1)            # [M, 1, hdim]
            key_proj = torch.tile(key_proj, (1, bsz, 1))        # [M, bsz, hdim]
            att_scores = self.attn_v(torch.tanh(query_proj + torch.permute(key_proj, (1,0,2)))).squeeze()

        elif self.attn_type == 'bilinear':
            att_scores = torch.einsum('n d, m d -> n m', query, self.attn_W(key)) / self.attn_scaler

        return nn.Softmax(dim=1)(att_scores)


    def forward(self, x_1, x_2):
        # assume: x_1 and x_2 [bsz, dim]
        x_1 = F.normalize(x_1, dim=1)                           # [bsz, dim]
        x_2 = F.normalize(x_2, dim=1)                           # [bsz, dim]

        x_1 = torch.sigmoid(self.mapping(x_1))  # [bsz, hdim]
        x_2 = torch.sigmoid(self.mapping(x_2))  # [bsz, hdim]
        #prototypes = F.normalize(self.prototype_mat, dim=1)     # [M, dim]

        #diag_sim = torch.diagonal(torch.einsum('m d, n d -> m n', x_1, x_2))[:, None] # [bsz, 1]

        """
        cos_sims_1 = torch.einsum('n d, m d -> n m', x_1, prototypes) # [bsz, M]
        att_weights_1 = self._att_func(query=x_1, key=prototypes)     # [bsz, M]     
        tau_1 = torch.einsum('b m, b m -> b', att_weights_1, cos_sims_1)[:, None]
        #tau_1 = torch.sigmoid(self.linear(tau_1) + self.pos_scaler * diag_sim)
        tau_1 = torch.sigmoid(self.linear(tau_1))

        cos_sims_2 = torch.einsum('n d, m d -> n m', x_2, prototypes) # [bsz, M]
        att_weights_2 = self._att_func(query=x_2, key=prototypes)     # [bsz, M]     
        tau_2 = torch.einsum('b m, b m -> b', att_weights_2, cos_sims_2)[:, None]
        #tau_2 = torch.sigmoid(self.linear(tau_2) + self.pos_scaler * diag_sim)
        tau_2 = torch.sigmoid(self.linear(tau_2))
        """

        sims_1 = torch.sigmoid(self.prototypes(x_1))  # [bsz, M]
        sims_2 = torch.sigmoid(self.prototypes(x_2))  # [bsz, M]

        #tau_1 = torch.sigmoid(self.linear(sims_1) + diag_sim * self.pos_scaler)
        #tau_2 = torch.sigmoid(self.linear(sims_2) + diag_sim * self.pos_scaler)

        #tau_1 = torch.sigmoid(self.linear(sims_1))
        #tau_2 = torch.sigmoid(self.linear(sims_2))

        #att_weights_1 = self._att_func(query=x_1, key=self.prototypes.weight)           # [bsz, M]     
        att_weights_1 = self._att_func(query=x_1, key=self.prototypes.weight)            # [bsz, M]     
        tau_1 = torch.einsum('b m, b m -> b', att_weights_1, sims_1)[:, None]            # [bsz, 1]   
        tau_1 = torch.sigmoid(self.linear_ab(tau_1))

        #att_weights_2 = self._att_func(query=x_2, key=self.prototypes.weight)           # [bsz, M]  
        att_weights_2 = self._att_func(query=x_2, key=self.prototypes.weight)            # [bsz, M]    
        tau_2 = torch.einsum('b m, b m -> b', att_weights_2, sims_2)[:, None]            # [bsz, 1]   
        tau_2 = torch.sigmoid(self.linear_ab(tau_2))

        tau = (tau_1 + tau_2) / 2.0

        return torch.clamp(tau, min=self.tau_min, max=self.tau_max).squeeze(), (att_weights_1, att_weights_2)





class iSogCLR_New_v1_Loss(nn.Module):
    def __init__(self, N=15000000, gamma=0.8, tau_init=0.07, tau_min=0.05, tau_max=1.0, 
                 rho=0.3, bsz=256, eta_init=0.001, beta_u=0.9, c=1.0, eps_attn_weight=1e-3):

        super(iSogCLR_New_v1_Loss, self).__init__()
        self.gamma = gamma
        self.tau_init = tau_init
        self.tau_min = tau_min
        self.tau_max = tau_max
        self.rho = rho
        self.eta_init = eta_init
        self.beta_u = beta_u

        self.s = torch.zeros(N).cuda()
        self.eps = 0.0
        self.grad_clip = 3.0

        #self.mask_neg = (1.0 - torch.eye(bsz * 2)).cuda()
        #self.num_neg = 2 * bsz - 1

        self.mask_neg = (1.0 - torch.eye(bsz)).repeat(2,2).cuda()
        self.num_neg = 2 * bsz - 2

        self.c = c # the coefficient for the tau_aug_loss, dicarded

        self.eps_attn_weight = eps_attn_weight


    def forward(self, index, features, taus, attn_weights):
        #using an additional neural network to compute the temperature for each input

        #Args:
        #    index: index of each sample [bsz].  e.g. [512]
        #    features: hidden vector of shape [bsz, n_views, dim].  e.g. [512, 2, 128]
        #    taus: temperatures generated by the TempGenerator [bsz, ]. e.g. [512]
        #Returns:
        #    A loss scalar.

        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0) # shape: [bsz * 2, dim]
        bsz = contrast_feature.shape[0] // 2

        sim = torch.einsum('i d, j d -> i j', contrast_feature, contrast_feature) # [bsz * 2, bsz * 2]
        pos_sim = torch.cat([torch.diagonal(sim, offset=bsz), torch.diagonal(sim, offset=-bsz)])[:, None] # [bsz * 2, 1]

        h_i = sim - pos_sim

        tau = taus.repeat(2) # [bsz * 2, 1]

        h_d_tau = (h_i / tau).clone().detach_()

        exp_h_d_tau = torch.exp(h_d_tau) * self.mask_neg

        g = torch.sum(exp_h_d_tau, dim=1, keepdim=True) / self.num_neg

        s1 = (1.0-self.gamma) * self.s[index] + self.gamma * g.squeeze()[:bsz]
        s2 = (1.0-self.gamma) * self.s[index] + self.gamma * g.squeeze()[bsz:]

        self.s[index] = (s1 + s2) / 2.0

        # feature loss
        feat_weights_1 = exp_h_d_tau[:bsz, :] / (s1[:, None] + self.eps)
        feat_loss_1 = torch.sum(feat_weights_1 * h_i[:bsz, :], dim=1, keepdim=True) / self.num_neg 

        feat_weights_2 = exp_h_d_tau[bsz:, :] / (s2[:, None] + self.eps)
        feat_loss_2 = torch.sum(feat_weights_2 * h_i[bsz:, :], dim=1, keepdim=True) / self.num_neg

        feat_loss = (feat_loss_1 + feat_loss_2).mean()

        # temp loss
        temp_weight_1 = torch.log(s1) + self.rho - torch.sum(feat_weights_1 * h_d_tau[:bsz, :], dim=1, keepdim=False) / self.num_neg
        temp_loss_1 = torch.mean(temp_weight_1 * tau[:bsz])

        temp_weight_2 = torch.log(s2) + self.rho - torch.sum(feat_weights_2 * h_d_tau[bsz:, :], dim=1, keepdim=False) / self.num_neg
        temp_loss_2 = torch.mean(temp_weight_2 * tau[bsz:])

        temp_loss = temp_loss_1 + temp_loss_2

        loss = feat_loss + temp_loss

        # attention regularization loss
        att_weights_1, att_weights_2 = attn_weights
        loss += self.eps_attn_weight * (torch.mean(torch.sum(att_weights_1 * torch.log(att_weights_1), dim=1)) + torch.mean(torch.sum(att_weights_2 * torch.log(att_weights_2), dim=1)))

        return loss, tau.mean(), 0.0, 0.0, 0.0



"""
class iSogCLR_New_v2_Loss(nn.Module):
    def __init__(self, N=15000000, gamma=0.8, tau_init=0.07, tau_min=0.05, tau_max=1.0, 
                 rho=0.3, bsz=256, eta_init=0.001, beta_u=0.9):

        super(iSogCLR_New_v2_Loss, self).__init__()
        self.gamma = gamma
        self.tau_init = tau_init
        self.tau_min = tau_min
        self.tau_max = tau_max
        self.rho = rho
        self.eta_init = eta_init
        self.beta_u = beta_u

        self.s = torch.zeros(N).cuda()
        self.tau = torch.ones(N).cuda() * self.tau_init
        self.u = torch.zeros(N).cuda()
        self.eps = 0.0
        self.grad_clip = 3.0

        self.mask_neg = (1.0 - torch.eye(bsz * 2)).cuda()
        self.num_neg = 2 * bsz - 1


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

        s1 = (1.0-self.gamma) * self.s[index] + self.gamma * g.squeeze()[:bsz]
        s2 = (1.0-self.gamma) * self.s[index] + self.gamma * g.squeeze()[bsz:]

        self.s[index] = (s1 + s2) / 2.0

        weights_1 = exp_sim_d_temps[:bsz, :] / (s1[:, None] + self.eps)
        loss_1 = torch.sum(weights_1 * sim[:bsz, :], dim=1, keepdim=True) / self.num_neg - pos_sim[:bsz, :]

        weights_2 = exp_sim_d_temps[bsz:, :] / (s2[:, None] + self.eps)
        loss_2 = torch.sum(weights_2 * sim[bsz:, :], dim=1, keepdim=True) / self.num_neg - pos_sim[bsz:, :]

        loss = loss_1 + loss_2

        # gradient of tau
        grad_tau_1 = torch.log(s1) + self.rho - torch.sum(weights_1 * sim_d_temps[:bsz, :], dim=1, keepdim=False) / self.num_neg
        grad_tau_2 = torch.log(s2) + self.rho - torch.sum(weights_2 * sim_d_temps[bsz:, :], dim=1, keepdim=False) / self.num_neg

        grad_tau = ((grad_tau_1 + grad_tau_2) / 2.0).clamp_(min=-self.grad_clip, max=self.grad_clip)
        
        self.u[index] = (1.0-self.beta_u) * self.u[index] + self.beta_u * grad_tau

        self.tau[index] = (self.tau[index] - self.eta_init * self.u[index]).clamp_(min=self.tau_min, max=self.tau_max)
        
        return loss.mean(), tau.mean(), self.eta_init, grad_tau.mean().item(), 0.0
"""
