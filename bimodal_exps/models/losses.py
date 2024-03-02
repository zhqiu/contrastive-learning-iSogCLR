"""
    implementation of other two-way contrastive losses
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist


# https://github.com/Spijkervet/SimCLR/blob/master/simclr/modules/gather.py
class GatherLayer(torch.autograd.Function):
    """Gather tensors from all process, supporting backward propagation."""

    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        output = [torch.zeros_like(input) for _ in range(dist.get_world_size())]
        dist.all_gather(output, input)
        return tuple(output)

    @staticmethod
    def backward(ctx, *grads):
        (input,) = ctx.saved_tensors
        grad_out = torch.zeros_like(input)
        grad_out[:] = grads[dist.get_rank()]
        return grad_out


class CLIP_Loss(nn.Module):

    def __init__(self, world_size=8, temperature=0.01, personalized_tau=False, image_tau=None, text_tau=None):
        super(CLIP_Loss, self).__init__()
        self.world_size = world_size
        self.temperature = temperature
        self.personalized_tau = personalized_tau # if true, then temperatures are learnable
        self.image_tau = image_tau
        self.text_tau = text_tau

    def forward(self, image_features, text_features, image_idx=None, text_idx=None):
        if self.world_size > 1:
            image_features = torch.cat(GatherLayer.apply(image_features), dim=0)
            text_features = torch.cat(GatherLayer.apply(text_features), dim=0)

        if self.personalized_tau:
            image_temp = self.image_tau[image_idx]
            text_temp = self.text_tau[text_idx]
            sim = torch.einsum('i d, j d -> i j', text_features, image_features)
            labels = torch.arange(image_features.shape[0], device=image_features.device)
            total_loss = (F.cross_entropy(sim / text_temp, labels) + F.cross_entropy(sim.t() / image_temp, labels)) / 2

        else:
            sim = torch.einsum('i d, j d -> i j', text_features, image_features) / self.temperature
            labels = torch.arange(image_features.shape[0], device=image_features.device)
            total_loss = (F.cross_entropy(sim, labels) + F.cross_entropy(sim.t(), labels)) / 2

        return total_loss


class SogCLR_Loss(nn.Module):
    def __init__(self, N=8000000, gamma=0.1, temperature=0.07, world_size=8):
        """
        Inputs:
           N is number of samples in training set
        """
        super(SogCLR_Loss, self).__init__()
        self.world_size = world_size
        self.s_I = torch.zeros(N).cuda()
        self.s_T = torch.zeros(N).cuda()
        self.b_I = torch.zeros(N).cuda()
        self.b_T = torch.zeros(N).cuda()
        self.gamma = gamma
        self.temperature = temperature
        self.eps = 1e-10

    def forward(self, image_features, text_features, image_ids, text_ids):
        """
        Inputs:
            image_features, text_features is l2-normalized tensor
            image_features, text_features: [batch_size, emb_dim]
        """

        if self.world_size > 1:
            image_features = torch.cat(GatherLayer.apply(image_features), dim=0)
            text_features = torch.cat(GatherLayer.apply(text_features), dim=0)

        # compute the logits (similarity between each image-text pair)
        sim = torch.einsum('i d, j d -> i j', image_features, text_features)
        diag_sim = torch.diagonal(sim)

        batch_size = sim.shape[0]

        # E_I(x_i)*E_T(t) - E_I(x_i)*E_T(t_i)
        image_diffs = sim - diag_sim[:, None]
        # E_I(x)*E_T(t_i) - E_I(x_i)*E_T(t_i)
        text_diffs = sim - diag_sim[None, :]

        image_diffs_d_temps = (image_diffs / self.temperature).clone().detach_()
        text_diffs_d_temps = (text_diffs / self.temperature).clone().detach_()

        # update b
        old_b_I = self.b_I[image_ids]
        new_b_I = torch.max(image_diffs_d_temps, old_b_I[:, None].tile(1, batch_size))
        self.b_I[image_ids] = torch.max(new_b_I, dim=1)[0]

        old_b_T = self.b_T[text_ids]
        new_b_T = torch.max(text_diffs_d_temps, old_b_T[None, :].tile(batch_size, 1))
        self.b_T[text_ids] = torch.max(new_b_T, dim=0)[0]

        exp_image_diffs = torch.exp(image_diffs_d_temps - self.b_I[image_ids][:, None]) # -b to avoid exp operation overflow
        exp_text_diffs = torch.exp(text_diffs_d_temps - self.b_T[text_ids][None, :])

        g_I = torch.sum(exp_image_diffs, dim=1, keepdim=True) / (batch_size-1)
        g_T = torch.sum(exp_text_diffs, dim=0, keepdim=True) / (batch_size-1)

        self.s_I[image_ids] = (1.0-self.gamma) * self.s_I[image_ids] * torch.exp(old_b_I - self.b_I[image_ids]) + self.gamma * g_I.squeeze()
        self.s_T[text_ids] = (1.0-self.gamma) * self.s_T[text_ids] * torch.exp(old_b_T - self.b_T[text_ids]) + self.gamma * g_T.squeeze()
        
        s_I = self.s_I[image_ids].reshape(g_I.shape)
        s_T = self.s_T[text_ids].reshape(g_T.shape)

        weights_image = exp_image_diffs / (s_I + self.eps)
        weights_text = exp_text_diffs / (s_T + self.eps)

        if torch.any(torch.isnan(weights_image)):
            assert 0, "weights_image has nan."
        if torch.any(torch.isnan(weights_text)):
            assert 0, "weights_text has nan."

        image_loss = torch.sum(weights_image * image_diffs, dim=1, keepdim=True) / (batch_size-1)
        text_loss = torch.sum(weights_text * text_diffs, dim=0, keepdim=True) / (batch_size-1)

        total_loss = image_loss.mean() + text_loss.mean()

        return total_loss



class SogCLR_DRO_Loss(nn.Module):
    def __init__(self, N=2900000, gamma=0.8, tau_init=0.07, tau_min=5e-3, tau_max=1.0, rho_init=0.1, bsz=128,
                 eta_init=0.01, eta_min=1e-4, beta_u=0.9, eta_sched=None, eta_exp_gamma=0.8, world_size=8, eps=1e-10):
        
        #Inputs:
        #   N is number of samples in training set
        
        super(SogCLR_DRO_Loss, self).__init__()
        self.world_size = world_size
        self.gamma_I, self.gamma_T = gamma, gamma
        self.tau_min = tau_min
        self.tau_max = tau_max
        self.tau_init = tau_init
        self.s_I = torch.zeros(N).cuda()
        self.s_T = torch.zeros(N).cuda()
        self.tau_I = torch.ones(N).cuda() * self.tau_init
        self.tau_T = torch.ones(N).cuda() * self.tau_init
        self.u_I = torch.zeros(N).cuda()
        self.u_T = torch.zeros(N).cuda()
        self.b_I = torch.zeros(N).cuda()
        self.b_T = torch.zeros(N).cuda()
        self.rho_I = rho_init
        self.rho_T = rho_init
        self.eps = eps
        self.eta_sched = eta_sched
        self.eta_exp_gamma = eta_exp_gamma  # multiplicative factor of learning rate decay for exponential eta_sched
        self.eta_init = eta_init
        self.eta_min = eta_min
        self.beta_u = beta_u
        self.batch_size = bsz
        self.grad_clip = 5.0
        self.mask_neg = (1.0 - torch.eye(bsz)).cuda()

    def forward(self, image_features, text_features, image_ids, text_ids, epoch, max_epoch):
        if self.world_size > 1:
            image_features = torch.cat(GatherLayer.apply(image_features), dim=0)
            text_features = torch.cat(GatherLayer.apply(text_features), dim=0)

        # compute the logits (similarity between each image-text pair)
        sim = torch.einsum('i d, j d -> i j', image_features, text_features)
        diag_sim = torch.diagonal(sim)

        batch_size = sim.shape[0]

        # learnable temperature for each image and text
        tau_image = self.tau_I[image_ids]
        tau_text = self.tau_T[text_ids]

        # E_I(x_i)*E_T(t) - E_I(x_i)*E_T(t_i)
        image_diffs = sim - diag_sim[:, None]
        # E_I(x)*E_T(t_i) - E_I(x_i)*E_T(t_i)
        text_diffs = sim - diag_sim[None, :]

        image_diffs_d_temps = (image_diffs / tau_image[:, None]).clone().detach_()
        text_diffs_d_temps = (text_diffs / tau_text[None, :]).clone().detach_()

        # update b
        old_b_I = self.b_I[image_ids]
        new_b_I = torch.max(image_diffs_d_temps, old_b_I[:, None].tile(1, batch_size))
        self.b_I[image_ids] = torch.max(new_b_I, dim=1)[0]

        old_b_T = self.b_T[text_ids]
        new_b_T = torch.max(text_diffs_d_temps, old_b_T[None, :].tile(batch_size, 1))
        self.b_T[text_ids] = torch.max(new_b_T, dim=0)[0]

        exp_image_diffs = torch.exp(image_diffs_d_temps - self.b_I[image_ids][:, None]) # -b to avoid exp operation overflow
        exp_text_diffs = torch.exp(text_diffs_d_temps - self.b_T[text_ids][None, :])

        g_I = torch.sum(exp_image_diffs, dim=1, keepdim=True) / (batch_size-1)
        g_T = torch.sum(exp_text_diffs, dim=0, keepdim=True) / (batch_size-1)

        self.s_I[image_ids] = (1.0-self.gamma_I) * self.s_I[image_ids] * torch.exp(old_b_I - self.b_I[image_ids]) + self.gamma_I * g_I.squeeze()
        self.s_T[text_ids] = (1.0-self.gamma_T) * self.s_T[text_ids] * torch.exp(old_b_T - self.b_T[text_ids]) + self.gamma_T * g_T.squeeze()
        
        s_I = self.s_I[image_ids].reshape(g_I.shape)
        s_T = self.s_T[text_ids].reshape(g_T.shape)

        weights_image = exp_image_diffs / (s_I + self.eps)
        weights_text = exp_text_diffs / (s_T + self.eps)

        if torch.any(torch.isnan(weights_image)):
            assert 0, "weights_image has nan."
        if torch.any(torch.isnan(weights_text)):
            assert 0, "weights_text has nan."

        image_loss = torch.sum(weights_image * image_diffs, dim=1, keepdim=True) / (batch_size-1)
        text_loss = torch.sum(weights_text * text_diffs, dim=0, keepdim=True) / (batch_size-1)

        total_loss = image_loss.mean() + text_loss.mean()

        # gradient of tau for image and text
        grad_tau_image = torch.log(s_I) + self.b_I[image_ids][:, None] + self.rho_I - torch.sum(weights_image * image_diffs_d_temps, dim=1, keepdim=True) / (batch_size-1)
        grad_tau_text = torch.log(s_T) + self.b_T[text_ids][None, :] + self.rho_T - torch.sum(weights_text * text_diffs_d_temps, dim=0, keepdim=True) / (batch_size-1)
       
        self.u_I[image_ids] = (1.0-self.beta_u) * self.u_I[image_ids] + self.beta_u * grad_tau_image.squeeze().clamp_(min=-self.grad_clip, max=self.grad_clip)
        self.u_T[text_ids] = (1.0-self.beta_u) * self.u_T[text_ids] + self.beta_u * grad_tau_text.squeeze().clamp_(min=-self.grad_clip, max=self.grad_clip)

        if self.eta_sched == 'cosine':
            eta_cur = self.eta_min + (self.eta_init - self.eta_min) * math.cos(math.pi * (epoch / max_epoch) / 2)
        elif self.eta_sched == 'exp':
            eta_cur = (self.eta_init - self.eta_min) * self.eta_exp_gamma ** (epoch-1) + self.eta_min
        elif self.eta_sched == 'const':
            eta_cur = self.eta_init
        else:
            assert 0, self.eta_sched + " is not supported."

        self.tau_I[image_ids] = (tau_image - eta_cur * self.u_I[image_ids]).clamp_(min=self.tau_min, max=self.tau_max)
        self.tau_T[text_ids] = (tau_text - eta_cur * self.u_T[text_ids]).clamp_(min=self.tau_min, max=self.tau_max)

        avg_image_tau = tau_image.mean().item()
        avg_text_tau = tau_text.mean().item()

        return total_loss, avg_image_tau, avg_text_tau, eta_cur, grad_tau_image.mean().item(), grad_tau_text.mean().item(), old_b_I.mean().item(), old_b_T.mean().item()


"""
    https://github.com/goel-shashank/CyCLIP/blob/52d77af2a5f1a4bff01b4c371d6b98e2d0340137/src/train.py
"""
class CyCLIP_Loss(nn.Module):
    def __init__(self, world_size, temperature, cylambda_1=0.25 , cylambda_2=0.25):
        super(CyCLIP_Loss, self).__init__()

        self.world_size = world_size
        self.temperature = temperature
        self.criterion = nn.CrossEntropyLoss().cuda()
        self.cylambda_1 = cylambda_1
        self.cylambda_2 = cylambda_2


    def forward(self, image_features, text_features):
        if self.world_size > 1:
            image_features = torch.cat(GatherLayer.apply(image_features), dim=0)
            text_features = torch.cat(GatherLayer.apply(text_features), dim=0)

        batch_size = len(image_features)

        logits_text_per_image = (image_features @ text_features.t()) / self.temperature
        logits_image_per_text = logits_text_per_image.t()

        target = torch.arange(batch_size).long().cuda()

        # contrastive loss, the same as CLIP
        contrastive_loss = (self.criterion(logits_text_per_image, target) + self.criterion(logits_image_per_text, target)) / 2.0 

        # inmodal_cyclic_loss
        logits_image_per_image = (image_features @ image_features.t()) / self.temperature
        logits_text_per_text = (text_features @ text_features.t()) / self.temperature
        inmodal_cyclic_loss = (logits_image_per_image - logits_text_per_text).square().mean() * (self.temperature ** 2) * batch_size

        # crossmodal_cyclic_loss
        crossmodal_cyclic_loss = (logits_text_per_image - logits_image_per_text).square().mean() * (self.temperature ** 2) * batch_size

        loss = contrastive_loss + self.cylambda_1 * inmodal_cyclic_loss + self.cylambda_2 * crossmodal_cyclic_loss

        return loss


"""
    VICReg
    https://github.com/facebookresearch/vicreg/blob/main/main_vicreg.py
"""
def off_diagonal(x):
    # return a flattened view of the off-diagonal elements of a square matrix
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()
    

class VICReg_Loss(nn.Module):
    def __init__(self, world_size, dim_size, sim_coeff=25.0, std_coeff=25.0, cov_coeff=1.0):
        super(VICReg_Loss, self).__init__()

        self.world_size = world_size
        self.dim_size = dim_size
        self.sim_coeff = sim_coeff
        self.std_coeff = std_coeff
        self.cov_coeff = cov_coeff


    def forward(self, image_features, text_features):
        if self.world_size > 1:
            x = torch.cat(GatherLayer.apply(image_features), dim=0)
            y = torch.cat(GatherLayer.apply(text_features), dim=0)

        batch_size = len(x)

        repr_loss = F.mse_loss(x, y) # invariance term

        x = x - x.mean(dim=0)
        y = y - y.mean(dim=0)

        std_x = torch.sqrt(x.var(dim=0) + 0.0001)
        std_y = torch.sqrt(y.var(dim=0) + 0.0001)
        std_loss = torch.mean(F.relu(1 - std_x)) / 2 + torch.mean(F.relu(1 - std_y)) / 2  # variance term

        cov_x = (x.T @ x) / (batch_size - 1)
        cov_y = (y.T @ y) / (batch_size - 1)
        cov_loss = off_diagonal(cov_x).pow_(2).sum().div(
            self.dim_size
        ) + off_diagonal(cov_y).pow_(2).sum().div(self.dim_size)  # covariance term

        loss = (
            self.sim_coeff * repr_loss
            + self.std_coeff * std_loss
            + self.cov_coeff * cov_loss
        )

        return loss


