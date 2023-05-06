"""
    cross entropy for one dimension and sup-isogclr for another dimension
"""

import torch
import torch.nn as nn
from torch.nn import functional as F



class TempGenerator(torch.nn.Module):
    def __init__(self, feature_dim, M=8, tau_min=0.05, tau_max=0.7):
        super(TempGenerator, self).__init__()

        self.feature_dim = feature_dim
        self.tau_min, self.tau_max = tau_min, tau_max
        self.temp_generator = nn.Sequential(
                                    nn.Linear(self.feature_dim, M),
                                    nn.Sigmoid(),
                                    nn.Linear(M, 1)
                                )

    def forward(self, x_1, x_2):
        tau_1 = torch.sigmoid(self.temp_generator(x_1))   # [bsz, dim] -> [bsz, 1]
        tau_2 = torch.sigmoid(self.temp_generator(x_2))
        tau_1 = (self.tau_max - self.tau_min) * tau_1.squeeze() + self.tau_min
        tau_2 = (self.tau_max - self.tau_min) * tau_2.squeeze() + self.tau_min
        tau = (tau_1 + tau_2) / 2.0
        return tau



class CE_iSogCLR_Loss(nn.Module):
    def __init__(self, 
                 multi_task,          # number of tasks in each batch
                 num_pos,             # number of positive items for each label
                 N=100000,            # size of moving average estimator (# samples)
                 tau=1.0,             # temperature parameter in iSogCLR loss
                 alpha=1.0,           # use alpha to balance cross-entropy loss and iSogCLR loss
                 enable_isogclr=False,
                 enable_temp_net=True, feature_dim=256,
                 tau_min=0.05, tau_max=0.7, rho=0.3, eta=0.01, beta=0.9,
                 debug_mode=False):
        super(CE_iSogCLR_Loss, self).__init__()

        self.multi_task = multi_task
        self.num_pos = num_pos
        self.alpha = alpha
        self.eps = 1e-6

        self.debug_mode = debug_mode

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.N = N
        self.u = torch.zeros(N).reshape(-1, 1).to(self.device)
        self.tau = tau

        # settings for isogclr
        self.enable_isogclr = enable_isogclr
        if self.enable_isogclr:
            self.tau_min, self.tau_max = tau_min, tau_max # lower and upper bound for learnable tau
            self.rho = rho                                # tunable parameter for isogclr, recommended values for unimodal tasks: [0.1~0.5]
            self.enable_temp_net = enable_temp_net
            if self.enable_temp_net:
                self.temp_net = TempGenerator(feature_dim=feature_dim, tau_min=tau_min, tau_max=tau_max).to(self.device)
            else:
                self.eta = eta                                # learning rate for learnable tau
                self.beta = beta                              # momentum parameter for the gradients of learnable tau
                self.learnable_tau = torch.ones(N).reshape(-1, 1) * self.tau
                self.grad_tau = torch.zeros(N).reshape(-1, 1)

    def forward(self, 
               logits,   # logit for each sample-label pair from network, shape: [multi_task * (num_pos + num_neg), num_classes]
               labels,   # label for each sample, shape: [multi_task * (num_pos + num_neg)]   
               index,    # index for each sample, shape: [multi_task * (num_pos + num_neg)]                        
               gamma=0.9):

        # cross entropy loss
        log_probs = F.log_softmax(logits, dim=1)
        ce_loss = F.nll_loss(log_probs, labels)

        # iSogCLR loss

        if self.debug_mode:
            print("labels:", labels)

        num_items_per_task = labels.shape[0] // self.multi_task
        task_labels = labels.reshape(self.multi_task, -1)[:, 0].repeat_interleave(num_items_per_task) # shape: [multi_task * (num_pos + num_neg)]

        if self.debug_mode:
            print("task_labels:", task_labels)
            print("logits:", logits)

        logits_pair = logits.gather(1, task_labels[:,None]).reshape(self.multi_task, -1) # shape: [multi_task, (num_pos + num_neg)]

        # we can perform cross entropy on this logits
        #labels = torch.zeros(self.multi_task, dtype=torch.long).cuda()
        #isogclr_loss = F.nll_loss(F.log_softmax(logits_pair, dim=1), labels)

        pos_logits_pair = logits_pair[:, :self.num_pos]
        neg_logits_pair = logits_pair[:, self.num_pos:]

        if self.debug_mode:
            print("pos_logits_pair:", pos_logits_pair)
            print("neg_logits_pair:", neg_logits_pair)

        neg_pos_logits_diff = torch.repeat_interleave(neg_logits_pair, self.num_pos, dim=0) \
                              - torch.cat(torch.unbind(pos_logits_pair))[:, None]            # shape: [multi_task * num_pos, num_neg]

        if self.debug_mode:
            print("neg_pos_logits_diff:", neg_pos_logits_diff)

        if self.enable_isogclr:
            if self.enable_temp_net:
                pass
            else:
                pass
        else:
            tau = self.tau 

        logits_diff_temp = (neg_pos_logits_diff / tau).detach()
        exp_logits_diff_temp = torch.exp(logits_diff_temp)               # shape: [multi_task * num_pos, num_neg]

        index_pos = index.reshape(self.multi_task, -1)[:, :self.num_pos]
        index_pos = torch.cat(torch.unbind(index_pos))                   # shape: [multi_task * num_pos]

        if self.debug_mode:
            print("index:", index)
            print("index_pos:", index_pos)

        if self.u[index_pos].sum() == 0:
            gamma = 1.0

        u = (1 - gamma) * self.u[index_pos] + gamma * (torch.mean(exp_logits_diff_temp, dim=1, keepdim=True) + self.eps) # shape: [multi_task * num_pos, 1]

        weights = exp_logits_diff_temp / u  # shape: [multi_task * num_pos, num_neg]

        isogclr_loss = torch.mean(weights * neg_pos_logits_diff, dim=1, keepdim=True) # shape: [multi_task * num_pos, 1]
        isogclr_loss = isogclr_loss.mean()

        loss = ce_loss + self.alpha * isogclr_loss

        return loss






