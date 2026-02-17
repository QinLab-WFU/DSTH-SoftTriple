import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn import init

class SoftTriple(nn.Module):
    def __init__(self, la, gamma, tau, margin, dim, cN, K, vi, device=0):
        super(SoftTriple, self).__init__()
        self.la = la
        self.gamma = 1. / gamma
        self.tau = tau
        self.margin = margin
        self.cN = cN
        self.K = K
        self.fc = Parameter(torch.Tensor(dim, cN * K))
        self.vi = vi
        self.weight = torch.zeros(cN * K, cN * K, dtype=torch.bool).to(device)
        for i in range(0, cN):
            for j in range(0, K):
                self.weight[i * K + j, i * K + j + 1:(i + 1) * K] = 1
        init.kaiming_uniform_(self.fc, a=math.sqrt(5))
        return

    def forward(self, images, texts, target):
        device = images.device
        centers = F.normalize(self.fc, p=2, dim=0)
        simInd_I = images.matmul(centers)
        simInd_T = texts.matmul(centers)
        simStruc_I = simInd_I.reshape(-1, self.cN, self.K)
        simStruc_T = simInd_T.reshape(-1, self.cN, self.K)
        prob_I = F.softmax(simStruc_I * self.gamma, dim=2)
        prob_T = F.softmax(simStruc_T * self.gamma, dim=2)
        simClass_I = torch.sum(prob_I * simStruc_I, dim=2)
        simClass_T = torch.sum(prob_T * simStruc_T, dim=2)
        marginM_I = torch.zeros(simClass_I.shape).to(device)
        marginM_T = torch.zeros(simClass_T.shape).to(device)
        target = target.to(device)
        marginM_I = target - marginM_I
        marginM_T = target - marginM_T
        marginM_I[marginM_I > 0] = self.margin
        marginM_T[marginM_T > 0] = self.margin
        target = target.float()
        lossClassify_I = F.cross_entropy(self.la * (simClass_I - marginM_I), target)
        lossClassify_T = F.cross_entropy(self.la * (simClass_T - marginM_T), target)
        lossClassify = lossClassify_I + lossClassify_T
        if self.tau > 0 and self.K > 1:
            simCenter = centers.t().matmul(centers)
            reg = torch.sum(torch.sqrt(2.0 + 1e-5 - 2. * simCenter[self.weight])) / (self.cN * self.K * (self.K - 1.))
            index = target.sum(dim=1) > 1
            label_ = target[index].float()
            x_ = images[index]
            t_ = texts[index]
            cos_sim = label_.mm(label_.T)
            if len((cos_sim == 0).nonzero()) == 0:
                reg_term_i = 0
                reg_term_t = 0
                reg_term_it = 0
            else:
                i_sim = F.normalize(x_, p=2, dim=1).mm(F.normalize(x_, p=2, dim=1).T)
                t_sim = F.normalize(t_, p=2, dim=1).mm(F.normalize(t_, p=2, dim=1).T)
                it_sim = F.normalize(x_, p=2, dim=1).mm(F.normalize(t_, p=2, dim=1).T)
                neg_i = self.vi * F.relu(i_sim)
                neg_t = self.vi * F.relu(t_sim)
                neg_it = self.vi * F.relu(it_sim)
                reg_term_i = torch.where(cos_sim == 0, neg_i, torch.zeros_like(i_sim)).sum() / len((cos_sim == 0).nonzero())
                reg_term_t = torch.where(cos_sim == 0, neg_t, torch.zeros_like(t_sim)).sum() / len((cos_sim == 0).nonzero())
                reg_term_it = torch.where(cos_sim == 0, neg_it, torch.zeros_like(it_sim)).sum() / len((cos_sim == 0).nonzero())
            return lossClassify + self.tau * reg + reg_term_i + reg_term_t + reg_term_it
        else:
            return lossClassify
