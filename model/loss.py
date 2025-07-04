# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import pdb
import torch
import torch.nn.functional as F
from torch import nn
import pickle

class NormSoftmaxLoss(nn.Module):
    def __init__(self, temperature=0.05):
        super().__init__()

        self.temperature = temperature

    def forward(self, x):
        "Assumes input x is similarity matrix of N x M \in [-1, 1], computed using the cosine similarity between normalised vectors"
        i_logsm = F.log_softmax(x/self.temperature, dim=1)
        j_logsm = F.log_softmax(x.t()/self.temperature, dim=1)

        # sum over positives
        idiag = torch.diag(i_logsm)
        loss_i = idiag.sum() / len(idiag)

        jdiag = torch.diag(j_logsm)
        loss_j = jdiag.sum() / len(jdiag)

        return - loss_i - loss_j

class EgoNCE(nn.Module):
    def __init__(self, temperature=0.05, noun=True, verb=True):
        super().__init__()
        self.noun = noun
        self.verb = verb
        self.temperature = temperature

    def forward(self, x, mask_v, mask_n):
        mask_diag = torch.eye(x.shape[0]).cuda()
        if self.noun and self.verb:
            mask = mask_v * mask_n + mask_diag
        elif self.noun:
            mask = mask_n + mask_diag
        else:
            mask = mask_v + mask_diag

        "Assumes input x is similarity matrix of N x M \in [-1, 1], computed using the cosine similarity between normalised vectors"
        i_sm = F.softmax(x/self.temperature, dim=1)
        j_sm = F.softmax(x.t()/self.temperature, dim=1)

        mask_bool = mask > 0
        idiag = torch.log(torch.sum(i_sm * mask_bool, dim=1) )
        loss_i = idiag.sum() / len(idiag)

        jdiag = torch.log(torch.sum(j_sm * mask_bool, dim=1) )
        loss_j = jdiag.sum() / len(jdiag)
        return - loss_i - loss_j

# https://github.com/facebookresearch/r3m/blob/main/r3m/trainer.py
class InfoNCE(nn.Module):
    def __init__(self, temperature=0.05, num_negatives=3, noun=True, verb=True):
        super().__init__()
        self.temperature = temperature
        self.num_neg = num_negatives
        self.noun = noun
        self.verb = verb

    def forward_tcn(self, frame_embeds, text_embeds, mask_bool):
        before, after, CF1, CF2, CF3 = text_embeds
        before.requires_grad = False
        after.requires_grad = False
        CF1.requires_grad = False
        CF2.requires_grad = False
        CF3.requires_grad = False
        epsilon = 1e-8

        bs, num_frame, _ = frame_embeds.shape
        frame_embeds = frame_embeds.reshape(bs, num_frame, -1)
        f0 = frame_embeds[:, 0]
        f1 = frame_embeds[:, 1]
        f2 = frame_embeds[:, 2]
        f3 = frame_embeds[:, 3]

        sim_0_1 = sim_matrix(f0, f1)
        sim_0_3 = sim_matrix(f0, f3)
        sim_0_before = sim_matrix(f0, before)
        sim_0_after = sim_matrix(f0, after)
        sim_0_cf1 = sim_matrix(f0, CF1)
        sim_0_cf2 = sim_matrix(f0, CF2)
        sim_0_cf3 = sim_matrix(f0, CF3)

        sim_3_0 = sim_matrix(f3, f0) 
        sim_3_2 = sim_matrix(f3, f2)
        sim_3_after = sim_matrix(f3, after)
        sim_3_before = sim_matrix(f3, before)
        sim_3_cf1 = sim_matrix(f3, CF1)
        sim_3_cf2 = sim_matrix(f3, CF2)
        sim_3_cf3 = sim_matrix(f3, CF3)

        denom_tcn_0 = epsilon + torch.exp(sim_0_1/self.temperature) + torch.exp(sim_0_before/self.temperature) + \
            torch.exp(sim_0_3/self.temperature) + torch.exp(sim_0_after/self.temperature) + \
            torch.exp(sim_0_cf1/self.temperature) + torch.exp(sim_0_cf2/self.temperature) + torch.exp(sim_0_cf3/self.temperature) 

        denom_tcn_3 = epsilon + torch.exp(sim_3_2/self.temperature) + torch.exp(sim_3_after/self.temperature) + \
                torch.exp(sim_3_0/self.temperature) + torch.exp(sim_3_before/self.temperature) + \
                torch.exp(sim_3_cf1/self.temperature) + torch.exp(sim_3_cf2/self.temperature) + torch.exp(sim_3_cf3/self.temperature)       

        denom_tcn_0 = denom_tcn_0.sum(-1).unsqueeze(-1)
        denom_tcn_3 = denom_tcn_3.sum(-1).unsqueeze(-1)

        tcn_0 = -torch.log( torch.sum( ( (torch.exp(sim_0_1/self.temperature) + epsilon) / denom_tcn_0 ) * mask_bool, dim=-1 ) ) - \
                torch.log( torch.sum( ( (torch.exp(sim_0_before/self.temperature) + epsilon) / denom_tcn_0 ) * mask_bool, dim=-1 ) )
            
        tcn_3 = -torch.log( torch.sum( ( (torch.exp(sim_3_2/self.temperature) + epsilon) / denom_tcn_3 ) * mask_bool, dim=-1 ) ) - \
                torch.log( torch.sum( ( (torch.exp(sim_3_after/self.temperature) + epsilon) / denom_tcn_3 ) * mask_bool, dim=-1 ) )
        
        tcn_0 /= 2
        tcn_3 /= 2
        
        tcn = .2*((tcn_3 + tcn_0) / 2.0).mean()

        return tcn


    def forward(self, text_embeds, video_embeds, v_embeds, n_embeds, frame_embeds):
        loss_dict = {}
        
        narration, before, after, CF1, CF2, CF3 = text_embeds
        narration = narration[0]  # TODO check why this is a tuple
        narration.requires_grad = False
        before.requires_grad = False
        after.requires_grad = False
        CF1.requires_grad = False
        CF2.requires_grad = False
        CF3.requires_grad = False

        # video_text_alignment
        assert video_embeds.requires_grad
        # video-text only
        # EgoNCE

        loss_align, x, _ = self.forward_align(narration, video_embeds, v_embeds, n_embeds)
        mask_bool = torch.eye(x.shape[0]).cuda()
        
        loss_dict['align'] = loss_align.item()
        
        ## Within Video TCN Loss

        assert frame_embeds.requires_grad
        
        tcn = self.forward_tcn(frame_embeds, text_embeds[1:], mask_bool)

        loss_dict['tcn'] = tcn.item()

        loss = loss_align + tcn

        return loss_dict, loss
    
    def forward_align(self, text_embeds, video_embeds, v_embeds, n_embeds, parent=False):
        x = sim_matrix(video_embeds, text_embeds)
        # print(x.shape)
        mask_diag = torch.eye(x.shape[0]).cuda()

        mask_v = sim_matrix(v_embeds, v_embeds)
        mask_n = sim_matrix(n_embeds, n_embeds)
        
        if self.noun and self.verb:
            mask = mask_v * mask_n + mask_diag
        elif self.noun:
            mask = mask_n + mask_diag
        else:
            mask = mask_v + mask_diag
        
        mask_bool = (mask > 0)*1

        if parent:
            return 0., x, mask_bool
        
        "Assumes input x is similarity matrix of N x M \in [-1, 1], computed using the cosine similarity between normalised vectors"
        align_sm = F.softmax(x/self.temperature, dim=1) 
        idiag = torch.log(torch.sum(align_sm * mask_bool, dim=1) )

        loss_align = idiag.sum() / len(idiag)
        loss_align = - loss_align

        return loss_align, x, mask_bool
    
    def forward_cf(self, text_embeds, video_embeds, summ_align, mask):
        epsilon = 1e-8
        dim = text_embeds.shape[0]
        
        vid_expanded = video_embeds.unsqueeze(0).expand(dim, -1, -1)
        sim_cf = F.cosine_similarity(vid_expanded.unsqueeze(2), text_embeds.unsqueeze(1), dim=-1)
        align_sim = torch.exp(summ_align/self.temperature) + epsilon

        denom_tcn_0 = torch.exp(sim_cf/self.temperature).sum(0) + align_sim
        
        denom_tcn_1 = denom_tcn_0.sum(-1).unsqueeze(-1)
            
        tcn_0 = -torch.log( torch.sum( ( align_sim / denom_tcn_1 ) * mask, dim=-1 ) ) 
        
        tcn = tcn_0.mean().contiguous()

        return tcn
        
    
    def forward_summary(self, summary_embeds, video_embeds, cf_parent, v_embeds, n_embeds):

        _, x, mask_bool = self.forward_align(
                        text_embeds=summary_embeds, 
                        video_embeds=video_embeds,
                        v_embeds=v_embeds,
                        n_embeds=n_embeds,
                        parent=True,
                        )
        
        loss_align = self.forward_cf(
                        text_embeds=cf_parent, 
                        video_embeds=video_embeds,
                        summ_align=x,
                        mask=mask_bool,
                        )

        loss_dict = {
            "parent_align": loss_align.item(),
        }

        return loss_dict, loss_align


def sim(tensor1, tensor2):
    cs = torch.nn.CosineSimilarity(1)
    # if l2dist:
    #     d = - torch.linalg.norm(tensor1 - tensor2, dim = -1)
    # else:
    d = cs(tensor1, tensor2)
    return d

def sim_matrix(a, b, eps=1e-8):
    """
    added eps for numerical stability
    """
    # a_n, b_n = a.norm(dim=1)[:, None], b.norm(dim=1)[:, None]
    a_norm = F.normalize(a, dim=-1) #a / torch.max(a_n, eps * torch.ones_like(a_n))
    b_norm = F.normalize(b, dim=-1) #b / torch.max(b_n, eps * torch.ones_like(b_n))

    sim_mt = torch.mm(a_norm, b_norm.T)
    return sim_mt

class EgoMILNCE(nn.Module):
    def __init__(self, temperature=0.05):
        super().__init__()
        self.temperature = temperature

    def forward(self, x, num_samples):
        mask_diag = torch.eye(num_samples)
        if x.shape[0] % num_samples != 0 or x.shape[1] % num_samples != 0:
            print('ERROR: Incorrect num_samples. shapes must be divisible by num_samples...')
            raise ValueError
        patch = torch.ones(int(x.shape[0]/num_samples), int(x.shape[1]/num_samples))
        mask = torch.kron(mask_diag, patch).cuda()

        # TODO: Do we need mask_v and mask_n for hierarchical contrastive learning as well?

        "Assumes input x is similarity matrix of N x M \in [-1, 1], computed using the cosine similarity between normalised vectors"
        i_sm = F.softmax(x/self.temperature, dim=1)
        j_sm = F.softmax(x.t()/self.temperature, dim=1)

        mask_bool = mask > 0
        idiag = torch.log(torch.sum(i_sm * mask_bool, dim=1) )
        loss_i = idiag.sum() / len(idiag)

        jdiag = torch.log(torch.sum(j_sm * mask_bool.T, dim=1) )
        loss_j = jdiag.sum() / len(jdiag)
        return - loss_i - loss_j

class MaxMarginRankingLoss(nn.Module):

    def __init__(self, margin=0.2, fix_norm=True):
        super().__init__()
        self.fix_norm = fix_norm
        self.loss = nn.MarginRankingLoss(margin)
        self.margin = margin

    def forward(self, x, weight=None):
        n = x.size()[0]

        x1 = torch.diag(x)
        x1 = x1.unsqueeze(1)
        x1 = x1.expand(n, n)
        x1 = x1.contiguous().view(-1, 1)
        x1 = torch.cat((x1, x1), 0)

        x2 = x.view(-1, 1)
        x3 = x.transpose(0, 1).contiguous().view(-1, 1)

        x2 = torch.cat((x2, x3), 0)
        max_margin = F.relu(self.margin - (x1 - x2))

        if self.fix_norm:
            # remove the elements from the diagonal
            keep = torch.ones(x.shape) - torch.eye(x.shape[0])  # 128 x 128
            keep1 = keep.view(-1, 1)
            keep2 = keep.transpose(0, 1).contiguous().view(-1, 1)
            keep_idx = torch.nonzero(torch.cat((keep1, keep2), 0).flatten()).flatten()
            if x1.is_cuda:
                keep_idx = keep_idx.cuda()
            x1_ = torch.index_select(x1, dim=0, index=keep_idx)
            x2_ = torch.index_select(x2, dim=0, index=keep_idx)
            max_margin = F.relu(self.margin - (x1_ - x2_))

        return max_margin.mean()

class AdaptiveMaxMarginRankingLoss(nn.Module):

    def __init__(self, margin=0.4, fix_norm=True):
        super().__init__()
        self.fix_norm = fix_norm
        self.loss = nn.MarginRankingLoss(margin)
        self.margin = margin

    def forward(self, x, weight=None):
        n = x.size()[0]

        x1 = torch.diag(x)
        x1 = x1.unsqueeze(1)
        x1 = x1.expand(n, n)
        x1 = x1.contiguous().view(-1, 1)
        x1 = torch.cat((x1, x1), 0)

        w1 = weight.unsqueeze(1)
        w1 = w1.expand(n, n)
        w1 = w1.contiguous().view(-1, 1)
        w1 = torch.cat((w1, w1), 0)

        x2 = x.view(-1, 1)
        x3 = x.transpose(0, 1).contiguous().view(-1, 1)

        x2 = torch.cat((x2, x3), 0)
        max_margin = F.relu(  w1 * self.margin - (x1 - x2))

        if self.fix_norm:
            # remove the elements from the diagonal
            keep = torch.ones(x.shape) - torch.eye(x.shape[0])  # 128 x 128
            keep1 = keep.view(-1, 1)
            keep2 = keep.transpose(0, 1).contiguous().view(-1, 1)
            keep_idx = torch.nonzero(torch.cat((keep1, keep2), 0).flatten()).flatten()
            if x1.is_cuda:
                keep_idx = keep_idx.cuda()
            x1_ = torch.index_select(x1, dim=0, index=keep_idx)
            w1_ = torch.index_select(w1, dim=0, index=keep_idx)
            x2_ = torch.index_select(x2, dim=0, index=keep_idx)
            max_margin =  F.relu( w1_ * self.margin - (x1_ - x2_))

        return max_margin.mean()

class CrossEntropy(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss = nn.CrossEntropyLoss()

    def forward(self, output, target):
        return self.loss(output, target)