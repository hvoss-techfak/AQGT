import os
import sys

sys.path.append(os.getcwd())

import torch
import torch.nn as nn
import torch.nn.functional as F




# from https://github.com/TheTempAccount/Co-Speech-Motion-Generation/blob/main/src/losses/losses.py
class KeypointLoss(nn.Module):
    def __init__(self):
        super(KeypointLoss, self).__init__()

    def forward(self, pred_seq, gt_seq, gt_conf=None):
        # pred_seq: (B, C, T)
        if gt_conf is not None:
            gt_conf = gt_conf >= 0.01
            return F.mse_loss(pred_seq[gt_conf], gt_seq[gt_conf], reduction='mean')
        else:
            return F.mse_loss(pred_seq, gt_seq)

# from https://github.com/TheTempAccount/Co-Speech-Motion-Generation/blob/main/src/losses/losses.py
class VelocityLoss(nn.Module):
    def __init__(self,
                 velocity_length
                 ):
        super(VelocityLoss, self).__init__()
        self.velocity_length = velocity_length

    def forward(self, pred_seq, gt_seq, prev_seq):
        B, C, T = pred_seq.shape
        last_frame = prev_seq[:, :, -1:]
        gt_velocity = gt_seq - torch.cat([last_frame, gt_seq[:, :, :-1]], dim=-1)
        pred_velocity = pred_seq - torch.cat([last_frame, pred_seq[:, :, :-1]], dim=-1)

        assert gt_velocity.shape[0] == B and gt_velocity.shape[1] == C and gt_velocity.shape[2] == T
        gt_velocity = gt_velocity[:, :, :self.velocity_length]
        pred_velocity = pred_velocity[:, :, :self.velocity_length]
        return F.mse_loss(pred_velocity, gt_velocity)


class V7_loss(nn.Module):
    """
    V7_loss calculates the velocity, acceleration, jerk, jounce, crackle, pop, and lock for the generated gestures.
    """
    def __init__(self):

        super(V7_loss, self).__init__()
        self.criterion = F.smooth_l1_loss

    def calculateDiff(self, seq):
        """
        Calculating the derivative in regarding to time. We set the time to 1 frame, to ignore it and prevent exploding
        loss functions.
        @param seq:
        @return:
        """
        ret = seq.new_zeros((seq.shape[0], seq.shape[1] - 1, 159))
        for i in range(0, ret.shape[1]):
            frame = seq[:, i, :]
            next_frame = seq[:, i + 1, :]
            distance = ((next_frame - frame) + 1e-10).pow(2).sqrt()
            ret[:, i, :] = distance
        return ret

    def calculateV(self, pred_seq, gt_seq):
        return self.calculateDiff(pred_seq), self.calculateDiff(gt_seq)

    def forward(self, pred_seq, gt_seq):
        beta = 0.1
        velocity_pred, velocity_gt = self.calculateV(pred_seq, gt_seq)
        acceleration_pred, acceleration_gt = self.calculateV(velocity_pred, velocity_gt)
        jerk_pred, jerk_gt = self.calculateV(acceleration_pred, acceleration_gt)
        jounce_pred, jounce_gt = self.calculateV(jerk_pred, jerk_gt)
        crackle_pred, crackle_gt = self.calculateV(jounce_pred, jounce_gt)
        pop_pred, pop_gt = self.calculateV(crackle_pred, crackle_gt)
        lock_pred, lock_gt = self.calculateV(pop_pred, pop_gt)

        v1 = self.criterion(velocity_pred / beta, velocity_gt / beta) * beta
        v2 = self.criterion(acceleration_pred / beta, acceleration_gt / beta) * beta
        v3 = self.criterion(jerk_pred / beta, jerk_gt / beta) * beta
        v4 = self.criterion(jounce_pred / beta, jounce_gt / beta) * beta
        v5 = self.criterion(crackle_pred / beta, crackle_gt / beta) * beta
        v6 = self.criterion(pop_pred / beta, pop_gt / beta) * beta
        v7 = self.criterion(lock_pred / beta, lock_gt / beta) * beta

        return v1, v2, v3, v4, v5, v6, v7

# from https://github.com/TheTempAccount/Co-Speech-Motion-Generation/blob/main/src/losses/losses.py
class KLLoss(nn.Module):
    def __init__(self, kl_tolerance):
        super(KLLoss, self).__init__()
        self.kl_tolerance = kl_tolerance

    def forward(self, mu, var):
        kld_loss = -0.5 * torch.sum(1 + var - mu ** 2 - var.exp(), dim=1)
        if self.kl_tolerance is not None:
            above_line = kld_loss[kld_loss > self.kl_tolerance]
            if len(above_line) > 0:
                kld_loss = torch.mean(kld_loss, dim=0)
            else:
                kld_loss = 0
        else:
            kld_loss = torch.mean(kld_loss, dim=0)
        return kld_loss

# from https://github.com/TheTempAccount/Co-Speech-Motion-Generation/blob/main/src/losses/losses.py
class L2RegLoss(nn.Module):
    def __init__(self):
        super(L2RegLoss, self).__init__()

    def forward(self, x):
        # TODO: check
        return torch.sum(x ** 2)

# from https://github.com/TheTempAccount/Co-Speech-Motion-Generation/blob/main/src/losses/losses.py
class AudioLoss(nn.Module):
    def __init__(self):
        super(AudioLoss, self).__init__()

    def forward(self, dynamics, gt_poses):
        # pay attention, normalized
        mean = torch.mean(gt_poses, dim=-1).unsqueeze(-1)
        gt = gt_poses - mean
        return F.mse_loss(dynamics, gt)


L1Loss = nn.L1Loss
