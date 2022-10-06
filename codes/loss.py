#coding='utf-8'
import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossEntropyLoss(nn.Module):
    def __init__(self, cfg):
        super(CrossEntropyLoss, self).__init__()
        self.loss_func = nn.CrossEntropyLoss()
    def forward(self, y_pt, y_gt):
        return self.loss_func(y_pt, y_gt.squeeze().long())

class CrossEntropyLossAndL2Loss(nn.Module):
    def __init__(self, cfg):
        super(CrossEntropyLossAndL2Loss, self).__init__()
        self.cross = nn.CrossEntropyLoss()
        self.l1_loss = nn.L1Loss()
        self.l2_loss = nn.MSELoss()
    def forward(self, y_pt, y_gt, pre_pro, properties):
        cross_loss = self.cross(y_pt, y_gt)
        l2_area = self.l2_loss(pre_pro['area'], properties[:, 0])
        l2_orientation = self.l2_loss(pre_pro['orientation'], properties[:, 1])
        l2_perimeter = self.l2_loss(pre_pro['perimeter'], properties[:, 2])

        loss = 1.0 * cross_loss + 0.01 * l2_area + 0.01 * l2_orientation + 0.01 * l2_perimeter

        return loss

class MSELoss(nn.Module):
    def __init__(self, cfg):
        super(MSELoss, self).__init__()
        self.loss_func = nn.MSELoss()
    def forward(self, y_pt, y_gt):
        return self.loss_func(y_pt, y_gt)

class L1Loss(nn.Module):
    def __init__(self, cfg):
        super(L1Loss, self).__init__()
        self.loss_func = nn.L1Loss()
    def forward(self, y_pt, y_gt):
        return self.loss_func(y_pt, y_gt)

class RankingLoss(nn.Module):
    def __init__(self, cfg):
        super(RankingLoss, self).__init__()
        self.margin = cfg.train.margin
    def forward(self, y_pt, y_gt):

        y_pt = y_pt.sigmoid() # 经过sigmoid激活，确保在0-1范围内

        batch_size = y_pt.shape[0]

        left_ = y_pt[:batch_size//2] # 前半部分，类别1
        right_ = y_pt[batch_size//2:] # 后半部分，类别0
        new_gt = y_gt[:batch_size//2].float().cuda() # 标签，均为1

        rankingloss = F.margin_ranking_loss(left_, right_, new_gt, margin=self.margin, reduction='mean')
        return rankingloss

