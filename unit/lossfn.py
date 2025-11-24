import random
import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
from unit.plot_curves import plot_training_curves, plot_combined_task_curves
import gc
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
import torch.nn.functional as F

class FocalLoss_original(nn.Module):
    def __init__(self, alpha=0.25, gamma=2, reduction='mean'):
        super(FocalLoss_original, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, logits, targets, sigma_sq, key):
        """
        :param logits: Raw model outputs of shape (batch_size, ) or (batch_size, num_tasks)
        :param targets: Ground truth labels of shape (batch_size, ) or (batch_size, num_tasks)
        """
        probs = logits
        targets = targets.type_as(probs)  # Ensure same dtype

        pt = torch.where(targets == 1, probs, 1 - probs)  # p_t
        alpha_factor = torch.where(targets == 1, self.alpha, 1 - self.alpha)
        focal_weight = alpha_factor * (1 - pt) ** self.gamma

        bce_loss = F.binary_cross_entropy(probs, targets, reduction='none')
        loss = focal_weight * bce_loss

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss  # 'none'

class FocalLoss_Ada_flood(nn.Module):
    def __init__(self, alpha=0.25, gamma=2, reduction='mean'):
        super(FocalLoss_Ada_flood, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    # def forward(self, input, target, sigma_sq, key):
    #     # ce_loss = nn.CrossEntropyLoss(reduction='none')(input, target)  # 计算交叉熵损失
    #     # ce_loss = nn.BCELoss()(input, target)
    #     p = input[:,1]
    #     focal_loss = -self.alpha * (1 - p) ** self.gamma * torch.log(p)
    #     # loss_mean = focal_loss.mean()  # 计算Focal Loss
    #     # loss_sum = focal_loss.sum()
    #     # ce_loss = nn.BCEWithLogitsLoss()(input, target)
    #     # pt = torch.exp(-ce_loss)  # 计算概率
    #     # focal_loss_origianl = self.alpha * (1 - pt) ** self.gamma * ce_loss  # 计算Focal Loss
        
    #     if self.reduction == 'mean':
    #         return focal_loss.mean()
    #     elif self.reduction == 'sum':
    #         return focal_loss.sum()
    #     else:
    #         return focal_loss
    def forward(self, logits, targets, log_vars, key):
        """
        :param logits: Raw model outputs of shape (batch_size, ) or (batch_size, num_tasks)
        :param targets: Ground truth labels of shape (batch_size, ) or (batch_size, num_tasks)
        """
        probs = logits
        targets = targets.type_as(probs)  # Ensure same dtype

        pt = torch.where(targets == 1, probs, 1 - probs)  # p_t
        alpha_factor = torch.where(targets == 1, self.alpha, 1 - self.alpha)
        focal_weight = alpha_factor * (1 - pt) ** self.gamma

        bce_loss = F.binary_cross_entropy(probs, targets, reduction='none')
        loss = focal_weight * bce_loss

        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()
        else:
            loss  # 'none'
        
        # b = log_vars[key]
        # b = 0.005

        # return (loss-b).abs()+b
        return loss

class AdaptiveFocalLoss(nn.Module):
    def __init__(self, reduction='mean'):
        super(AdaptiveFocalLoss, self).__init__()
        # self.alpha = alpha
        # self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, input, target, sigma_sq, key):
        # Dynamically adjust focal loss parameters
        gamma = 2.5 - (sigma_sq[key] / sigma_sq.mean())
        alpha = (sigma_sq[key] / sigma_sq.max())
        # alpha = 0.85
        # print('task:', key, 'gamma:', gamma, 'alpha:', alpha)
        # ce_loss = nn.CrossEntropyLoss(reduction='none')(input, target)  # 计算交叉熵损失
        ce_loss = nn.BCELoss()(input, target)
        # ce_loss = nn.BCEWithLogitsLoss()(input, target)
        pt = torch.exp(-ce_loss)  # 计算概率
        focal_loss = alpha * (1 - pt) ** gamma * ce_loss  # 计算Focal Loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-1):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, input, target):
        # Flatten the input and target tensors
        input_flat = input.reshape(-1)
        target_flat = target.reshape(-1)

        intersection = (input_flat * target_flat).sum()
        dice_score = (2. * intersection + self.smooth) / (input_flat.sum() + target_flat.sum() + self.smooth)

        return 1 - dice_score  # Return Dice Loss (lower is better)

class CombinedLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, dice_weight=1.0, focal_weight=1.0, reduction='mean'):
        super(CombinedLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight
        self.reduction = reduction
        self.focal_loss = AdaptiveFocalLoss(reduction=self.reduction)
        self.dice_loss = DiceLoss()

    def forward(self, input, target):
        # Calculate Focal Loss
        focal_loss_value = self.focal_loss(input, target)

        # Calculate Dice Loss
        dice_loss_value = self.dice_loss(input, target)

        # Combine the two losses
        total_loss = self.focal_weight * focal_loss_value + self.dice_weight * dice_loss_value

        return total_loss