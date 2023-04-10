import torch
import torch.nn.functional as F
import torch.nn as nn
from einops import repeat, rearrange, reduce

def combined_loss(logits, targets, alpha=0.2, gamma=2.0, smooth=1e-5, reduction='mean'):
    # Calculate the focal loss
    fl = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
    pt = torch.exp(-fl)
    focal_loss = alpha * (1 - pt) ** gamma * fl

    if reduction == 'mean':
        fl = torch.mean(focal_loss)
    elif reduction == 'sum':
        fl = torch.sum(focal_loss)

    # Calculate the Dice loss
    prob = torch.sigmoid(logits)
    intersection = torch.sum(prob * targets, dim=(2, 3))
    union = torch.sum(prob + targets, dim=(2, 3))
    dice_loss = 1 - (2 * intersection + smooth) / (union + smooth)

    return focal_loss, dice_loss
    
    if reduction == 'mean':
        dl = torch.mean(dice_loss)
    elif reduction == 'sum':
        dl = torch.sum(dice_loss)

    # Combine the losses using the specified ratio
    loss = 20 * fl + dl

    return loss


# Assuming your prediction and ground truth tensors are named `pred` and `gt`, respectively
def mse_loss(pred, gt):
    mse_loss = nn.MSELoss(reduction='none')
    loss = mse_loss(pred, gt)
    return loss

def compute_iou(pred_mask, gt_mask):
    intersection = torch.logical_and(pred_mask, gt_mask)
    intersection = reduce(intersection, "b c h w -> b c", reduction='sum')
    union = torch.logical_or(pred_mask, gt_mask)
    union = reduce(union, "b c h w -> b c", reduction='sum') + 1e-8
    iou = intersection / union # if union > 0 else 0
    return iou


def ranked_combined_loss(pred_mask, gt_mask, iou_pred):
    gt_mask = repeat(gt_mask, "b 1 h w -> b c h w", c=3)
    fl, dl = combined_loss(pred_mask, gt_mask)
    fl = reduce(fl, "b c h w -> b c", reduction="mean")
    # dl = reduce(dl, "b c -> b c")
    segment_loss = 20*fl + dl
    min_losses, min_loss_indices = torch.min(segment_loss, dim=1)
    iou_loss = mse_loss(iou_pred, compute_iou(pred_mask, gt_mask))

    selected_losses = torch.gather(iou_loss, 1, min_loss_indices.unsqueeze(1))

    total_loss = min_losses + selected_losses
    total_loss = total_loss.mean()
    return total_loss, min_losses, selected_losses


def compute_all_loss(pred_mask, gt_mask, iou_pred):
    # import ipdb; ipdb.set_trace()
    fl, dl = combined_loss(pred_mask, gt_mask)
    segment_loss = 20*fl.mean() + dl.mean()
    iou_loss = mse_loss(iou_pred, compute_iou(pred_mask, gt_mask))
    total_loss = segment_loss.mean() + iou_loss.mean()
    return total_loss, segment_loss, iou_loss


# def stage2_loss(pred_mask, gt_mask):
#     loss

