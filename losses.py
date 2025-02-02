import torch

def dice_loss(pred, target, smooth=1e-5):
    intersection = torch.sum(pred * target, dim=(2, 3, 4))
    pred_sum = torch.sum(pred, dim=(2, 3, 4))
    target_sum = torch.sum(target, dim=(2, 3, 4))
    
    dice = (2.0 * intersection + smooth) / (pred_sum + target_sum + smooth)
    return 1 - dice.mean()

def smoothing_loss(deformation_field):
    dx = torch.abs(deformation_field[:, :, 1:, :, :] - deformation_field[:, :, :-1, :, :])
    dy = torch.abs(deformation_field[:, :, :, 1:, :] - deformation_field[:, :, :, :-1, :])
    dz = torch.abs(deformation_field[:, :, :, :, 1:] - deformation_field[:, :, :, :, :-1])

    return torch.mean(dx ** 2) + torch.mean(dy ** 2) + torch.mean(dz ** 2)

def composite_loss(pred, target, deformation_field, dice_weight=1.0, smooth_weight=0.1):
    dice = dice_loss(pred, target)
    smooth = smoothing_loss(deformation_field)
    total_loss = dice_weight * dice + smooth_weight * smooth
    return total_loss, {"dice_loss": dice.item(), "smooth_loss": smooth.item()}
