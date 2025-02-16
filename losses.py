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

# These are some loss functions to test out, I have added 4 for now. We need further experimentation to check the best combination.
# Chamfer Distance Loss
def chamfer_distance_loss(pred_points, target_points):
    """
    Computes Chamfer Distance between predicted and target point clouds.
    """
    pred_exp = pred_points.unsqueeze(2)  # Expand dimensions for broadcasting
    target_exp = target_points.unsqueeze(1)

    dist_matrix = torch.norm(pred_exp - target_exp, dim=-1)  # Pairwise distances
    min_pred_to_target = torch.min(dist_matrix, dim=2)[0]
    min_target_to_pred = torch.min(dist_matrix, dim=1)[0]

    chamfer_loss = torch.mean(min_pred_to_target) + torch.mean(min_target_to_pred)
    return chamfer_loss

# Hausdorff Distance Loss (Approximate)
def hausdorff_distance_loss(pred_points, target_points):
    """
    Computes approximate Hausdorff Distance between predicted and target point clouds.
    """
    pred_exp = pred_points.unsqueeze(2)
    target_exp = target_points.unsqueeze(1)

    dist_matrix = torch.norm(pred_exp - target_exp, dim=-1)
    max_pred_to_target = torch.max(torch.min(dist_matrix, dim=2)[0], dim=1)[0]
    max_target_to_pred = torch.max(torch.min(dist_matrix, dim=1)[0], dim=1)[0]

    hausdorff_loss = torch.mean(max_pred_to_target + max_target_to_pred)
    return hausdorff_loss

# Surface Loss 
def surface_loss(pred_surface, target_surface):
    """
    Penalizes discrepancies along surface boundaries.
    """
    loss = torch.mean((pred_surface - target_surface) ** 2)
    return loss

# Bending Energy Loss
def bending_energy_loss(deformation_field):
    """
    Encourages smooth deformations by penalizing second-order differences.
    """
    dxx = torch.abs(deformation_field[:, :, 2:, :, :] - 2 * deformation_field[:, :, 1:-1, :, :] + deformation_field[:, :, :-2, :, :])
    dyy = torch.abs(deformation_field[:, :, :, 2:, :] - 2 * deformation_field[:, :, :, 1:-1, :] + deformation_field[:, :, :, :-2, :])
    dzz = torch.abs(deformation_field[:, :, :, :, 2:] - 2 * deformation_field[:, :, :, :, 1:-1] + deformation_field[:, :, :, :, :-2])

    return torch.mean(dxx ** 2) + torch.mean(dyy ** 2) + torch.mean(dzz ** 2)
