import torch
import torch.nn.functional as F

def kl_divergence(alpha, num_classes, device=None):
    if not device:
        device = alpha.device
    beta = torch.ones([1, num_classes], dtype=torch.float32, device=device)
    S_alpha = torch.sum(alpha, dim=1, keepdim=True)
    S_beta = torch.sum(beta, dim=1, keepdim=True)
    lnB = torch.lgamma(S_alpha) - torch.sum(torch.lgamma(alpha), dim=1, keepdim=True)
    lnB_uni = torch.sum(torch.lgamma(beta), dim=1, keepdim=True) - torch.lgamma(S_beta)
    
    dg0 = torch.digamma(S_alpha)
    dg1 = torch.digamma(alpha)
    
    kl = torch.sum((alpha - beta) * (dg1 - dg0), dim=1, keepdim=True) + lnB + lnB_uni
    return kl

def weighted_edl_loss(output, target, epoch_num, num_classes, annealing_step, class_weights=None, ignore_index=-1, device=None):
    """
    Weighted Evidential Deep Learning (EDL) Loss
    Ensures mathematical scaling by [0.8, 1.2] to suppress class imbalance hallucination.
    """
    if not device:
         device = output.device
         
    # 1. Evidence and Dirichlet parameters
    evidence = output
    alpha = evidence + 1.0 
    
    B, C, H, W = alpha.shape
    alpha = alpha.permute(0, 2, 3, 1).reshape(-1, C) # [B*H*W, C]
    
    # Target and Ignored Masking
    target_flat = target.reshape(-1) # [B*H*W]
    
    # Create mask for valid pixels (not ignore_index)
    valid_mask = target_flat != ignore_index
    if not valid_mask.any():
        return torch.tensor(0.0, device=device, requires_grad=True)
        
    # Filter variables to only valid pixels
    alpha_valid = alpha[valid_mask]
    target_valid = target_flat[valid_mask]
    
    # 2. One-hot target
    y = F.one_hot(target_valid, num_classes=num_classes).float() 
    
    S = torch.sum(alpha_valid, dim=1, keepdim=True)
    
    # 3. Apply Class Weights to combat imbalance
    if class_weights is None:
        class_weights = torch.ones(num_classes, device=device)
    else:
        class_weights = torch.tensor(class_weights, device=device, dtype=torch.float32)
        
    pixel_weights = class_weights[target_valid].unsqueeze(1) # [N_valid, 1]
    
    # 4. Expected Cross-Entropy (Risk) using Digamma
    risk = y * (torch.digamma(S) - torch.digamma(alpha_valid))
    weighted_risk = pixel_weights * risk
    loss_ce = torch.sum(weighted_risk, dim=1, keepdim=True)
    
    # 5. KL Divergence Regularization
    annealing_coef = torch.min(torch.tensor(1.0, dtype=torch.float32), 
                               torch.tensor(epoch_num / annealing_step, dtype=torch.float32))
    
    alpha_tilde = y + (1 - y) * alpha_valid
    kl_div = kl_divergence(alpha_tilde, num_classes, device=device)
    
    # KL penalty must ALSO be weighted to suppress false-positive evidence
    weighted_kl_div = pixel_weights * kl_div
    
    loss = loss_ce + annealing_coef * weighted_kl_div
    
    return torch.mean(loss)
