import torch.nn as nn
import torch

"""
1.	Reconstruction Loss: 
    - Measures how well the decoder reconstructs the input.

2.	KL Divergence Loss: 
    - Encourages the latent distribution to be close to a standard normal distribution.
"""

def vae_loss_function(pred, gt_sample, mu, log_var):
    reconstruction_loss = nn.functional.mse_loss(pred, gt_sample, reduction='sum')
    kl_divergence = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    return reconstruction_loss + kl_divergence

