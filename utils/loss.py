#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch
import numpy as np
from .kspace import get_3D_shell

def charbonnier_loss(prediction, target, epsilon=1e-6):
    error = torch.sqrt((prediction - target)**2 + epsilon**2)
    loss = torch.mean(error)
    return loss


def kboundary_loss_fn(LR_kspace, HR_kspace, factor, kbound_lower, kbound_outer_layer):
    bd_idx = LR_kspace.shape[-1]//2
    
    HR_kspace = HR_kspace/(factor**3)
    HR_kspace_abs = torch.abs(HR_kspace)
    LR_kspace_abs = torch.abs(LR_kspace)

    HR_shell_mean = get_3D_shell(HR_kspace_abs, bd_idx).mean()
    LR_shell_mean = get_3D_shell(LR_kspace_abs, bd_idx-1).mean()

    # CHANGED TO MSE, FROM L1 LOSS
    diff = torch.pow(HR_shell_mean - LR_shell_mean, 2)
    first_loss = diff if((HR_shell_mean > 0.99*LR_shell_mean) or (HR_shell_mean < kbound_lower*LR_shell_mean)) else 0
    loss = first_loss

    return loss, first_loss


def kspace_loss(z_pred, z_true, kspace_mask):
    z_diff = z_pred - z_true

    # Calculate the absolute value of the difference (real, imag) and square each element
    z_abs_sq = torch.square(torch.abs(z_diff))
    z_abs_sq = z_abs_sq*kspace_mask

    # Calculate the mean squared error (MSE) loss in complex number space
    mse_loss = torch.mean(z_abs_sq)
    return mse_loss



def volumetric_psnr(img_in, img_out):
    try:
        img_out = img_out.cpu().detach().numpy()
    except:
        pass

    img_in = img_in[5:-5,5:-5,5:-5]
    img_out = img_out[5:-5,5:-5,5:-5]
    pixel_num = img_in.shape[-1]

    _mse_ = ((img_out-img_in)**2).sum()/(pixel_num**3)
    _psnr = 20*np.log10(1) - 10*np.log10(_mse_)

    return _psnr


def TVLoss3D(image):
    # Calculate the total variation of the image
    tv = torch.sum(torch.abs(image[:, :, :, :, :-1] - image[:, :, :, :, 1:])) + \
         torch.sum(torch.abs(image[:, :, :, :-1, :] - image[:, :, :, 1:, :])) + \
         torch.sum(torch.abs(image[:, :, :-1, :, :] - image[:, :, 1:, :, :]))
    return tv/image.numel()


# In[ ]:





# In[ ]:


import numpy as np
from scipy.ndimage import sobel

def compute_gmsd(image1, image2):
    """
    Compute Gradient Magnitude Similarity Deviation (GMSD) for 3D images.
    
    Args:
    - image1: numpy array of shape (W, D, H)
    - image2: numpy array of shape (W, D, H)
    
    Returns:
    - GMSD value
    """
    # Compute gradients in each direction
    grad_x1 = sobel(image1, axis=0)
    grad_y1 = sobel(image1, axis=1)
    grad_z1 = sobel(image1, axis=2)
    
    grad_x2 = sobel(image2, axis=0)
    grad_y2 = sobel(image2, axis=1)
    grad_z2 = sobel(image2, axis=2)
    
    # Compute gradient magnitudes
    grad_mag1 = np.sqrt(grad_x1**2 + grad_y1**2 + grad_z1**2)
    grad_mag2 = np.sqrt(grad_x2**2 + grad_y2**2 + grad_z2**2)
    
    # Compute the similarity deviation (mean squared difference)
    diff = np.mean((grad_mag1 - grad_mag2)**2)
    
    return np.sqrt(diff)

