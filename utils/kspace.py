#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch
import torch.fft

def to_k_space(arr):
    try:
        img_tensor = torch.from_numpy(arr)
    except:
        img_tensor = arr
    
    transformed = torch.fft.fftn(img_tensor)
    fshift = torch.fft.fftshift(transformed)
    return fshift

def inv_fft(kspace_arr):
    try:
        img_kspace_tensor = torch.from_numpy(kspace_arr)
    except:
        img_kspace_tensor = kspace_arr
    f_ishift = torch.fft.ifftshift(img_kspace_tensor)
    img_back = torch.fft.ifftn(f_ishift)
    img_back = torch.abs(img_back)
    return img_back


def get_kspaceImg(arr, adjust=1):
    kspace = to_k_space(arr)
    return 20*torch.log(torch.abs((kspace+1)*adjust))


def get_3D_shell(arr, half_size_of_the_shell):
    if arr.shape[-1]%2 !=0:
        raise ValueError("Shape error")

    c = arr.shape[-1]//2
    half_size = half_size_of_the_shell

    #compute
    # c-half_size; role: touch the walls
    # c-half_size:c+half_size; role: span
    # c-half_size+1:c+half_size-1; role of +1 & -1: avoid counting the same elements

    left = arr[c-half_size, c-half_size:c+half_size, c-half_size:c+half_size]
    right = arr[c+half_size-1, c-half_size:c+half_size, c-half_size:c+half_size]

    up = arr[c-half_size+1:c+half_size-1, c+half_size-1, c-half_size:c+half_size]
    down = arr[c-half_size+1:c+half_size-1, c+half_size-1, c-half_size:c+half_size]

    front = arr[c-half_size+1:c+half_size-1, c-half_size+1:c+half_size-1, c-half_size+1:c+half_size-1]
    back = arr[c-half_size+1:c+half_size-1, c-half_size+1:c+half_size-1, c-half_size+1:c+half_size-1]

    #sub inner rings
    all_shell_values = torch.cat([left.ravel(), right.ravel(),
                                  up.ravel(), down.ravel(),
                                  front.ravel(), back.ravel()])
    return all_shell_values

def get_all_3D_shell_values(arr, shell_range=None):
    kspace_shell_values = []

    if shell_range==None:
        shell_range = range(1, arr.shape[-1]//2+1)

    for half_size in shell_range:
        cubic_shell_values = get_3D_shell(arr, half_size)
        kspace_shell_values.append(cubic_shell_values)
    return kspace_shell_values


def fill_2D_shell(arr, half_size_of_the_shell, fill_value):
    if arr.shape[-1]%2 !=0:
        raise ValueError("Shape error")

    c = arr.shape[-1]//2
    half_size = half_size_of_the_shell

    #compute
    arr[c-half_size:c+half_size, c-half_size] = fill_value #right
    arr[c-half_size:c+half_size, c+half_size-1] = fill_value #left
    arr[c-half_size, (c-half_size+1):(c+half_size-1)] = fill_value #up
    arr[c+half_size-1, (c-half_size+1):(c+half_size-1)] = fill_value #down
    return arr

def fill_3D_shell(arr, half_size_of_the_shell, fill_value):
    if arr.shape[-1]%2 !=0:
        raise ValueError("Shape error")

    c = arr.shape[-1]//2
    half_size = half_size_of_the_shell

    #compute
    #side_wall_left
    arr[c-half_size, c-half_size:c+half_size, c-half_size:c+half_size] = fill_value
    #side_wall_right
    arr[c+half_size-1, c-half_size:c+half_size, c-half_size:c+half_size] = fill_value

    #sub inner rings
    sub_start, sub_end = c-half_size+1, c+half_size-1
    if sub_start ==sub_end:
        pass
    else:
        for sub in range(sub_start, sub_end):
            arr[sub, :, :] = fill_2D_shell(arr[sub, :, :], half_size, fill_value)
    return arr


# In[ ]:


def central_crop_3D(kspace_arr, factor):
    try:
        img_tensor = torch.from_numpy(kspace_arr)
    except:
        img_tensor = kspace_arr

    half_boxsize = img_tensor.shape[-1]//(factor*2) #half_boxsize
    half_boxsize = int(half_boxsize) # For if factor is not power of 2

    c = img_tensor.shape[-1]//2
    start = c-half_boxsize
    end = c+half_boxsize
    return img_tensor[start:end, start:end, start:end]


def torch_sinc_downsampler_3D(arr, factor=2):
    try:
        img_tensor = torch.from_numpy(arr)
    except:
        img_tensor = arr

    img_tensor_fft = to_k_space(img_tensor)
    img_tensor_fft_center = central_crop_3D(img_tensor_fft, factor)
    img_back = inv_fft(img_tensor_fft_center)/(factor**3)
    return img_back


import torch.nn.functional as F
def sinc_upsampler(tensor, factor):
    lr_size = tensor.shape[0]
    hr_size = int(lr_size*factor) #192 in our case
    p_size = (hr_size - lr_size)//2 # padding size 
    
    tensor = torch.clone(tensor)
    lr_ksapce = to_k_space(tensor)
    hr_ksapce = lr_ksapce*(factor**3)
    
    padded_hr_ksapce = F.pad(hr_ksapce, (p_size,p_size,p_size,p_size,p_size,p_size))
    upsampled_tensor = inv_fft(padded_hr_ksapce)
    upsampled_tensor = torch.clamp(upsampled_tensor, 0, 1)
    return upsampled_tensor


def central_replacement_3d(hr_img, dip_img, factor=2):
    hr_img_kspace = to_k_space(hr_img)
    dip_img_kspace = to_k_space(dip_img)
    hr_img_kspace_center = central_crop_3D(hr_img_kspace, factor)

    hr_size = dip_img_kspace.shape[-1]  
    half_boxsize = hr_size//(factor*2) #half_boxsize
    half_boxsize = int(half_boxsize) # For if factor is not power of 2
    c = hr_size//2
    start = c-half_boxsize
    end = c+half_boxsize

    dip_img_kspace_replaced = torch.clone(dip_img_kspace)   
    dip_img_kspace_replaced[start:end, start:end, start:end] = hr_img_kspace_center

    dip_img_central_replacement = inv_fft(dip_img_kspace_replaced).numpy()
    return dip_img_central_replacement


# In[ ]:


import numpy as np
import torch

def gen_ksapce_mask(shape, size, kspace_mask_weight=1.):
    if shape.lower()=='v':
        kspace_mask = torch.zeros((size,size,size))
        for half_size in range(1, size//2+1):
            if half_size ==size:
                value = half_size//2
            else:
                value = half_size
            kspace_mask = fill_3D_shell(kspace_mask, half_size, value)

        kspace_mask = kspace_mask/kspace_mask.max()
    
    elif shape.lower()=='u':
        kspace_mask = torch.zeros((size,size,size))
        mag = np.linspace(0,5,size//2)
        kweight = np.power(2, mag)  
        for half_size in range(1, size//2+1):
            if half_size ==size:
                value = kweight[half_size-1]/2
            else:
                value = kweight[half_size-1]
            kspace_mask = fill_3D_shell(kspace_mask, half_size, value)

        kspace_mask = kspace_mask/kspace_mask.max()
        
    elif shape.lower()=='i':
        kspace_mask = torch.ones((size,size,size))*kspace_mask_weight
    return kspace_mask

