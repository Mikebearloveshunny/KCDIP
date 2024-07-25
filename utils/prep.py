#!/usr/bin/env python
# coding: utf-8

# In[ ]:

import torch
import torch.nn as nn
import torchvision
import sys

import numpy as np
from PIL import Image
import PIL
import numpy as np

import matplotlib.pyplot as plt


import torch.fft
import torchvision.transforms as transforms
import torchvision

def write_log(msg, filename):
    print(msg)
    with open(filename, 'a') as file:
        file.write(msg + '\n')


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


def nor(arr):
    norm = (arr-arr.min())/(arr.max()-arr.min())
    return norm


def torch2np(tensor):
    return tensor.detach().cpu().numpy()



def pil_to_np(img_PIL):
    '''Converts image in PIL format to np.array.
    
    From W x H x C [0...255] to C x W x H [0..1]
    '''
    ar = np.array(img_PIL)

    if len(ar.shape) == 3:
        ar = ar.transpose(2,0,1)
    else:
        ar = ar[None, ...]

    return ar.astype(np.float32) / 255.

def np_to_pil(img_np): 
    '''Converts image in np.array format to PIL image.
    
    From C x W x H [0..1] to  W x H x C [0...255]
    '''
    ar = np.clip(img_np*255,0,255).astype(np.uint8)
    
    if img_np.shape[0] == 1:
        ar = ar[0]
    else:
        ar = ar.transpose(1, 2, 0)

    return Image.fromarray(ar)

def np_to_torch(img_np):
    '''Converts image in numpy.array to torch.Tensor.

    From C x W x H [0..1] to  C x W x H [0..1]
    '''
    return torch.from_numpy(img_np)[None, :]

def torch_to_np(img_var):
    '''Converts an image in torch.Tensor format to np.array.

    From 1 x C x W x H [0..1] to  C x W x H [0..1]
    '''
    return img_var.detach().cpu().numpy()[0]