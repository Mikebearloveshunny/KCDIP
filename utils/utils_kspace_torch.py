#!/usr/bin/env python
# coding: utf-8

# In[ ]:


def get_kspaceImg(arr, adjust=1):
    kspace = to_k_space(arr)
    return 20*torch.log(torch.abs((kspace+1)*adjust))

def get_2D_shell(arr, half_size_of_the_shell):
    if arr.shape[-1]%2 !=0:
        raise ValueError("Shape error")

    c = arr.shape[-1]//2
    half_size = half_size_of_the_shell

    #compute
    right = arr[c-half_size:c+half_size, c-half_size]
    left = arr[c-half_size:c+half_size, c+half_size-1]
    up = arr[c-half_size, (c-half_size+1):(c+half_size-1)]
    down = arr[c+half_size-1, (c-half_size+1):(c+half_size-1)]

    layer_values = torch.cat([up, down, left, right])
    #print(up, down, left, right)
    #print(layer_values.shape, layer_values.mean())
    return layer_values

#new version speed: 6.09 ms ± 83.7 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)
#old version speed: 90.5 ms ± 1.02 ms per loop (mean ± std. dev. of 7 runs, 10 loops each)
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

from tqdm import tqdm
def get_all_3D_shell_values(arr, shell_range=None):
    kspace_shell_values = []

    if shell_range==None:
        shell_range = range(1, arr.shape[-1]//2+1)

    for half_size in shell_range:
        cubic_shell_values = get_3D_shell(arr, half_size)
        kspace_shell_values.append(cubic_shell_values)
    return kspace_shell_values


# In[ ]:


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


import torch
import torch.nn.functional as F

def complex_log(complex_number):
    magnitude = torch.abs(complex_number)
    log_magnitude = torch.log(magnitude+0.01)

    # Combine the real and imaginary parts with the logarithm of the magnitude
    log_complex_number = torch.complex(log_magnitude, torch.angle(complex_number))
    return log_complex_number


# In[ ]:





# In[ ]:


'''old version
def get_3D_shell(arr, half_size_of_the_shell):
    if arr.shape[-1]%2 !=0:
        raise ValueError("Shape error")

    c = arr.shape[-1]//2
    half_size = half_size_of_the_shell

    #compute
    side_wall_left = arr[c-half_size, c-half_size:c+half_size, c-half_size:c+half_size]
    side_wall_right = arr[c+half_size-1, c-half_size:c+half_size, c-half_size:c+half_size]

    #sub inner rings
    sub_start, sub_end = c-half_size+1, c+half_size-1
    sub_shell_values = []
    for sub in range(sub_start, sub_end):
        sub_arr = arr[sub, :, :]
        shell_values = get_2D_shell(sub_arr, half_size)
        sub_shell_values.append(shell_values)

    if sub_start !=sub_end:
        sub_shell_values = torch.cat(sub_shell_values)
    else:
        sub_shell_values = torch.tensor([]).cuda()
    all_shell_values = torch.cat([sub_shell_values, side_wall_left.ravel(), side_wall_right.ravel()])
    return all_shell_values
'''

