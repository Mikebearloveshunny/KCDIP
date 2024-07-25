#!/usr/bin/env python
# coding: utf-8

# In[ ]:


def get_kspaceImg(arr, adjust=1):
    kspace = to_k_space(arr)
    return 20*np.log(np.abs(kspace*adjust)).numpy()


import numpy as np
def get_2D_shell(arr, half_size_of_the_shell):
    arr = np.array(arr)
    if arr.shape[-1]%2 !=0:
        raise ValueError("Shape error")
    
    c = arr.shape[-1]//2
    half_size = half_size_of_the_shell
    
    #compute
    right = arr[c-half_size:c+half_size, c-half_size]
    left = arr[c-half_size:c+half_size, c+half_size-1]
    up = arr[c-half_size, (c-half_size+1):(c+half_size-1)]
    down = arr[c+half_size-1, (c-half_size+1):(c+half_size-1)]

    layer_values = np.concatenate([up, down, left, right])
    #print(up, down, left, right)
    #print(layer_values.shape, layer_values.mean())
    return layer_values


def get_3D_shell(arr, half_size_of_the_shell):
    arr = np.array(arr)
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
        sub_shell_values = np.concatenate(sub_shell_values)
    else:
        sub_shell_values = np.array([])
    all_shell_values = np.concatenate([sub_shell_values, side_wall_left.ravel(), side_wall_right.ravel()])
    #print(all_shell_values.shape, all_shell_values.mean())
    #print(all_shell_values)
    return all_shell_values

from tqdm import tqdm
def get_all_3D_shell_values(arr):
    kspace_shell_values = []
    pbar = tqdm(range(1, arr.shape[-1]//2+1))
    for half_size in pbar:
        cubic_shell_values = get_3D_shell(arr, half_size)
        kspace_shell_values.append(cubic_shell_values)
    return kspace_shell_values


# In[ ]:


def fill_2D_shell(arr, half_size_of_the_shell, fill_value):
    arr = np.array(arr)
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
    arr = np.array(arr)
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

