#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import torch
import numpy as np

def create_folder(path):
    if not os.path.exists(path):
        os.makedirs(path)
    else:
        print(f"Folder already exists at: {path}")


import pickle
from collections import defaultdict
def save_info_dict(item_list, info_path):    
    try:
        with open(f"{info_path}.pkl", 'rb') as pickle_file:
            info_dict = pickle.load(pickle_file)
    except:
        info_dict = defaultdict(list)
    
    output_line = ""
    for item_name, value in item_list:
        item_info = item_name%value
        info_dict[item_info.split(" ")[0]].append(item_info.split(" ")[1])
        output_line += item_info
        output_line += "  "
        
    with open(f"{info_path}.pkl", "wb") as pickle_file:
        pickle.dump(info_dict, pickle_file)
    
    return output_line

def write_log(msg, filename):
    print(msg)
    with open(filename, 'a') as file:
        file.write(msg + '\n')

        
# *****        
def nor(arr):
    norm = (arr-arr.min())/(arr.max()-arr.min())
    return norm

def torch2np(tensor):
    return tensor.detach().cpu().numpy()


def mean_std(arr, name='name'):
    arr = np.array(arr)
    print(name,":" ,arr.mean().round(2), "±", np.std(arr).round(2))
    

def proc_bcp_name(path):
    key = path.split("/")[-1]
    if "T1w" in key:
        mod = "T1w"
        name = key.split("_")[0]
        
    elif "T2w" in key:
        mod = "T2w"
        name = key.split("_")[1]
    else:
        raise ValueError("error!")
    
    name = "age" + name
    return name, mod

