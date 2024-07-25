#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch
import torch.nn as nn
import torch.nn.functional as F

# (v) batchNorm -> InstanceNorm #nn.BatchNorm3d(out_channels), #InstanceNorm
# residual connection
# ReLu -> LeakyReLu

from prep import *
from sr_common import *

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, conv_residual=False, stride=1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1),
            nn.InstanceNorm3d(out_channels), # (v) batchNorm -> InstanceNorm
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.InstanceNorm3d(out_channels), # (v) batchNorm -> InstanceNorm
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class Down(nn.Module):
    def __init__(self, in_channels, out_channels, conv_residual=False):
        super().__init__()
        self.mpconv = nn.Sequential(
            nn.MaxPool3d(2),     #use trilinear pooling
            DoubleConv(in_channels, out_channels, conv_residual)
        )

    def forward(self, x):
        return self.mpconv(x)

class Up(nn.Module):
    def __init__(self, in_channels, out_channels, conv_residual=False, trilinear=False):
        super().__init__()
        if trilinear:
            self.up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose3d(in_channels//2, in_channels//2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels, conv_residual)

    def forward(self, x1, x2):
        #x1 is from the mainstreaml; x2 is from the skip connection
        x1 = self.up(x1)
        '''
        diffZ = x2.size()[2] - x1.size()[2]
        diffY = x2.size()[3] - x1.size()[3]
        diffX = x2.size()[4] - x1.size()[4]
        x1 = nn.functional.pad(x1, (diffX // 2, diffX - diffX // 2,
                                    diffY // 2, diffY - diffY // 2,
                                    diffZ // 2, diffZ - diffZ // 2))
        '''
        # Calculate padding dynamically based on the size difference between x2 and x1
        padding_dims = [0, 0, 0, 0, 0, 0]  # Initialize padding dimensions
        for dim in range(3):
            size_diff = x2.size(dim+2) - x1.size(dim+2)
            if size_diff % 2 == 0:
                # If the size difference is even, split it evenly on both sides
                padding_dims[2 * dim] = size_diff // 2
                padding_dims[2 * dim + 1] = size_diff // 2
            else:
                # If the size difference is odd, add the extra pixel to the end
                padding_dims[2 * dim] = size_diff // 2
                padding_dims[2 * dim + 1] = size_diff // 2 + 1

        # Apply padding to x1
        x1 = nn.functional.pad(x1, tuple(padding_dims))
        
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class UNet3D(nn.Module):
    def __init__(self, in_channels, out_channels, filter_n = 64, trilinear=False, conv_residual=False):
        super().__init__()
        self.inc = DoubleConv(in_channels, filter_n)
        self.down1 = Down(filter_n, filter_n*2)
        self.down2 = Down(filter_n*2, filter_n*4)
        self.down3 = Down(filter_n*4, filter_n*8)
        self.down4 = Down(filter_n*8, filter_n*8)
        self.up1 = Up(filter_n*16, filter_n*4, trilinear=trilinear)
        self.up2 = Up(filter_n*8, filter_n*2, trilinear=trilinear)
        self.up3 = Up(filter_n*4, filter_n, trilinear=trilinear)
        self.up4 = Up(filter_n*2, filter_n, trilinear=trilinear)
        self.out = nn.Conv3d(filter_n, out_channels, kernel_size=1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        # (v) Add a few layers (merge the features previously learned)
        out = self.out(x)
        out = torch.sigmoid(out)  # apply sigmoid activation function
        return out


# In[ ]:

class asym_UNet3D(nn.Module):
    def __init__(self, in_channels, out_channels, factor, trilinear=False, filter_n = 64):
        super().__init__()
        self.inc = DoubleConv(in_channels, filter_n)
        self.down1 = Down(filter_n, filter_n*2)
        self.down2 = Down(filter_n*2, filter_n*4)
        self.down3 = Down(filter_n*4, filter_n*8)
        self.down4 = Down(filter_n*8, filter_n*8)
        self.up1 = Up(filter_n*16, filter_n*4, trilinear=trilinear)
        self.up2 = Up(filter_n*8, filter_n*2, trilinear=trilinear)
        self.up3 = Up(filter_n*4, filter_n, trilinear=trilinear)
        self.up4 = Up(filter_n*2, filter_n, trilinear=trilinear)
        
        self.final_up = nn.Upsample(scale_factor=factor, mode='trilinear', align_corners=True)
        self.out = nn.Conv3d(filter_n, out_channels, kernel_size=1)
#         self.out = DoubleConv(filter_n, out_channels)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        # (v) Add a few layers (merge the features previously learned)
        x = self.final_up(x)
        out = self.out(x)
        out = torch.sigmoid(out)  # apply sigmoid activation function
        return out


def create_highpass_filter(kernel_size, normalize=False):
    # Create a 3D high-pass filter
    highpass_filter = torch.ones(1, 1, kernel_size, kernel_size, kernel_size)
    center_pixel = kernel_size // 2
    key_value = kernel_size ** 3 - 1
    highpass_filter[0, 0, center_pixel, center_pixel, center_pixel] = -(key_value)
    if normalize:
        return highpass_filter/key_value
    else:
        return highpass_filter

# # Example usage
# kernel_size = 3
# highpass_filter = create_highpass_filter(kernel_size)


class SharpBlock(nn.Module):
    def __init__(self, in_channels, kernel_size):
        super(SharpBlock, self).__init__()
        # kernel = torch.Tensor(
        #     [[[-1, -1, -1], [-1,  -1, -1],[-1, -1, -1]],
        #      [[-1, -1, -1], [-1,  8, -1],[-1, -1, -1]],
        #       [[-1, -1, -1], [-1,  -1, -1],[-1, -1, -1]]])

        kernel = create_highpass_filter(kernel_size)

        #self.weight = nn.Parameter(data=kernel.unsqueeze(0).repeat(in_channels, 1, 1, 1), requires_grad=False)
        #self.weight = nn.Parameter(data=kernel.unsqueeze(0).unsqueeze(1).repeat(in_channels, 1, 1, 1, 1), requires_grad=False)
        self.weight = nn.Parameter(data=kernel.repeat(in_channels, 1, 1, 1, 1), requires_grad=False)
        #self.depthwise_conv = nn.Conv3d(in_channels, in_channels, kernel_size=3, stride=1, padding=1, groups=in_channels, bias=False)
        self.depthwise_conv = nn.Conv3d(in_channels, in_channels, kernel_size=kernel_size, stride=1, padding='same', groups=in_channels, bias=False)
        self.depthwise_conv.weight = self.weight

    def forward(self, x):
        x = self.depthwise_conv(x)
        return x

class SharpUNet3D(nn.Module):
    def __init__(self, in_channels, out_channels, filter_n = 64, trilinear=False, conv_residual=False):
        super().__init__()
        self.inc = DoubleConv(in_channels, filter_n)
        self.down1 = Down(filter_n, filter_n*2)
        self.down2 = Down(filter_n*2, filter_n*4)
        self.down3 = Down(filter_n*4, filter_n*8)
        self.down4 = Down(filter_n*8, filter_n*8)

        self.sb1 = SharpBlock(filter_n, 3) #in_channels, kernel_size
        self.sb2 = SharpBlock(filter_n*2, 5)
        self.sb3 = SharpBlock(filter_n*4, 7)
        self.sb4 = SharpBlock(filter_n*8, 9)

        self.up1 = Up(filter_n*16, filter_n*4, trilinear=trilinear)
        self.up2 = Up(filter_n*8, filter_n*2, trilinear=trilinear)
        self.up3 = Up(filter_n*4, filter_n, trilinear=trilinear)
        self.up4 = Up(filter_n*2, filter_n, trilinear=trilinear)
        self.out = nn.Conv3d(filter_n, out_channels, kernel_size=1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x4 = self.sb4(x4)
        x3 = self.sb3(x3)
        x2 = self.sb2(x2)
        x1 = self.sb1(x1)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        # (v) Add a few layers (merge the features previously learned)
        out = self.out(x)
        out = torch.sigmoid(out)  # apply sigmoid activation function
        return out


# In[ ]:





# In[ ]:


def get_noise_3d(input_depth, method, noise_size, noise_type='u', var=1./10):
    """Returns a pytorch.Tensor of size (1 x `input_depth` x `spatial_size[0]` x `spatial_size[1]`)
    initialized in a specific way.
    Args:
        input_depth: number of channels in the tensor
        method: `noise` for fillting tensor with noise; `meshgrid` for np.meshgrid
        spatial_size: spatial size of the tensor to initialize
        noise_type: 'u' for uniform; 'n' for normal
        var: a factor, a noise will be multiplicated by. Basically it is standard deviation scaler.
    """
    if method == 'noise':
        shape = [1, input_depth, noise_size[0], noise_size[1], noise_size[2]]
        net_input = torch.zeros(shape)

        fill_noise(net_input, noise_type)
        net_input *= var
    return net_input


def central_crop_3D(kspace_arr, factor):
    try:
        img_tensor = torch.from_numpy(kspace_arr)
    except:
        img_tensor = kspace_arr

    half_boxsize = img_tensor.shape[-1]//(factor*2) #half_boxsize
    half_boxsize = int(half_boxsize) # For if factor is not power of 2
    # print(img_tensor.shape, img_tensor.shape[-1]/factor/2)

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


def sinc_upsampler(tensor, hr_size, factor):
    tensor = torch.clone(tensor)
    lr_ksapce = to_k_space(tensor)
    hr_ksapce = lr_ksapce*(factor**3)
    
    lr_size = tensor.shape[0]
    p_size = (hr_size - lr_size)//2 # two sides

    padded_hr_ksapce = F.pad(hr_ksapce, (p_size,p_size,p_size,p_size,p_size,p_size))
    upsampled_tensor = inv_fft(padded_hr_ksapce)
    upsampled_tensor = torch.clamp(upsampled_tensor, 0, 1)
    
    return upsampled_tensor
# In[ ]:


from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim
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

def psnr_3D(img_in, img_out):
    img_out = img_out.cpu().detach().numpy()
    if len(img_out.shape) !=3:
        raise ValueError("Dimension is wrong. img_out.shape=", img_out.shape)

    psnr_arr = []
    for i in range(img_in.shape[-1]):
        psnr_arr.append(compare_psnr(img_in[:,i,:], img_out[:,i,:]))
    return np.array(psnr_arr)


def ssim_loss(y_true, y_pred, max_val=1.0):
    # Convert 3D objects to tensor
    y_true = torch.tensor(y_true, dtype=torch.float32)
    y_pred = torch.tensor(y_pred, dtype=torch.float32)

    # Calculate mean and variance for y_true and y_pred
    mu1 = torch.mean(y_true, dim=[1, 2, 3])
    mu2 = torch.mean(y_pred, dim=[1, 2, 3])
    sigma1 = torch.var(y_true, dim=[1, 2, 3])
    sigma2 = torch.var(y_pred, dim=[1, 2, 3])
    sigma12 = torch.mean((y_true - mu1) * (y_pred - mu2), dim=[1, 2, 3])

    # Define constants for SSIM calculation
    c1 = (0.01 * max_val) ** 2
    c2 = (0.03 * max_val) ** 2
    ssim = (2 * mu1 * mu2 + c1) * (2 * sigma12 + c2) / ((mu1 ** 2 + mu2 ** 2 + c1) * (sigma1 + sigma2 + c2))
    return torch.mean(1 - ssim)


def TVLoss3D(image):
    # Calculate the total variation of the image
    tv = torch.sum(torch.abs(image[:, :, :, :, :-1] - image[:, :, :, :, 1:])) +          torch.sum(torch.abs(image[:, :, :, :-1, :] - image[:, :, :, 1:, :])) +          torch.sum(torch.abs(image[:, :, :-1, :, :] - image[:, :, 1:, :, :]))
    #return tv
    return tv/image.numel()


# In[ ]:
import pickle
from collections import defaultdict
def save_info_dict(item_list, info_path):
    output_line = ""
    
    try:
        with open(f"{info_path}.pkl", 'rb') as pickle_file:
            info_dict = pickle.load(pickle_file)
    except:
        info_dict = defaultdict(list)
    
    for item_name, value in item_list:
        item_info = item_name%value
        info_dict[item_info.split(" ")[0]].append(item_info.split(" ")[1])
        output_line += item_info
        output_line += "  "
        
    with open(f"{info_path}.pkl", "wb") as pickle_file:
        pickle.dump(info_dict, pickle_file)
    
    return output_line



def print_and_save(text, output_file):
    with open(output_file, "a") as file:
        print(text)
        file.write(text + "\n")

