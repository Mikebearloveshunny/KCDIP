#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch
import torch.nn as nn
import torch.nn.functional as F

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
        out = self.out(x)
        out = torch.sigmoid(out)  # apply sigmoid activation function
        return out

# for self-supervised
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
        x = self.final_up(x)
        out = self.out(x)
        out = torch.sigmoid(out)  # apply sigmoid activation function
        return out


def fill_noise(x, noise_type):
    """Fills tensor `x` with noise of type `noise_type`."""
    if noise_type == 'u':
        x.uniform_()
    elif noise_type == 'n':
        x.normal_() 
    else:
        assert False

        
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


def get_params(net):
    params = [x for x in net.parameters() ]            
    return params

