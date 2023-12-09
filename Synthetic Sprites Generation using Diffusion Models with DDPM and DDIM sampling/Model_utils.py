import torch
import torch.nn as nn
import numpy as np
from torchvision.utils import save_image, make_grid
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from PIL import Image
import os

class ResidualConvBlock(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int, is_res: bool = False 
    ) -> None:
        super().__init__()
        
        # Check if input and output are the same for the residual connection 
        self.same_channels = in_channels == out_channels
        
        # Flag for where or not to use residual connection
        self.is_res = is_res
        
        # conv layers
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, 1, 1),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.is_res:
            x1 = self.conv1(x)
            x2 = self.conv2(x1)
            
            # if input and output channels are the same, add residual connectinos directly
            if self.same_channels:
                out = x + x2   # we are just concatenation the blocks 
            else:
                # if not, apply a 1x1 conv layer to match the dim before adding residual connection 
                shortcut = nn.Conv2d(x.shape[1], x2.shape[1], kernel_size=1, stride=1, padding=0).to(x.device)
                out = shortcut(x) + x2 
                # out = f(input) + input 

            return out / 1.414 # normalize the output tensor 

        else: # if not residual, just adding the blocks
            x1 = self.conv1(x)
            x2 = self.conv2(x)
            return x2
    
    # to get the number of output channels for this block 
    def get_out_channels(self):
        return self.conv2[0].out_channels
    
    # to set the number of output channels for this block 
    def set_out_channels(self, out_channels):
        self.conv1[0].out_channels = out_channels
        self.conv2[0].in_channels = out_channels
        self.conv2[0].out_channels = out_channels
    

class UnetUp(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UnetUp, self).__init__()
        
        # list of layers for the upsampling block 
        layers = [
            nn.ConvTranspose2d(in_channels, out_channels, 2, 2),
            ResidualConvBlock(out_channels, out_channels),
            ResidualConvBlock(out_channels, out_channels),
        ]
        
        self.model = nn.Sequential(*layers)
    
    def forward(self, x, skip):
        # concatenate the input tensor x with the skip connection tensor along the channel dimentsion 
        x = torch.cat((x, skip), 1)
        x = self.model(x)
        return x

class UnetDown(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UnetDown, self).__init__()
        
        layers = [
            ResidualConvBlock(in_channels, out_channels), 
            ResidualConvBlock(out_channels, out_channels),
            nn.MaxPool2d(kernel_size=2)
        ]
        
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.model(x)
    
# one-layer feed-forward nn for embedding input data of 
# dimensionality input dim to an embedding space of dimensionality emb dim 
class EmbedFC(nn.Module):
    def __init__(self, input_dim, emb_dim):
        super(EmbedFC, self).__init__()
        
        layers = [
            nn.Linear(input_dim, emb_dim),
            nn.GELU(),
            nn.Linear(emb_dim, emb_dim)
        ]
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        # flatten the input tensor 
        x = x.view(-1, self.input_dim)
        return self.model(x)
    

