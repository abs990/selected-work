from typing import List
import torch
import torch.nn as nn
from two_conv_block import twoConvBlock

class downStep(nn.Module):
  def __init__(self):
    super(downStep, self).__init__()
    block_conv_kernel_size = (3,3)
    pool_kernel_size = (2,2)
    pool_stride = (2,2)
    self.layer1 = twoConvBlock(input_channels=1, output_channels=64, kernel=block_conv_kernel_size)
    self.layer2 = nn.Sequential(
      nn.MaxPool2d(kernel_size=pool_kernel_size, stride=pool_stride),
      twoConvBlock(input_channels=64, output_channels=128, kernel=block_conv_kernel_size)
    )
    self.layer3 = nn.Sequential(
      nn.MaxPool2d(kernel_size=pool_kernel_size, stride=pool_stride),
      twoConvBlock(input_channels=128, output_channels=256, kernel=block_conv_kernel_size)      
    )
    self.layer4 = nn.Sequential(
      nn.MaxPool2d(kernel_size=pool_kernel_size, stride=pool_stride),
      twoConvBlock(input_channels=256, output_channels=512, kernel=block_conv_kernel_size)      
    )
    self.layer5 = nn.Sequential(
      nn.MaxPool2d(kernel_size=pool_kernel_size, stride=pool_stride),
      twoConvBlock(input_channels=512, output_channels=1024, kernel=block_conv_kernel_size)      
    )

  def forward(self, x):
    conv_output = []
    x = self.layer1(x)
    conv_output.append(x)
    x = self.layer2(x)
    conv_output.append(x)
    x = self.layer3(x)
    conv_output.append(x)
    x = self.layer4(x)
    conv_output.append(x)
    x = self.layer5(x)
    conv_output.append(x)    
    return conv_output 