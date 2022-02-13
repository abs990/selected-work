from typing import List
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from two_conv_block import twoConvBlock

class upStep(nn.Module):
  def __init__(self):
    super(upStep, self).__init__()
    up_conv_kernel_size = (2,2)
    block_conv_kernel_size = (3,3)
    up_conv_stride = 2
    self.layer1 = nn.ConvTranspose2d(in_channels=1024, out_channels=512, kernel_size=up_conv_kernel_size, stride=up_conv_stride)
    self.layer2 = nn.Sequential(
      twoConvBlock(input_channels=1024, output_channels=512, kernel=block_conv_kernel_size),
      nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=up_conv_kernel_size, stride=up_conv_stride)
    )
    self.layer3 = nn.Sequential(
      twoConvBlock(input_channels=512, output_channels=256, kernel=block_conv_kernel_size),
      nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=up_conv_kernel_size, stride=up_conv_stride)
    )
    self.layer4 = nn.Sequential(
      twoConvBlock(input_channels=256, output_channels=128, kernel=block_conv_kernel_size),
      nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=up_conv_kernel_size, stride=up_conv_stride)
    )
    self.layer5 = nn.Sequential(
      twoConvBlock(input_channels=128, output_channels=64, kernel=block_conv_kernel_size)
      #nn.Conv1d(in_channels=64,out_channels=2,kernel_size=(1,1))
      ,nn.Conv2d(in_channels=64,out_channels=2,kernel_size=(1,1))
      #,nn.Sigmoid()
    )
    self.crop1 = transforms.CenterCrop(56)
    self.crop2 = transforms.CenterCrop(104)
    self.crop3 = transforms.CenterCrop(200)
    self.crop4 = transforms.CenterCrop(392)

  def forward(self, down_step_feature_maps: List):
    down_step_feature_maps.reverse()
    result = self.layer1(down_step_feature_maps[0])
    down_1 = self.crop1(down_step_feature_maps[1])
    result = torch.cat((result, down_1),1)
    result = self.layer2(result)
    down_2 = self.crop2(down_step_feature_maps[2])
    result = torch.cat((result, down_2),1)
    result = self.layer3(result)
    down_3 = self.crop3(down_step_feature_maps[3])
    result = torch.cat((result, down_3),1)
    result = self.layer4(result)
    down_4 = self.crop4(down_step_feature_maps[4])
    result = torch.cat((result, down_4),1)
    result = self.layer5(result)
    return result 