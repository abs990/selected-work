import torch.nn as nn

class twoConvBlock(nn.Module):
  def __init__(self,input_channels: int, output_channels: int , kernel: tuple[int, int]):
    super(twoConvBlock, self).__init__()
    self.layers = nn.Sequential(
      nn.Conv2d(in_channels=input_channels,out_channels=output_channels,kernel_size=kernel),
      nn.ReLU(),
      nn.Conv2d(in_channels=output_channels,out_channels=output_channels,kernel_size=kernel),
      nn.BatchNorm2d(num_features=output_channels),
      nn.ReLU()
    )

  def forward(self, x):
    return self.layers(x)