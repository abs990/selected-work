import torch.nn as nn
from down_step import downStep
from up_step import upStep

class UNet(nn.Module):
  def __init__(self):
    super(UNet, self).__init__()
    self.down_step_block = downStep()
    self.up_step_block = upStep()

  def forward(self, x):
    return self.up_step_block(self.down_step_block(x))