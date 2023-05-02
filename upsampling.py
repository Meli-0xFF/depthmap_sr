import math
import torch
import torch.nn as nn
from torch.nn.functional import *


class Upsampling_module(nn.Module):
  def __init__(self, upscale_factor=2):
    super(Upsampling_module, self).__init__()
    self.upscale_factor = upscale_factor
    self.feature_extraction = nn.Conv2d(in_channels=1,
                                        out_channels=1,
                                        kernel_size=5,
                                        padding=2)
    self.upsample = nn.ConvTranspose2d(in_channels=1,
                                       out_channels=1,
                                       kernel_size=3,
                                       stride=2,
                                       padding=1)
    self.act = nn.ReLU()

  def forward(self, x):
    s = [value * self.upscale_factor for value in list(x.shape)]
    x = self.act(self.feature_extraction(x))
    x = self.act(self.upsample(x, output_size=s))
    return x