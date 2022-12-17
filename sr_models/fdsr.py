import math
import torch
import torch.nn as nn
from torch.nn.functional import *


class FDSR_Net(nn.Module):
  def __init__(self, num_feats, kernel_size):
    super(FDSR_Net, self).__init__()

    self.conv_tex1 = nn.Conv2d(in_channels=16, out_channels=num_feats,
                              kernel_size=kernel_size, padding=1)
    self.tex_cbl2 = Conv_BN_ACT(in_channels=num_feats, out_channels=num_feats, kernel_size=kernel_size,
                                   alpha_in=0, alpha_out=0.25,
                                   stride=1, padding=1, dilation=1, groups=1, norm_layer=nn.BatchNorm2d,
                                   activation_layer=nn.LeakyReLU())
    self.tex_cbl3 = Conv_BN_ACT(in_channels=num_feats, out_channels=num_feats, kernel_size=kernel_size,
                                   alpha_in=0.25, alpha_out=0.25,
                                   stride=1, padding=1, dilation=1, groups=1, norm_layer=nn.BatchNorm2d,
                                   activation_layer=nn.LeakyReLU())
    self.tex_cbl4 = Conv_BN_ACT(in_channels=num_feats, out_channels=num_feats, kernel_size=kernel_size,
                                   alpha_in=0.25, alpha_out=0.25,
                                   stride=1, padding=1, dilation=1, groups=1, norm_layer=nn.BatchNorm2d,
                                   activation_layer=nn.LeakyReLU())

    self.conv_dp1 = nn.Conv2d(in_channels=16, out_channels=num_feats,
                              kernel_size=kernel_size, padding=1)
    self.MSB1 = MS_RB(num_feats, kernel_size)
    self.MSB2 = MS_RB(56, kernel_size)
    self.MSB3 = MS_RB(80, kernel_size)
    self.MSB4 = MS_RB(104, kernel_size)
    self.conv_recon1 = nn.Conv2d(in_channels=104, out_channels=4 * num_feats, kernel_size=kernel_size, padding=1)
    self.conv_recon2 = nn.Conv2d(in_channels=num_feats, out_channels=4 * num_feats, kernel_size=kernel_size, padding=1)
    self.restore = nn.Conv2d(in_channels=num_feats, out_channels=1, kernel_size=kernel_size, padding=1)
    self.act = nn.LeakyReLU()

  def forward(self, x):
    image, depth = x
    re_im = pixel_unshuffle(image, 4)
    re_dp = pixel_unshuffle(depth, 4)
    dp_in = self.act(self.conv_dp1(re_dp))
    dp1 = self.MSB1(dp_in)
    tex1 = self.act(self.conv_tex1(re_im))
    tex2 = self.tex_cbl2(tex1)
    ca1_in = torch.cat((dp1, tex2[0]), dim=1)
    dp2 = self.MSB2(ca1_in)
    tex3 = self.tex_cbl3(tex2)
    ca2_in = torch.cat((dp2, tex3[0]), dim=1)
    dp3 = self.MSB3(ca2_in)
    tex4 = self.tex_cbl4(tex3)
    ca3_in = torch.cat((dp3, tex4[0]), dim=1)
    dp4 = self.MSB4(ca3_in)
    up1 = pixel_shuffle(self.conv_recon1(self.act(dp4)), 2)
    up2 = pixel_shuffle(self.conv_recon2(up1), 2)
    out = self.restore.forward(up2)
    out = depth + out
    return out


class OctaveConv(nn.Module):
  def __init__(self, in_channels, out_channels, kernel_size, alpha_in=0.5, alpha_out=0.5, stride=1, padding=1,
               dilation=1, groups=1):
    super(OctaveConv, self).__init__()
    self.downsample = nn.AvgPool2d(kernel_size=(2, 2), stride=2)

    assert stride == 1 or stride == 2, "Stride should be 1 or 2."
    self.stride = stride
    self.is_dw = groups == in_channels
    assert 0 <= alpha_in <= 1 and 0 <= alpha_out <= 1, "Alphas should be in the interval from 0 to 1."
    self.alpha_in, self.alpha_out = alpha_in, alpha_out
    self.conv_l2l = None if alpha_in == 0 or alpha_out == 0 else \
      nn.Conv2d(in_channels=int(alpha_in * in_channels), out_channels=int(alpha_out * out_channels),
                kernel_size=kernel_size, stride=1, padding=1, dilation=dilation,
                groups=int(math.ceil(alpha_in * groups)))
    self.conv_l2h = None if alpha_in == 0 or alpha_out == 1 or self.is_dw else \
      nn.Conv2d(int(alpha_in * in_channels), out_channels - int(alpha_out * out_channels),
                kernel_size, stride=1, padding=padding, dilation=dilation, groups=groups)
    self.conv_h2l = None if alpha_in == 1 or alpha_out == 0 or self.is_dw else \
      nn.Conv2d(in_channels - int(alpha_in * in_channels), int(alpha_out * out_channels),
                kernel_size, stride=1, padding=padding, dilation=dilation, groups=groups)
    self.conv_h2h = None if alpha_in == 1 or alpha_out == 1 else \
      nn.Conv2d(in_channels - int(alpha_in * in_channels), out_channels - int(alpha_out * out_channels),
                kernel_size, stride=1, padding=padding, dilation=dilation,
                groups=int(math.ceil(groups - alpha_in * groups)))

  def forward(self, x):
    x_h, x_l = x if type(x) is tuple else (x, None)

    x_h = self.downsample(x_h) if self.stride == 2 else x_h
    x_h2h = self.conv_h2h(x_h)
    x_h2l = self.conv_h2l(self.downsample(x_h)) if self.alpha_out > 0 and not self.is_dw else None
    if x_l is not None:
      x_l2l = self.downsample(x_l) if self.stride == 2 else x_l
      x_l2l = self.conv_l2l(x_l2l) if self.alpha_out > 0 else None
      if self.is_dw:
        return x_h2h, x_l2l
      else:
        x_l2h = self.conv_l2h(x_l)
        x_l2h = interpolate(x_l2h, size=(200, 140), mode='bilinear', align_corners=False) if self.stride == 1 else x_l2h
        x_h = x_l2h + x_h2h
        x_l = x_h2l + x_l2l if x_h2l is not None and x_l2l is not None else None
        return x_h, x_l
    else:
      return x_h2h, x_h2l


class Conv_BN(nn.Module):
  def __init__(self, in_channels, out_channels, kernel_size, alpha_in=0.5, alpha_out=0.5, stride=1, padding=0,
               dilation=1, groups=1, norm_layer=nn.BatchNorm2d):
    super(Conv_BN, self).__init__()
    self.conv = OctaveConv(in_channels, out_channels, kernel_size, alpha_in, alpha_out, stride, padding,
                           dilation, groups)
    self.bn_h = None if alpha_out == 1 else norm_layer(int(out_channels * (1 - alpha_out)))
    self.bn_l = None if alpha_out == 0 else norm_layer(int(out_channels * alpha_out))

  def forward(self, x):
    x_h, x_l = self.conv.forward(x)
    x_h = self.bn_h(x_h)
    x_l = self.bn_l(x_l) if x_l is not None else None
    return x_h, x_l


class Conv_BN_ACT(nn.Module):
  def __init__(self, in_channels, out_channels, kernel_size, alpha_in=0.5, alpha_out=0.5, stride=1, padding=1,
               dilation=1, groups=1, norm_layer=nn.BatchNorm2d, activation_layer=nn.ReLU):
    super(Conv_BN_ACT, self).__init__()
    self.conv = OctaveConv(in_channels, out_channels, kernel_size, alpha_in, alpha_out, stride, padding,
                           dilation, groups)

    self.act = nn.LeakyReLU()

  def forward(self, x):
    x_h, x_l = self.conv(x)

    x_h = self.act(x_h)
    x_l = self.act(x_l) if x_l is not None else None
    return x_h, x_l


class MS_RB(nn.Module):
  def __init__(self, num_feats, kernel_size):
    super(MS_RB, self).__init__()
    self.conv1 = nn.Conv2d(in_channels=num_feats, out_channels=num_feats,
                           kernel_size=kernel_size, padding=1, dilation=1)
    self.conv2 = nn.Conv2d(in_channels=num_feats, out_channels=num_feats,
                           kernel_size=kernel_size, padding=2, dilation=2)
    self.conv4 = nn.Conv2d(in_channels=num_feats, out_channels=num_feats,
                           kernel_size=1, padding=0)
    self.act = nn.LeakyReLU()

  def forward(self, x):
    x1 = self.act(self.conv1(x))
    x2 = self.act(self.conv2(x))
    x3 = x1 + x2
    x4 = self.conv4(x3)
    out = x4 + x

    return out