import torch
from torch.nn.functional import *
from models_utils import *


class FSDR_Net(nn.Module):
  def __init__(self, num_feats, kernel_size):
    super(FSDR_Net, self).__init__()

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