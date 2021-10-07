import torch
import torch.nn as nn
import numpy as np

class ResnetBlock(nn.Module):
    def __init__(self, n_in):
        super(ResnetBlock, self).__init__()
        n_out = n_in
        self.res_block = nn.Sequential(
                            nn.ReflectionPad2d(1),
                            nn.Conv2d(n_in, n_out, 3),
                            nn.ReflectionPad2d(1),
                            nn.Conv2d(n_in, n_out, 3)
                        )

    def forward(self, x):
        out = x + self.res_block(x)

        return out

class ConvBlock(nn.Module):
    def __init__(self, n_in, n_out):
        super(ConvBlock, self).__init__()
        self.conv_block = nn.Sequential(
                            nn.ReflectionPad2d(1),
                            nn.Conv2d(n_in, n_out, 3),
                            nn.ReLU()
                        )

    def forward(self, x):
        out = self.conv_block(x)

        return out

def adain(a_f, b_f):
    size = a_f.shape
    b_mean, b_std = calc_mean_std(b_f)
    a_mean, a_std = calc_mean_std(a_f)

    normalized_f = (a_f - a_mean.expand(size)) / a_std.expand(size)

    return normalized_f * b_std.expand(size) + b_mean.expand(size)

def calc_mean_std(feat, eps=1e-5):
    size = feat.size()
    assert (len(size) == 4)
    N, C = size[:2]
    feat_var = feat.view(N, C, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().view(N, C, 1, 1)
    feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
    return feat_mean, feat_std
