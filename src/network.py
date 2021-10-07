import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import numpy as np
import net_module as net
from torch.optim import lr_scheduler


###############################################################
#-------------------------Encoder-----------------------------#
###############################################################
class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        # content encoder
        vgg_net = models.vgg19()
        vgg_net.load_state_dict(torch.load("../models/vgg19.pth"))

        vgg_list = [
            nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True), #relu1_1
            nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True), #relu1-2
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
            nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True), #relu2_1
            nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),# relu2_2
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
            nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),#relu3_1
            nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),#relu_3_2
            nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),#relu3_3
            nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),#relu3_4
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
            nn.Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),#relu4_1
            nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),#relu4_2
            nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),#relu4_3
            nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),#relu4_4
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
            nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),#relu5_1
            nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),#relu5_2
            nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),#relu5_3
            nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),#relu5_4
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
            nn.AdaptiveAvgPool2d(output_size=(7, 7)),
            nn.Linear(in_features=25088, out_features=4096, bias=True),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5, inplace=False),
            nn.Linear(in_features=4096, out_features=4096, bias=True),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5, inplace=False),
            nn.Linear(in_features=4096, out_features=1000, bias=True)]
        

        self.enc_1 = nn.Sequential(*vgg_list[:2])  # relu1_1
        self.enc_2 = nn.Sequential(*vgg_list[2:7]) # relu2_1
        self.enc_3 = nn.Sequential(*vgg_list[7:12]) # relu3_1
        self.enc_4 = nn.Sequential(*vgg_list[12:21]) # relu4_1
        self.enc_5 = nn.Sequential(*vgg_list[21:30]) # relu5_1

    def forward(self, x):
        out = {}

        x = self.enc_1(x)
        out["r11"] = x
        x = self.enc_2(x)
        out["r21"] = x
        x = self.enc_3(x)
        out["r31"] = x
        x = self.enc_4(x)
        out["r41"] = x
        x = self.enc_5(x)
        out["r51"] = x
        
        return out


##############################################################
#-------------------------Decoder----------------------------#
##############################################################
class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()

        self.resblock_41 = net.ResnetBlock(512)
        self.convblock_41 = net.ConvBlock(512, 256)
        self.resblock_31 = net.ResnetBlock(256)
        self.convblock_31 = net.ConvBlock(256, 128)

        self.convblock_21 = net.ConvBlock(128, 128)
        self.convblock_22 = net.ConvBlock(128, 64)

        self.convblock_11 = net.ConvBlock(64, 64)
        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")

        self.final_conv = nn.Sequential(*[nn.ReflectionPad2d(1),nn.Conv2d(64, 3, 3)])


    def forward(self, a_f, b_f):
        out = net.adain(a_f["r41"], b_f["r41"])
        out = self.upsample(self.convblock_41(self.resblock_41(out)))

        out += net.adain(a_f["r31"], b_f["r31"])
        out = self.upsample(self.convblock_31(self.resblock_31(out)))

        out = net.adain(a_f["r21"], b_f["r21"])
        out = self.upsample(self.convblock_22(self.convblock_21(out)))

        out = self.final_conv(self.convblock_11(out))

        return out


##############################################################
#------------------------Generator---------------------------#
##############################################################
class ReviseNet(nn.Module):
    def __init__(self, n_in):
        super(ReviseNet, self).__init__()
        n_out = 64
        down_blocks = []
        down_blocks +=[
                nn.ReflectionPad2d(1),
                nn.Conv2d(n_in, n_out, 3),
                nn.ReLU()]

        n_in = n_out
        down_blocks +=[
                nn.ReflectionPad2d(1),
                nn.Conv2d(n_in, n_out, 3, 2),
                nn.ReLU()]

        self.res_block =net.ResnetBlock(64)

        up_blocks = []
        up_blocks += [
                nn.Upsample(scale_factor=2, mode="nearest"),
                nn.ReflectionPad2d(1),
                nn.Conv2d(n_in, n_out, 3),
                nn.ReLU()]

        n_in = 64
        n_out = 3
        up_blocks += [
                nn.ReflectionPad2d(1),
                nn.Conv2d(n_in, n_out, 3)]
        self.down_sample = nn.Sequential(*down_blocks)
        self.up_sample = nn.Sequential(*up_blocks)

    def forward(self, input):
        return self.up_sample(self.res_block(self.down_sample(input)))





##############################################################
#--------------------Discirminator---------------------------#
##############################################################

class Discirminator(nn.Module):
    def __init__(self):
        super()
        self.n_layer = 3
        n_in = 3
        n_out = 32

        self.head = nn.Sequential([
            nn.Conv2d(n_in, n_out, 3, 1, 1),
            nn.BatchNorm2d(n_out),
            nn.LeakyReLU(0.2)])
        body = []
        n_in = n_out
        for _ in range(self.n_layer - 2):
            self.body.append(nn.Conv2d(n_in, n_out, 3, 1, 1))
            self.body.append(nn.BatchNorm2d(n_out))
            self.body.append(nn.LeakyReLU(0.2))
        
        self.body = nn.Sequetial(*body)
        self.tail = nn.Conv2d(n_in, 1, 3, 1, 1)
       
    def forward(self, x):
        x = self.tail(self.body(self.head(x)))
        return x

###############################################################
#---------------------------Basic Functions-------------------#
###############################################################
def gaussian_weights_init(m):
    classname = m.__class__.__name__
    if hasattr(m, 'weight') and classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)

def get_scheduler(optimizer, opts, cur_ep=-1):
    if opts.lr_policy == "lambda":
        def lambda_rule(ep):
            lr_l = 1.0 - max(0, ep - opts.n_ep_decay) / float(opts.n_ep - opts.n_ep_decay + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule, last_epoch=cur_ep)
    elif opts.lr_policy == "step":
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opts.n_ep_decay, gamma=0.1, last_epoch=cur_ep)
    else:
        return NotImplementedError("no such learn rate policy")
    return scheduler

def make_laplace_pyramid(x, levels):
    """
    Make Laplacian Pyramid
    """
    pyramid = []
    current = x
    for i in range(levels):
        pyramid.append(laplacian(current))
        current = tensor_resample(
            current,
            (max(current.shape[2] // 2, 1), max(current.shape[3] // 2, 1)))
    pyramid.append(current)
    return pyramid

def laplacian(x):
    """
    Laplacian

    return:
       x - upsample(downsample(x))
    """
    return x - tensor_resample(
        tensor_resample(x, [x.shape[2] // 2, x.shape[3] // 2]),
        [x.shape[2], x.shape[3]])

def fold_laplace_pyramid(pyramid):
    """
    Fold Laplacian Pyramid
    """
    current = pyramid[-1]
    for i in range(len(pyramid) - 2, -1, -1):  # iterate from len-2 to 0
        up_h, up_w = pyramid[i].shape[2], pyramid[i].shape[3]
        current = pyramid[i] + tensor_resample(current, (up_h, up_w))
    return current

def tensor_resample(tensor, dst_size, mode='bilinear'):
    return F.interpolate(tensor, dst_size, mode=mode, align_corners=False)
