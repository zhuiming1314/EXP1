import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from net_module import calc_mean_std

def calc_emd_loss(pred, target):
    b, _, h, w = pred.shape
    pred = pred.reshape([b, -1, w * h])
    pred_norm = (pred**2).sum(1).reshape([b, -1, 1]).sqrt()
    #pred = pred.transpose([0, 2, 1])
    pred = pred.transpose(1, 2)
    target_t = target.reshape([b, -1, w * h])
    target_norm = (target**2).sum(1).reshape([b, 1, -1]).sqrt()
    similarity = torch.bmm(pred, target_t) / pred_norm / target_norm
    dist = 1. - similarity
    return dist

def mean_variance_norm(feat):
    size = feat.shape
    mean, std = calc_mean_std(feat)
    normalized_feat = (feat - mean.expand(size)) / std.expand(size)
    return normalized_feat

class CalcStyleEmdLoss():
    def __init__(self):
        super(CalcStyleEmdLoss, self).__init__()

    def __call__(self, pred, target):
        CX_M = calc_emd_loss(pred, target)
        m1 = CX_M.min(2).values
        #m1 = torch.unsqueeze(m1, 0)
        #m1 = torch.min(CX_M, 2, keepdim=True).values
        m2 = CX_M.min(1).values
        #m2 = torch.unsqueeze(m2, 0)
        #m2 = torch.min(CX_M, 1, keepdim=True).values
        m = torch.cat((m1.mean(1), m2.mean(1)))
        #m = torch.cat((torch.mean(m1), torch.mean(m2)))
        loss_remd = torch.max(m)

        return loss_remd

class CalcContentReltLoss():
    def __init__(self):
        super(CalcContentReltLoss, self).__init__()

    def __call__(self, pred, target):
        dM = 1
        Mx = calc_emd_loss(pred, target)
        Mx = Mx / Mx.sum(1, keepdim=True)
        My = calc_emd_loss(target, target)
        My = My / My.sum(1, keepdim=True)

        loss_content = torch.abs(dM * (Mx - My)).mean() * pred.shape[2] * pred.shape[3]

        return loss_content

class CalcContentLoss():
    def __init__(self):
        self.mse_loss = nn.MSELoss()

    def __call__(self, pred, target, norm=False):
        
        if norm == False:
            return self.mse_loss(pred, target)
        else:
            return self.mse_loss(mean_variance_norm(pred), mean_variance_norm(target))


class CalcStyleLoss():
    def __init__(self):
        self.mse_loss = nn.MSELoss()

    def __call__(self, pred, target):
        pred_mean, pred_std = calc_mean_std(pred)
        target_mean, target_std = calc_mean_std(target)

        return self.mse_loss(pred_mean, target_mean) + self.mse_loss(pred_std, target_std)

class CalcGanLoss():
    def __init__(self, gan_mode, target_real_label=1.0, target_fake_lable=0.0):
        super(CalcGanLoss, self).__init__()

        self.target_real_label = target_real_label
        self.target_fake_label = target_fake_lable

        self.gan_mode = gan_mode

        if self.gan_mode == 'lsgan':
            self.loss = nn.MSELoss()
        elif self.gan_mode == "vanilla":
            self.loss = nn.BCEWithLogitsLoss()
        elif self.gan_mode in ["wgan", "wgangp", "hinge", "logistic"]:
            self.loss = None
        else:
            raise NotImplementedError("gan mode %d not implemented" % gan_mode)

    def get_target_tensor(self, prediction, target_is_real):
        if target_is_real:
            if not hasattr(self, "target_real_tensor"):
                self.target_real_tensor = torch.ones_like(prediction).cuda()
            target_tensor = self.target_real_tensor
        else:
            if not hasattr(self, "target_fake_tensor"):
                self.target_fake_tensor = torch.zeros_like(prediction).cuda()
            target_tensor = self.target_fake_tensor
        return target_tensor


    def __call__(self, prediction, target_is_real, is_updating_d=None):
        if self.gan_mode in ["lsgan", "vanilla"]:
            target_tensor = self.get_target_tensor(prediction, target_is_real)
            loss = self.loss(prediction, target_tensor)
        elif self.gan_mode.find("wgan") != -1:
            if target_is_real:
                loss = -prediction.mean()
            else:
                loss = prediction.mean()
        elif self.gan_mode == "hinge":
            if target_is_real:
                loss = F.relu(1 - prediction) if is_updating_d else -prediction
            else:
                loss = F.relu(1 + prediction) if is_updating_d else prediction
            loss = loss.mean()
        elif self.gan_mode == 'logistic':
            pass
        else:
            print("error")
