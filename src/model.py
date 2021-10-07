import torch
from torch.autograd import backward
from torch.functional import norm
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules import loss
from torch.optim.adam import Adam
import network
import criterion

class LapNet(nn.Module):

    def __init__(self, args):
        super(LapNet, self).__init__()

        # parameters
        lr = 0.0001
        betas = (0.5, 0.999)
        weight_decay = 0.0001

        self.enc = network.Encoder()
        self.dec = network.Decoder()

        # define loss functions
        self.calc_style_emd_loss = criterion.CalcStyleEmdLoss()
        self.calc_content_relt_loss = criterion.CalcContentReltLoss()
        self.calc_content_loss = criterion.CalcContentLoss()
        self.calc_style_loss = criterion.CalcStyleLoss()

        # define the layer and weight of feature
        self.content_layers = args.content_layers
        self.style_layers = args.style_layers
        self.content_weight = args.content_weight
        self.style_weight = args.style_weight


        # define optimizer
        self.dec_opt = torch.optim.Adam(self.dec.parameters(), lr=lr, betas=betas, weight_decay=weight_decay)

    def forward(self, a, b):
        self.a_f = self.enc(a)
        self.b_f = self.enc(b)

        self.a2b = self.dec(self.a_f, self.b_f)

        self.image_display = torch.cat((a[0:1].detach().cpu(), b[0:1].detach().cpu(),
                                    self.a2b[0:1].detach().cpu(),
                                        a[1:2].detach().cpu(), b[1:2].detach().cpu(),
                                    self.a2b[1:2].detach().cpu()))

        return self.a2b

    def backward_dec(self):
        self.a2b_f = self.enc(self.a2b)

        # content loss
        self.loss_c = 0
        for layer in self.content_layers:
            self.loss_c += self.calc_content_loss(self.a2b_f[layer], self.a_f[layer], norm=True)

        # style loss
        self.loss_s = 0
        for layer in self.style_layers:
            self.loss_s += self.calc_style_loss(self.a2b_f[layer], self.b_f[layer])

        # identify loss
        # to do
        # ralative loss
        self.loss_style_remd = self.calc_style_emd_loss(self.a2b_f["r31"], self.b_f["r31"])\
                                + self.calc_style_emd_loss(self.a2b_f["r41"], self.b_f["r41"])

        self.loss_content_relt = self.calc_content_relt_loss(self.a2b_f["r31"], self.a_f["r31"])\
                                + self.calc_content_relt_loss(self.a2b_f["r41"], self.a_f["r41"])

        
        self.loss = self.loss_c * self.content_weight + self.loss_s * self.style_weight + \
                        self.loss_style_remd + self.loss_content_relt
        print("loss_c:{}    loss_s:{}".format(self.loss_c, self.loss_s))
        print("loss_style_remd:{}    loss_content_relt:{}".format(self.loss_style_remd, self.loss_content_relt))
        self.loss.backward()
        return self.loss

    def update_dec(self, input_a, input_b):
        self.forward(input_a, input_b)
        print("self.a2b.size=", self.a2b.size())
        
        self.dec_opt.zero_grad()

        self.backward_dec()

        self.dec_opt.step()

    def initialize(self):
        self.dec.apply(network.gaussian_weights_init)

    def set_scheduler(self, opts, last_ep=0):
        self.dec_sch = network.get_scheduler(self.dec_opt, opts, last_ep)

    def save_model(self, filename, ep, total_iter):
        state = {
            "dec": self.dec.state_dict(),
            "dec_opt": self.dec_opt.state_dict(),
            "ep": ep,
            "total_iter": total_iter
        }
        torch.save(state, filename)

    def resume(self, filename, train=True):
        checkpoint = torch.load(filename)

        if train:
            self.dec.load_state_dict(checkpoint["dec"])
            self.dec_opt.load_state_dict(checkpoint["dec_opt"])
        
            return checkpoint["ep"], checkpoint["total_iter"]

    def set_gpu(self, gpu):
        self.gpu = gpu
        self.enc.cuda(self.gpu)
        self.dec.cuda(self.gpu)

class LapAdaINModel1(nn.Module):
    def __init__(self, args):
        super(LapAdaINModel1, self).__init__()

        # define draft module
        self.enc = network.Encoder()
        self.dec = network.Decoder()

        self.set_requires_grad([self.enc], False)
        self.set_requires_grad([self.dec], False)

        # define revise module
        self.rev = network.ReviseNet(3)
        self.dis = network.Discirminator()

        # define loss functions
        self.calc_style_emd_loss = criterion.CalcStyleEmdLoss()
        self.calc_content_relt_loss = criterion.CalcContentReltLoss()
        self.calc_content_loss = criterion.CalcContentLoss()
        self.calc_style_loss = criterion.CalcStyleLoss()
        self.calc_gan_loss = criterion.CalcGanLoss("lsgan")

        self.content_layers = args.content_layers
        self.style_layers = args.style_layers
        self.content_weight = args.content_weight
        self.style_weight = args.style_weight

        # define optimizer
        self.rev_opt = torch.optim.Adam(self.rev.parameters(), lr=1e-4, betas=(0.9, 0.999))
        self.dis_opt = torch.optim.Adam(self.dis.parameters(), lr=1e-4, betas=(0.9, 0.999))

    def setup_input(self, a, b):
        self.pyr_a = network.make_laplace_pyramid(self.a, 1)
        self.pyr_b = network.make_laplace_pyramid(self.b, 1)
        self.pyr_a.append(a)
        self.pry_b.append(b)

    def initialize(self):
        self.rev.apply(network.gaussian_weights_init)
        self.dis.apply(network.gaussian_weights_init)

    def set_requires_grad(self, nets, requires_grad=False):
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.trainable = requires_grad

    def forward(self):
        a_f = self.enc(self.pyr_a[1])
        b_f = self.enc(self.pyr_b[1])

        a2b = self.dec(a_f, b_f)
        a2b_up = F.interpolate(a2b, scale_factor=2)

        rev_input = torch.cat((self.pyr_a[0], a2b_up), 1)
        a2b_rev_lap = self.rev(rev_input)
        a2b_rev = network.fold_laplace_pyramid([a2b_rev_lap, a2b])

        self.a2b_rev = a2b_rev

    def backward_gen(self):
        self.a2b_f = self.enc(self.a2b_rev)
        self.a_f = self.enc(self.pyr_a[2])
        self.b_f = self.enc(self.pyr_a[2])

        # content loss
        self.loss_c = 0
        for layer in self.content_layers:
            self.loss_c += self.calc_content_loss(self.a2b_f[layer], self.a_f[layer], norm=True)

        # style loss
        self.loss_s = 0
        for layer in self.style_layers:
            self.loss_s += self.calc_style_loss(self.a2b_f[layer], self.b[layer])

        # relative loss
        self.loss_style_remd = self.calc_style_emd_loss(self.a2b_f["r31"], self.b_f["r31"])\
                                + self.calc_style_emd_loss(self.a2b_f["r41"], self.b_f["r41"])
        self.loss_content_relt = self.calc_content_relt_loss(self.a2b_f["r31"], self.a_f["r31"])\
                                + self.calc_content_relt_loss(self.a2b_f["r41"], self.a_f["r41"])
        
        # gan loss
        pred_fake = self.dis(self.a2b_rev)
        self.loss_gan_g = self.calc_gan_loss(pred_fake, True)
        
        self.loss = self.loss_gan_g +\
                    self.loss_c * self.content_weight+\
                    self.loss_s * self.style_weight+\
                    self.loss_style_remd * 10 +\
                    self.loss_content_relt * 16

        print("loss_c:{}    loss_s:{}".format(self.loss_c, self.loss_s))
        print("loss_style_remd:{}    loss_content_relt:{}".format(self.loss_style_remd, self.loss_content_relt))
        print("loss_gan_g:{}".format(self.loss_gan_g))
        self.loss.backward()
        return self.loss

    def update_gen(self):
        self.set_requires_grad(self.dis, False)
        self.rev_opt.zero_grad()
        self.backward_gen()
        self.rev_opt.step()

    def backward_dis(self):
        pred_fake = self.dis(self.a2b_rev.detach())
        loss_dis_fake = self.calc_gan_loss(pred_fake, False)
        pred_real = self.dis(self.pyr_b[2])
        loss_dis_real = self.calc_gan_loss(pred_real, True)

        self.loss_dis = 0.5 * (loss_dis_fake + loss_dis_real)
        print("loss_gen_dis:{}".format(self.loss_dis))
        
        self.loss_dis.backward()
        return self.loss_dis

    def update_dis(self):
        self.forward()

        self.set_requires_grad(self.dis, True)
        self.dis_opt.zero_grad()
        self.backward_dis()
        self.step()

    def save_model(self, filename, ep, total_iter):
        state = {
            "rev": self.rev.state_dict(),
            "rev_opt": self.rev_opt.state_dict(),
            "dis": self.dis.state_dict(),
            "dis_opt": self.dis_opt.state_dict(),
            "ep": ep,
            "total_iter": total_iter
        }
        torch.save(state, filename)

    def resume(self, filename, train=True):
        checkpoint = torch.load(filename)

        if train:
            self.rev.load_state_dict(checkpoint["rev"])
            self.rev_opt.load_state_dict(checkpoint["rev_opt"])
            self.dis.load_state_dict(checkpoint["dis"])
            self.dis_opt.load_state_dict(checkpoint["dis_opt"])
        
            return checkpoint["ep"], checkpoint["total_iter"]

class LapAdaINModel2(nn.Module):
    def __init__(self, args):
        super(LapAdaINModel2, self).__init__()

        # define draft module
        self.enc = network.Encoder()
        self.dec = network.Decoder()

        self.set_requires_grad([self.enc], False)
        self.set_requires_grad([self.dec], False)

        # define first revise module
        self.rev = network.ReviseNet(3)
        self.set_requires_grad([self.rev], False)

        # define second revise module
        self.rev2 = network.ReviseNet(3)

        self.dis = network.Discirminator()

        # define loss functions
        self.calc_style_emd_loss = criterion.CalcStyleEmdLoss()
        self.calc_content_relt_loss = criterion.CalcContentReltLoss()
        self.calc_content_loss = criterion.CalcContentLoss()
        self.calc_style_loss = criterion.CalcStyleLoss()
        self.calc_gan_loss = criterion.CalcGanLoss("lsgan")

        self.content_layers = args.content_layers
        self.style_layers = args.style_layers
        self.content_weight = args.content_weight
        self.style_weight = args.style_weight

        # define optimizer
        self.rev2_opt = torch.optim.Adam(self.rev2.parameters(), lr=1e-4, betas=(0.9, 0.999))
        self.dis_opt = torch.optim.Adam(self.dis.parameters(), lr=1e-4, betas=(0.9, 0.999))

    def setup_input(self, a, b):
        self.input_a = a
        self.input_b = b
        self.pyr_a = network.make_laplace_pyramid(self.a, 2)
        self.pyr_b = network.make_laplace_pyramid(self.b, 2)
        self.pyr_a.append(a)
        self.pry_b.append(b)

    def initialize(self):
        self.rev2.apply(network.gaussian_weights_init)
        self.dis.apply(network.gaussian_weights_init)

    def set_requires_grad(self, nets, requires_grad=False):
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.trainable = requires_grad

    def forward(self):
        a_f = self.enc(self.pyr_a[2])
        b_f = self.enc(self.pyr_b[2])

        self.a2b = self.dec(a_f, b_f)
        a2b_up = F.interpolate(self.a2b, scale_factor=2)

        rev_input = torch.cat((self.pyr_a[1], a2b_up), 1)
        a2b_rev_lap = self.rev(rev_input)
        self.a2b_rev = network.fold_laplace_pyramid([a2b_rev_lap, self.a2b])

        a2b_up2 = F.interpolate(self.a2b_rev, scale_factor=2)

        rev_input2 = torch.cat((self.pyr_a[0], a2b_up2), 1)
        a2b_rev2_lap = self.rev2(rev_input2)
        a2b_rev2 = network.fold_laplace_pyramid([a2b_rev2_lap, a2b_rev_lap, self.a2b])

        self.a2b_rev2 = a2b_rev2

    def backward_gen(self):
        self.a2b_f = self.enc(self.a2b_rev2)
        self.a_f = self.enc(self.pyr_a[3])
        self.b_f = self.enc(self.pyr_a[3])

        # content loss
        self.loss_c = 0
        for layer in self.content_layers:
            self.loss_c += self.calc_content_loss(self.a2b_f[layer], self.a_f[layer], norm=True)

        # style loss
        self.loss_s = 0
        for layer in self.style_layers:
            self.loss_s += self.calc_style_loss(self.a2b_f[layer], self.b[layer])

        # relative loss
        self.loss_style_remd = self.calc_style_emd_loss(self.a2b_f["r41"], self.b_f["r41"])
        self.loss_content_relt =  self.calc_content_relt_loss(self.a2b_f["r41"], self.a_f["r41"])
        
        # gan loss
        pred_fake = self.dis(self.a2b_rev2)
        self.loss_gan_g = self.calc_gan_loss(pred_fake, True)
        
        self.loss = self.loss_gan_g +\
                    self.loss_c * self.content_weight+\
                    self.loss_s * self.style_weight+\
                    self.loss_style_remd * 10 +\
                    self.loss_content_relt * 16

        print("loss_c:{}    loss_s:{}".format(self.loss_c, self.loss_s))
        print("loss_style_remd:{}    loss_content_relt:{}".format(self.loss_style_remd, self.loss_content_relt))
        print("loss_gan_g:{}".format(self.loss_gan_g))
        self.loss.backward()
        return self.loss

    def update_gen(self):
        self.set_requires_grad(self.dis, False)
        self.rev2_opt.zero_grad()
        self.backward_gen()
        self.rev2_opt.step()

    def backward_dis(self):
        pred_fake = self.dis(self.a2b_rev2.detach())
        loss_dis_fake = self.calc_gan_loss(pred_fake, False)
        pred_real = self.dis(self.pyr_b[3])
        loss_dis_real = self.calc_gan_loss(pred_real, True)

        self.loss_dis = 0.5 * (loss_dis_fake + loss_dis_real)
        print("loss_gen_dis:{}".format(self.loss_dis))
        
        self.loss_dis.backward()
        return self.loss_dis

    def update_dis(self):
        self.forward()

        self.set_requires_grad(self.dis, True)
        self.dis_opt.zero_grad()
        self.backward_dis()
        self.step()

    def save_model(self, filename, ep, total_iter):
        state = {
            "rev2": self.rev2.state_dict(),
            "rev2_opt": self.rev2_opt.state_dict(),
            "dis": self.dis.state_dict(),
            "dis_opt": self.dis_opt.state_dict(),
            "ep": ep,
            "total_iter": total_iter
        }
        torch.save(state, filename)

    def resume(self, filename, train=True):
        checkpoint = torch.load(filename)

        if train:
            self.rev2.load_state_dict(checkpoint["rev"])
            self.rev2_opt.load_state_dict(checkpoint["rev_opt"])
            self.dis.load_state_dict(checkpoint["dis"])
            self.dis_opt.load_state_dict(checkpoint["dis_opt"])
        
            return checkpoint["ep"], checkpoint["total_iter"]