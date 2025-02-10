# 注意里面对每一层都加了权重，根据个人需要自行选择
import torch
import torch.nn as nn
from torchvision import models
import time

class Vgg19(nn.Module):
    def __init__(self, requires_grad=False):
        super(Vgg19, self).__init__()
        vgg_pretrained_features = models.vgg19(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h_relu1 = self.slice1(X)
        h_relu2 = self.slice2(h_relu1)
        h_relu3 = self.slice3(h_relu2)
        h_relu4 = self.slice4(h_relu3)
        h_relu5 = self.slice5(h_relu4)
        out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
        return out

class VGGLoss(nn.Module):
    def __init__(self, layids = None, device=torch.device("cpu")):
        super(VGGLoss, self).__init__()
        self.vgg = Vgg19()
        if torch.cuda.is_available():
            #self.vgg.cuda()
            self.vgg = self.vgg.to(device)
        self.criterion = nn.L1Loss()
        self.weights = [1.0/32, 1.0/16, 1.0/8, 1.0/4, 1.0]
        self.layids = layids

    def forward(self, x, y):
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)
        loss = 0
        if self.layids is None:
            self.layids = list(range(len(x_vgg)))
        for i in self.layids:
            loss += self.weights[i] * self.criterion(x_vgg[i], y_vgg[i].detach())
        return loss


class StyleLoss(nn.Module):
    def __init__(self, layids = None):
        super(StyleLoss, self).__init__()
        self.vgg = Vgg19()
        if torch.cuda.is_available():
            self.vgg.cuda()
        self.criterion = nn.L1Loss()
        self.weights = [1.0, 1.0, 1.0, 1.0, 1.0]
        self.layids = layids

    def forward(self, x, y):
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)
        loss = []
        if self.layids is None:
            self.layids = list(range(len(x_vgg)))
        for i in self.layids:
            x_gram = self.gram(x_vgg[i].detach())
            y_gram = self.gram(y_vgg[i].detach())
            loss.append(self.weights[i]*self.criterion(x_gram, y_gram).item())
        return loss

    def gram(self, x):
        """
        gram compute
        """
        b, c, h, w = x.size()
        x = x.view(b*c, -1)
        return torch.mm(x, x.t())
