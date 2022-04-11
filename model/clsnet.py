from turtle import forward
import torch
import torch.nn as nn
from model.resnet import resnet18

class ClsSubNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.cls = resnet18(pretrained = True)

        self.preConv = nn.Sequential(
            nn.Conv2d(6, 3, kernel_size=3, padding=1),
            nn.BatchNorm2d(3),
            nn.ReLU(inplace=True),
            nn.Conv2d(3, 3, kernel_size=3, padding=1),
            nn.BatchNorm2d(3),
            nn.ReLU(inplace=True))
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, 2)

    def forward(self, x):
        x = self.preConv(x)

        x1,x2,x3,x4 = self.cls(x)

        x = self.avgpool(x4)

        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

class clsBlock(nn.Module):
    def __init__(self, in_channels, out_channels) -> None:
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(in_channels,out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True))

        self.mp = nn.Sequential(nn.MaxPool2d(2))

    def forward(self, x):
        x = self.block(x)
        x = self.mp(x)

        return x

class clsDecoder(nn.Module):
    def __init__(self, in_channels = 128) -> None:
        super().__init__()

        # here pic_size 256 
        self.down1 = clsBlock(in_channels*2, in_channels*2)    # 128*2 concat, 256;
        in_channels = in_channels * 2 
        # here pic_size 128
        self.down2 = clsBlock(in_channels*3, in_channels*2)   # 256*3 concat, 512;
        in_channels = in_channels * 2
        # here pic_size 64
        self.down3 = clsBlock(in_channels*3, in_channels*2)  # 512*3 concat, 1024; 
        in_channels = in_channels*2
        # here pic_size 32
        self.down4 = clsBlock(in_channels*3, in_channels)    # 1024*3 concat, 1024; 
        # here pic_size 16

        self.outConv = nn.Sequential(                           # 1024*3 concat, 2048
            nn.Conv2d(in_channels*3,in_channels*2, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels*2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels*2, in_channels*2, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels*2),
            nn.ReLU(inplace=True))

        self.avgPool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(2048, 2)
    
    def forward(self, x1, x2, x3, x4, x5):
        x2_hat = self.down1(x1)
        if x2 is not None:
            x2_hat = torch.concat([x2_hat, x2], dim=1)
        x3_hat = self.down2(x2_hat)
        if x3 is not None:
            x3_hat = torch.concat([x3_hat, x3], dim=1)
        x4_hat = self.down3(x3_hat)
        if x4 is not None:
            x4_hat = torch.concat([x4_hat, x4], dim=1)
        x5_hat = self.down4(x4_hat)
        if x5 is not None:
            x5_hat = torch.concat([x5_hat, x5], dim=1)
        
        out = self.outConv(x5_hat)
        out = self.avgPool(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)
        return out


class clsDecoder2(nn.Module):
    def __init__(self, in_channels = 1024) -> None:
        super().__init__()

        self.outConv = nn.Sequential(                           # 1024*2, 1024
            nn.Conv2d(in_channels*2,in_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels,in_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True))

        self.avgPool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(1024, 2)
    
    def forward(self, x):
        out = self.outConv(x)
        out = self.avgPool(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)
        return out

class clsNet2(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        