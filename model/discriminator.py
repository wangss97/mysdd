import torch
import torch.nn as nn


class discriminatorBlock(nn.Module):
    def __init__(self, in_channel, out_channel) -> None:
        super().__init__()

        self.conv = nn.Conv2d(in_channel, out_channel, kernel_size=4, stride=2, padding=1)
        self.bn = nn.BatchNorm2d(out_channel)
        self.relu = nn.LeakyReLU()
    
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class discriminator(nn.Module):
    def __init__(self, in_channel=3) -> None:
        super().__init__()

        self.conv1 = nn.Conv2d(in_channel, 64, kernel_size=4, stride=2, padding=1)
        self.relu = nn.LeakyReLU()

        self.midConv = nn.Sequential(
            discriminatorBlock(64,128),
            discriminatorBlock(128,256),
            discriminatorBlock(256,512),
            discriminatorBlock(512,512),
            discriminatorBlock(512,512),
            discriminatorBlock(512,128)
        )

        self.lastConv = nn.Conv2d(128, 1,kernel_size=4, stride=2, padding=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)

        x = self.midConv(x)
        
        x = self.lastConv(x)    # [batchsize, 1, 1, 1]
        x = self.sigmoid(x)      

        return torch.squeeze(x)