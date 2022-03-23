import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, in_channels, base_width):
        super(Encoder, self).__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels,base_width, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_width, base_width, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width),
            nn.ReLU(inplace=True))
        self.mp1 = nn.Sequential(nn.MaxPool2d(2))
        self.block2 = nn.Sequential(
            nn.Conv2d(base_width,base_width*2, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width*2),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_width*2, base_width*2, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width*2),
            nn.ReLU(inplace=True))
        self.mp2 = nn.Sequential(nn.MaxPool2d(2))
        self.block3 = nn.Sequential(
            nn.Conv2d(base_width*2,base_width*4, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width*4),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_width*4, base_width*4, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width*4),
            nn.ReLU(inplace=True))
        self.mp3 = nn.Sequential(nn.MaxPool2d(2))
        self.block4 = nn.Sequential(
            nn.Conv2d(base_width*4,base_width*8, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width*8),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_width*8, base_width*8, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width*8),
            nn.ReLU(inplace=True))
        self.mp4 = nn.Sequential(nn.MaxPool2d(2))
        self.block5 = nn.Sequential(
            nn.Conv2d(base_width*8,base_width*8, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width*8),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_width*8, base_width*8, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width*8),
            nn.ReLU(inplace=True))


    def forward(self, x):
        b1 = self.block1(x)
        mp1 = self.mp1(b1)
        b2 = self.block2(mp1)
        mp2 = self.mp3(b2)
        b3 = self.block3(mp2)
        mp3 = self.mp3(b3)
        b4 = self.block4(mp3)
        mp4 = self.mp4(b4)
        b5 = self.block5(mp4)
        return b2,b3,b4,b5

class Decoder(torch.nn.Module):
    def __init__(self, in_channel, block_list_size, out_channel) -> None:
        super().__init__()
        self.preConv = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=in_channel*block_list_size, out_channels=in_channel, kernel_size=1, stride=1, padding=0),
            torch.nn.BatchNorm2d(in_channel),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(in_channels=in_channel, out_channels=in_channel, kernel_size=1, stride=1, padding=0),
            torch.nn.BatchNorm2d(in_channel),
            torch.nn.ReLU(inplace=True)
        )

        self.up1 = DecoderUp(in_channel, in_channel//2)
        in_channel = in_channel // 2
        # self.up2 = DecoderUp(in_channel+in_channel+in_channel, in_channel//2)   #skip connect + concat,  in_channel = in_channel(上采样) + in_channel(mem跳步连接) +in_channel(原始跳步连接)
        # self.up2 = DecoderUp(in_channel+in_channel, in_channel//2)   #skip connect,  in_channel = in_channel(上采样) + in_channel(mem跳步连接)
        self.up2 = DecoderUp(in_channel, in_channel//2)   # no skip connect,  in_channel = in_channel(上采样)
        in_channel = in_channel // 2
        self.up3 = DecoderUp(in_channel, in_channel//2)   # no skip connect
        in_channel = in_channel // 2
        self.up4 = DecoderUp(in_channel, in_channel//2)   #no skip connect
        in_channel = in_channel // 2
        # self.up5 = DecoderUp(in_channel, in_channel//2)   #no skip connect
        # in_channel = in_channel // 2

        self.outConv = torch.nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=1, stride=1)
        # self.outActi = torch.nn.ReLU()

    def forward(self, x1, x2, x3, x4):
        '''  '''
        x4 = self.preConv(x4)

        x3_hat = self.up1(x4)
        # if skip
        if x3 is not None:
            x3_hat = torch.concat([x3_hat, x3],dim=1)

        x2_hat = self.up2(x3_hat)
        # if skip
        if x2 is not None:
            x2_hat = torch.concat([x2_hat, x2], dim=1)

        x1_hat = self.up3(x2_hat)
        # if skip
        if x1 is not None:
            x1_hat = torch.concat([x1_hat, x1], dim=1)

        x = self.up4(x1_hat)
        # x = self.up5(x)
        x = self.outConv(x)
        # x = self.outActi(x)

        return x

class DecoderUp(torch.nn.Module):
    def __init__(self, in_channel, out_channel) -> None:
        super().__init__()
        self.up = torch.nn.UpsamplingBilinear2d(scale_factor=2.0)
        self.convBlock = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=in_channel, out_channels=out_channel,kernel_size=3, stride=1, padding=1),
            torch.nn.BatchNorm2d(out_channel),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(in_channels=out_channel, out_channels=out_channel,kernel_size=3, stride=1,padding=1),
            torch.nn.BatchNorm2d(out_channel),
            torch.nn.ReLU(inplace=True),
        )
    
    def forward(self, x):
        x = self.up(x)
        x = self.convBlock(x)
        return x