import torch
import torch.nn as nn
from model.resnet import resnet18


class DiscriminativeSubNetwork(nn.Module):
    def __init__(self,in_channels=3, out_channels=2, base_channels=64, out_features=False, depth=4):
        super(DiscriminativeSubNetwork, self).__init__()
        base_width = base_channels
        self.encoder_segment = EncoderDiscriminative(in_channels, base_width, depth)
        self.decoder_segment = DecoderDiscriminative(base_width, out_channels=out_channels, depth=depth)
        self.out_features = out_features

    def forward(self, x):
        x_list = self.encoder_segment(x)
        output_segment = self.decoder_segment(x_list)
        return output_segment

        # b1,b2,b3,b4,b5,b6 = self.encoder_segment(x)

        # output_segment = self.decoder_segment(b1,b2,b3,b4,b5,b6)
        # if self.out_features:
        #     return output_segment, b2, b3, b4, b5, b6
        # else:
        #     return output_segment


class EncoderDiscriminative(nn.Module):
    def __init__(self, in_channels, base_width, depth=6):
        super(EncoderDiscriminative, self).__init__()

        # first_layer = nn.Sequential(
        #     nn.Conv2d(in_channels,base_width, kernel_size=3, padding=1),
        #     nn.BatchNorm2d(base_width),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(base_width, base_width, kernel_size=3, padding=1),
        #     nn.BatchNorm2d(base_width),
        #     nn.ReLU(inplace=True),
        # )

        # self.net = [first_layer]

        # for i in range(1, depth-1):
        #     inrate = 2**(i-1)
        #     outrate = 2**i
        #     if inrate > 8:
        #         inrate = 8
        #     if outrate > 8:
        #         outrate = 8

        #     layer = nn.Sequential(
        #         nn.MaxPool2d(2),
        #         nn.Conv2d(base_width * inrate,base_width * outrate, kernel_size=3, padding=1),
        #         nn.BatchNorm2d(base_width * outrate),
        #         nn.ReLU(inplace=True),
        #         nn.Conv2d(base_width * outrate, base_width * outrate, kernel_size=3, padding=1),
        #         nn.BatchNorm2d(base_width * outrate),
        #         nn.ReLU(inplace=True),
        #     )
        #     self.net.append(layer)

        # inrate = 2**(depth-1 -1)
        # if inrate > 8:
        #     inrate = 8
        # last_layer = nn.Sequential(
        #         nn.MaxPool2d(2),
        #         nn.Conv2d(base_width * inrate,base_width * inrate, kernel_size=3, padding=1),
        #         nn.BatchNorm2d(base_width * inrate),
        #         nn.ReLU(inplace=True),
        #         nn.Conv2d(base_width * inrate, base_width * inrate, kernel_size=3, padding=1),
        #         nn.BatchNorm2d(base_width * inrate),
        #         nn.ReLU(inplace=True),
        # )
        # self.net.append(last_layer)
        

        # self.net = nn.ModuleList(self.net)

        self.depth = depth

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

        self.mp5 = nn.Sequential(nn.MaxPool2d(2))
        self.block6 = nn.Sequential(
            nn.Conv2d(base_width*8,base_width*8, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width*8),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_width*8, base_width*8, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width*8),
            nn.ReLU(inplace=True))


    def forward(self, x):

        # res = []
        # for layer in self.net:
        #     x = layer(x)
        #     res.append(x)
        # return res

        b1 = self.block1(x)
        mp1 = self.mp1(b1)
        b2 = self.block2(mp1)
        mp2 = self.mp3(b2)
        b3 = self.block3(mp2)
        mp3 = self.mp3(b3)
        b4 = self.block4(mp3)
        mp4 = self.mp4(b4)
        b5 = self.block5(mp4)
        if self.depth == 5:
            return [b1,b2,b3,b4,b5]
        mp5 = self.mp5(b5)
        b6 = self.block6(mp5)
        return [b1,b2,b3,b4,b5,b6]

class DecoderDiscriminative(nn.Module):
    def __init__(self, base_width, out_channels=1, depth=6):
        super(DecoderDiscriminative, self).__init__()

        
        # outrate = 2**(depth-2)
        # inrate = 2**(depth-2 -1)
        # if outrate > 8:
        #     outrate = 8
        # if inrate > 8:
        #     inrate = 8

        # self.up = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
        #                          nn.Conv2d(base_width * outrate, base_width * inrate, kernel_size=3, padding=1),
        #                          nn.BatchNorm2d(base_width * inrate),
        #                          nn.ReLU(inplace=True))
        # self.net = []

        # for i in range(depth-2, 0, -1):
        #     outrate = 2**i
        #     inrate = 2**(i-1)
        #     if outrate > 8:
        #         outrate = 8
        #     if inrate > 8:
        #         inrate = 8
        #     if inrate == 1:
        #         upperrate = 1
        #     else:
        #         upperrate = inrate // 2

        #     layer = nn.Sequential(
        #         nn.Conv2d(base_width * (inrate+outrate), base_width * inrate, kernel_size=3, padding=1),
        #         nn.BatchNorm2d(base_width * inrate),
        #         nn.ReLU(inplace=True),
        #         nn.Conv2d(base_width * inrate, base_width * inrate, kernel_size=3, padding=1),
        #         nn.BatchNorm2d(base_width * inrate),
        #         nn.ReLU(inplace=True),
                
        #         nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
        #         nn.Conv2d(base_width * inrate, base_width * upperrate, kernel_size=3, padding=1),
        #         nn.BatchNorm2d(base_width * upperrate),
        #         nn.ReLU(inplace=True)
        #     )
        #     self.net.append(layer)

        # last_layer = nn.Sequential(
        #     nn.Conv2d(base_width*2, base_width, kernel_size=3, padding=1),
        #     nn.BatchNorm2d(base_width),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(base_width, base_width, kernel_size=3, padding=1),
        #     nn.BatchNorm2d(base_width),
        #     nn.ReLU(inplace=True)
        # )
        # self.net.append(last_layer)
        # self.net = nn.ModuleList(self.net)
        # self.fin_out = nn.Sequential(nn.Conv2d(base_width, out_channels, kernel_size=1, padding=0))

        self.depth = depth
        self.up_b = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                                 nn.Conv2d(base_width * 8, base_width * 8, kernel_size=3, padding=1),
                                 nn.BatchNorm2d(base_width * 8),
                                 nn.ReLU(inplace=True))
        self.db_b = nn.Sequential(
            nn.Conv2d(base_width*(8+8), base_width*8, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width*8),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_width * 8, base_width * 8, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width * 8),
            nn.ReLU(inplace=True)
        )


        self.up1 = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                                 nn.Conv2d(base_width * 8, base_width * 4, kernel_size=3, padding=1),
                                 nn.BatchNorm2d(base_width * 4),
                                 nn.ReLU(inplace=True))
        self.db1 = nn.Sequential(
            nn.Conv2d(base_width*(4+8), base_width*4, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width*4),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_width * 4, base_width * 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width * 4),
            nn.ReLU(inplace=True)
        )

        self.up2 = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                                 nn.Conv2d(base_width * 4, base_width * 2, kernel_size=3, padding=1),
                                 nn.BatchNorm2d(base_width * 2),
                                 nn.ReLU(inplace=True))
        self.db2 = nn.Sequential(
            nn.Conv2d(base_width*(2+4), base_width*2, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width*2),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_width * 2, base_width * 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width * 2),
            nn.ReLU(inplace=True)
        )

        self.up3 = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                                 nn.Conv2d(base_width * 2, base_width, kernel_size=3, padding=1),
                                 nn.BatchNorm2d(base_width),
                                 nn.ReLU(inplace=True))
        self.db3 = nn.Sequential(
            nn.Conv2d(base_width*(2+1), base_width, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_width, base_width, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width),
            nn.ReLU(inplace=True)
        )

        self.up4 = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                                 nn.Conv2d(base_width, base_width, kernel_size=3, padding=1),
                                 nn.BatchNorm2d(base_width),
                                 nn.ReLU(inplace=True))
        self.db4 = nn.Sequential(
            nn.Conv2d(base_width*2, base_width, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_width, base_width, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width),
            nn.ReLU(inplace=True)
        )

        self.fin_out = nn.Sequential(nn.Conv2d(base_width, out_channels, kernel_size=1, padding=0))

    def forward(self, x_list):
        # x_deepest = x_list[-1]
        # x = self.up(x_deepest)

        # x_list = x_list[:-1][::-1]
        # for i,layer in enumerate(self.net):
        #     x = torch.cat((x_list[i],x),dim=1)
        #     x = layer(x)
        # x = self.fin_out(x)
        # return x

        b1,b2,b3,b4,b5,b6 = x_list
        
        up_b = self.up_b(b6)
        cat_b = torch.cat((up_b,b5),dim=1)
        db_b = self.db_b(cat_b)

        up1 = self.up1(db_b)
        cat1 = torch.cat((up1,b4),dim=1)
        db1 = self.db1(cat1)

        up2 = self.up2(db1)
        cat2 = torch.cat((up2,b3),dim=1)
        db2 = self.db2(cat2)

        up3 = self.up3(db2)
        cat3 = torch.cat((up3,b2),dim=1)
        db3 = self.db3(cat3)

        up4 = self.up4(db3)
        cat4 = torch.cat((up4,b1),dim=1)
        db4 = self.db4(cat4)

        out = self.fin_out(db4)
        return out


class PatchSegNet(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

        base_width = 16
        self.block1 = nn.Sequential(
            nn.Conv2d(6, base_width, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_width, base_width, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width),
            nn.ReLU(inplace=True))
        self.mp1 = nn.MaxPool2d(2)

        self.block2 = nn.Sequential(
            nn.Conv2d(base_width,base_width*2, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width*2),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_width*2, base_width*2, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width*2),
            nn.ReLU(inplace=True))
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.fc = nn.Linear(base_width*2, 2, bias=False)

    def forward(self, x):
        x = self.block1(x)
        x = self.mp1(x)

        x = self.block2(x)
        x = self.avgpool(x)
        
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x


class SegDecoder(torch.nn.Module):
    def __init__(self, in_channel,block_list_size, out_channel=1, depth=4) -> None:
        super().__init__()
        self.preConv = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=in_channel*(block_list_size+1), out_channels=in_channel, kernel_size=1, stride=1, padding=0),
            torch.nn.BatchNorm2d(in_channel),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(in_channels=in_channel, out_channels=in_channel, kernel_size=1, stride=1, padding=0),
            torch.nn.BatchNorm2d(in_channel),
            torch.nn.ReLU(inplace=True)
        )

        self.depth = depth
        if self.depth == 4:
            print(f'seg decoder depth {depth},expected bottleneck width: 16')
        if self.depth == 5:
            print(f'seg decoder depth {depth},expected bottleneck width: 8')

        ''' 规则乘2上采样 '''
        self.up1 = DecoderUp(in_channel, in_channel//2)                                                                                                                                                        
        in_channel = in_channel // 2
        ''' 与encoder对称上采样 '''
        # self.up1 = DecoderUp(in_channel, in_channel)

        self.up2 = DecoderUp(in_channel*2, in_channel//2)   #  skip connect,  in_channel = in_channel(上采样) + skip
        in_channel = in_channel // 2
        self.up3 = DecoderUp(in_channel*2, in_channel//2)   #  skip connect
        in_channel = in_channel // 2
        self.up4 = DecoderUp(in_channel*2, in_channel//2)   #  skip connect
        in_channel = in_channel // 2
        if self.depth == 5:
            self.up5 = DecoderUp(in_channel, in_channel//2)   #no skip connect
            in_channel = in_channel // 2

        self.outConv = torch.nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=1, stride=1)
        # self.outActi = torch.nn.ReLU()

    def forward(self, x1=None, x2=None, x3=None, x_in=None):
        '''  '''
        x4_hat = self.preConv(x_in)

        x3_hat = self.up1(x4_hat)
        if x3 is not None:
            x3_hat = torch.concat([x3_hat, x3],dim=1)

        x2_hat = self.up2(x3_hat)
        if x2 is not None:
            x2_hat = torch.concat([x2_hat, x2],dim=1)

        x1_hat = self.up3(x2_hat)
        if x1 is not None:
            x1_hat = torch.concat([x1_hat, x1],dim=1)

        x = self.up4(x1_hat)
        if self.depth == 5:
            x = self.up5(x)

        out = self.outConv(x)

        return x, out

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