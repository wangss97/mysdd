from tkinter.messagebox import NO
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from model.resnet import resnet18, wide_resnet50_2
from model.memory import BlockMemory
from model.segnet import DiscriminativeSubNetwork


class FeatureExtractor(torch.nn.Module):
    def __init__(self) -> None:
        super(FeatureExtractor, self).__init__()
        # self.extractor = resnet18(pretrained=True, progress=True)
        self.extractor = wide_resnet50_2(pretrained=True, progress=True)

    def forward(self, input) -> torch.Tensor:
        return self.extractor(input)

class Attention(torch.nn.Module):
    def __init__(self, feature_size, key_size) -> None:
        super(Attention, self).__init__()
        self.Qdense = nn.Linear(feature_size, key_size, bias=True)
        self.Kdense = nn.Linear(feature_size, key_size, bias=True)
        self.Vdense = nn.Linear(feature_size, key_size, bias=True)

    def forward(self, x):
        ''' 计算相关度a (做点积，逐元素乘，相加) '''
              
        # x维度 [b, h*w, c]
        # q, k, v维度[b, h*w, key_size]
        # q =self.Qdense(x)
        # k =self.Kdense(x)
        # v =self.Vdense(x)
        q = x
        k = x
        v = x
        
        # [b, h*w, c] 矩阵乘 [b, c, h*w] = [b, h*w, h*w]
        # a[b, i,j]表示第b个feature map中第i个像素和第j个像素的相关度
        a = torch.matmul(q, torch.transpose(k, -1, -2))
        a = torch.softmax(a, dim=-1)

        # output = [b, h*w, h*w] 矩阵乘 [b, h*w, key_size] = [b, h*w, key_size]
        # output为注意力加权之后的像素
        output = torch.matmul(a, v)
        return output

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
        self.up2 = DecoderUp(in_channel+in_channel+in_channel, in_channel//2)   #skip connect + concat,  in_channel = in_channel(上采样) + in_channel(mem跳步连接) +in_channel(原始跳步连接)
        # self.up2 = DecoderUp(in_channel+in_channel, in_channel//2)   #skip connect,  in_channel = in_channel(上采样) + in_channel(mem跳步连接)
        # self.up2 = DecoderUp(in_channel, in_channel//2)   # no skip connect,  in_channel = in_channel(上采样)
        in_channel = in_channel // 2
        # self.up3 = DecoderUp(in_channel+in_channel, in_channel//2)   #skip connect
        self.up3 = DecoderUp(in_channel, in_channel//2)   # no skip connect
        in_channel = in_channel // 2
        # self.up4 = DecoderUp(in_channel+in_channel, in_channel//2)   #skip connect
        self.up4 = DecoderUp(in_channel, in_channel//2)   #no skip connect
        in_channel = in_channel // 2
        self.up5 = DecoderUp(in_channel, in_channel//2)
        in_channel = in_channel // 2

        # self.up1 = DecoderUp(in_channel, 256)
        # self.up2 = DecoderUp(256, 128)
        # self.up3 = DecoderUp(128, 64)
        # self.up4 = DecoderUp(64, 32)
        # self.up5 = DecoderUp(32, 16)

        self.outConv = torch.nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=1, stride=1)
        self.outActi = torch.nn.ReLU()

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
        x = self.up5(x)
        x = self.outConv(x)
        x = self.outActi(x)

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

class model(torch.nn.Module):
    def __init__(self, in_channel=3, out_channel=3, mem_block_list=[1,2,4,-1], device='cuda') -> None:
        super(model, self).__init__()

        # self.featureExtractor = FeatureExtractor()
        self.featureExtractor = wide_resnet50_2(pretrained=True, progress=True)
        decoder_input_channel = 2048
        self.pos_size = 0
        # self.featureExtractor = resnet18(pretrained=True, progress=True)
        # decoder_input_channel = 512


        self.key_size = 512
        ''' 当前的attention只是在单个图的所有像素之间做attention，不是在整个batch的所有像素之间做attention '''
        # self.atten = Attention(512, self.key_size)

        self.memeorys = nn.ModuleList([
                BlockMemory(block_list = [-1], mem_size=50, fea_dim=decoder_input_channel//8+self.pos_size, pos_size=self.pos_size, shrink_thres=0.02, device=device),
                BlockMemory(block_list = [-1], mem_size=50, fea_dim=decoder_input_channel//4+self.pos_size, pos_size=self.pos_size, shrink_thres=0.02, device=device),
                BlockMemory(block_list = [-1], mem_size=50, fea_dim=decoder_input_channel//2+self.pos_size, pos_size=self.pos_size, shrink_thres=0.02, device=device),
                BlockMemory(block_list = mem_block_list, mem_size=50, fea_dim=decoder_input_channel+self.pos_size, pos_size=self.pos_size, shrink_thres=0.02, device=device)
        ])

        print('mem_block_list:', mem_block_list, len(self.memeorys[-1].memorys), 'mem_fea_size:',self.memeorys[-1].memorys[-1].weight.shape)

        # self.memory = BlockMemory(block_list=mem_block_list, mem_size=100, fea_dim=decoder_input_channel, shrink_thres=0.0025, device=device)

        self.decoder = Decoder(in_channel= decoder_input_channel, block_list_size=len(mem_block_list) + 1, out_channel=out_channel)  #拼接未重构的bottleneck
        # self.decoder = Decoder(in_channel= decoder_input_channel, block_list_size=len(mem_block_list), out_channel=out_channel)  # 不拼接原始bottleneck

        self.segnet = DiscriminativeSubNetwork(in_channels=6, out_channels=2)

        pass

    def forward(self, input, ori_img):
        # b, c, h, w
        b, c, h, w = input.size()
        x1, x2, x3, x4 = self.featureExtractor(input)
        bottleneck = [x1,x2,x3,x4]

        # print('bottleneck shape:', bottleneck.shape)

        ''' trainable memory bank '''
        bottleneck_hat, entropy_loss, triplet_loss, norm_loss, compact_loss, distance_loss = [],[],[],[],[],[]
        for i in range(2, 4):
            x_hat_i, entropy_loss_i, triplet_loss_i, norm_loss_i, compact_loss_i, distance_loss_i = self.memeorys[i](bottleneck[i])
            bottleneck_hat.append(x_hat_i)
            entropy_loss.append(entropy_loss_i)
            triplet_loss.append(triplet_loss_i)
            norm_loss.append(norm_loss_i)
            compact_loss.append(compact_loss_i)
            distance_loss.append(distance_loss_i)
        
        
        # x1_hat, x2_hat, x3_hat, x4_hat = bottleneck_hat
        x1_hat, x2_hat, x3_hat, x4_hat = None, None, bottleneck_hat[-2], bottleneck_hat[-1]
        # x1_hat, x2_hat, x3_hat, x4_hat = None, None, None, bottleneck_hat[-1]

        ''' 拼接未重构的bottleneck '''
        x3_hat = torch.concat([x3, x3_hat], dim=1)
        x4_hat = torch.concat([x4, x4_hat], dim=1)

        output = self.decoder(x1_hat, x2_hat, x3_hat, x4_hat)

        ''' 使用segNet '''
        # mask = self.segnet(torch.concat([output, ori_img], dim=1))
        ''' 不适用segNet '''
        mask = torch.zeros((b,2,h,w))
        
        return output, mask, sum(entropy_loss), sum(triplet_loss), sum(norm_loss), sum(compact_loss), sum(distance_loss)

if __name__ == '__main__':
    pass