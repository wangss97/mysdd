from tkinter.messagebox import NO
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from model.resnet import resnet18, wide_resnet50_2, resnet50
from model.memory import BlockMemory, MemoryUnit
from model.segnet import DiscriminativeSubNetwork
from model.decoder import Decoder, Encoder


class FeatureExtractor(torch.nn.Module):
    def __init__(self) -> None:
        super(FeatureExtractor, self).__init__()
        # self.extractor = resnet18(pretrained=True, progress=True)
        # self.extractor = wide_resnet50_2(pretrained=True, progress=True)
        self.extractor = Encoder(in_channels=3, base_width=64)

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



class model(torch.nn.Module):
    def __init__(self, in_channel=3, out_channel=3, mem_block_list=[1,2,4,8], device='cuda') -> None:
        super(model, self).__init__()

        # self.featureExtractor = wide_resnet50_2(pretrained=True, progress=True)
        # decoder_input_channel = 1024

        # self.featureExtractor = resnet50(pretrained=True, progress=True)
        # decoder_input_channel = 512

        self.featureExtractor = Encoder(in_channels=3, base_width=128)
        decoder_input_channel = 1024

        # self.memory = BlockMemory(block_list = mem_block_list, mem_size_list=[50,50,50,50], fea_dim=decoder_input_channel, pos_ebd_weight = 1., shrink_thres=0.01, device=device)
        self.memory = MemoryUnit(input_size = 16, mem_size = 50, fea_dim = decoder_input_channel, shrink_thres = 0.01, device = device)

        print('mem_block_list:', mem_block_list, len(self.memory.memorys), 'mem_fea_size:',self.memory.memorys[-1].weight.shape)

        # self.decoder = Decoder(in_channel= decoder_input_channel, block_list_size=len(mem_block_list) + 1, out_channel=out_channel)  #拼接未重构的bottleneck
        self.decoder = Decoder(in_channel= decoder_input_channel, block_list_size=len(mem_block_list), out_channel=out_channel)  # 不拼接原始bottleneck

        self.segnet = DiscriminativeSubNetwork(in_channels=6, out_channels=2)

        pass

    def forward(self, input, ori_img, label_batch):
        # b, c, h, w
        # x1, x2, x3, x4 = self.featureExtractor(input)
        x2, x3, x4, _ = self.featureExtractor(input)    # wideRes

        ''' trainable memory bank '''
        entropy_loss, triplet_loss, norm_loss, compact_loss, distance_loss = 0, 0, 0, 0, 0
        # b, c, h, w = x4.shape
        # x4_hat = x4.permute((0,2,3,1)).reshape((-1,c))
        # x4_hat, entropy_loss, triplet_loss, norm_loss, compact_loss, distance_loss = self.memory(x4_hat,label_batch)
        # x4_hat = x4_hat.reshape((b, h, w, c)).permute((0, 3, 1, 2))
        # x4_hat, entropy_loss, triplet_loss, norm_loss, compact_loss, distance_loss = self.memory(x4, label_batch)
        
        
        x1_hat, x2_hat, x3_hat = None, None, None

        ''' 拼接未重构的bottleneck '''
        # x3_hat = torch.concat([x3, x3_hat], dim=1)
        # x4_hat = torch.concat([x4, x4_hat], dim=1)


        output = self.decoder(x1_hat, x2_hat, x3_hat, x4)

        ''' 不使用segNet '''
        # mask = torch.zeros((b,2,h,w))
        ''' 使用segNet '''
        mask = self.segnet(torch.concat([output, ori_img], dim=1))
        
        return output, mask, entropy_loss, triplet_loss, norm_loss, compact_loss, distance_loss

if __name__ == '__main__':
    pass