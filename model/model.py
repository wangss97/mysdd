from tkinter.messagebox import NO
from numpy import False_
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from model.clsnet import ClsSubNet, clsDecoder, clsDecoder2
from model.mff import MFF

from model.resnet import ResNet, resnet18, wide_resnet50_2, resnet50
from model.memory import BlockMemory, MemoryUnit, MemoryUnit_prototype
from model.segnet import DiscriminativeSubNetwork, PatchSegNet, SegDecoder
from model.decoder import Decoder, Encoder
from utils import patch_split


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
        # decoder_input_channel = 2048
        # depth = 5
        # decoder_input_channel = 1024
        # depth = 4

        # expansion = 1

        # self.featureExtractor = resnet50(pretrained=True, progress=True)
        # decoder_input_channel = 512

        self.featureExtractor = Encoder(in_channels=3, base_width=128)
        decoder_input_channel = 1024
        depth = 4
        # expansion = 1

        # self.mff = MFF(expansion=expansion)

        mem_size_list = [50 for i in mem_block_list]
        self.memory = BlockMemory(block_list = mem_block_list, mem_size_list=mem_size_list, fea_dim=decoder_input_channel,
            shrink_thres=0.02, device=device, pos=True, skip=True)

        # self.memory = MemoryUnit(mem_size = 50, fea_dim = decoder_input_channel, shrink_thres = 0.02,pos=True,skip=False, device = device)
        # self.memory = MemoryUnit_prototype(mem_size=50, fea_dim=decoder_input_channel, pos=True, skip=False)


        # self.decoder = Decoder(in_channel= decoder_input_channel, block_list_size=len(mem_block_list) + 1, out_channel=out_channel,depth=depth)  #拼接未重构的bottleneck
        self.decoder = Decoder(in_channel= decoder_input_channel, block_list_size=len(mem_block_list), out_channel=out_channel, depth=depth)  # 不拼接原始bottleneck

        self.segnet = DiscriminativeSubNetwork(in_channels=6, out_channels=2)
        # self.segnet = SegDecoder(in_channel=decoder_input_channel, out_channel=2,block_list_size=len(mem_block_list),depth=depth)
        # self.segnet = PatchSegNet()
        # self.clsnet = ClsSubNet()
        # self.clsDecoder = clsDecoder2()

        pass

    def forward(self, input, ori_img, label_batch):
        # b, c, h, w
        x1, x2, x3, x4, x5 = self.featureExtractor(input)
        # x5_hat = x4          # wideRes 1024
        x5_hat = x5

        ''' trainable memory bank '''
        entropy_loss, triplet_loss, norm_loss, compact_loss, distance_loss, l1_loss \
            = torch.randn(1), torch.randn(1), torch.randn(1), torch.randn(1), torch.randn(1), torch.randn(1)

        x5_hat, entropy_loss, triplet_loss, norm_loss, compact_loss, distance_loss, l1_loss = self.memory(x5_hat, label_batch)
        ''' single Mem '''
        # x5_hat, entropy_loss, triplet_loss, norm_loss, compact_loss, distance_loss = self.memory(x5_hat, label_batch)
        # x5_hat = self.mff([x3,x4,x5])
        ''' single Proto Mem '''
        # x5_hat, compact_loss, distance_loss, l1_loss = self.memory(x5_hat, label_batch)        
        
        x1_hat, x2_hat, x3_hat, x4_hat = None, None, None, None

        x5_dec, x4_dec, x3_dec, x2_dec, x1_dec, output = self.decoder(x5_hat)

        output_detach = output.detach()
        # output_patch = patch_split(output_detach, patch_size=8, stride=8)
        # img_patch = patch_split(ori_img, patch_size=8, stride=8)


        ''' 使用segNet '''
        # mask = self.segnet(torch.concat([output_detach, ori_img], dim=1))
        # mask = self.segnet(x2,x3,x4,x5_hat)
        # label = None
        ''' 使用独立clsNet '''
        # label = self.clsnet(torch.concat([output_detach, ori_img], dim=1))
        ''' 使用clsNet, 共用segNet encoder'''
        mask, label = self.segnet(torch.concat([output_detach, ori_img], dim=1))
        ''' 使用segNet, 使用clsDecoder '''
        # mask, label = self.segnet(torch.concat([output_detach, ori_img], dim=1))
        # label = self.clsDecoder(torch.concat([x5, x5_dec.detach()], dim=1))
        
        return output, mask, label, entropy_loss, triplet_loss, norm_loss, compact_loss, distance_loss, l1_loss

    def freeze_resnet(self):
        # freez full resnet18
        if not isinstance(self.featureExtractor, ResNet):
            print('can not freeze featureExtractor')
            return
        else:
            print('freeze featureExtractor')
            for param in self.featureExtractor.parameters():
                param.requires_grad = False
        
            
    def unfreeze(self):
        #unfreeze all:
        print('unfreeze all parameters')
        for param in self.parameters():
            param.requires_grad = True

if __name__ == '__main__':
    pass