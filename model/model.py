from tkinter.messagebox import NO
from numpy import False_
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from model.clsnet import ClsSubNet, clsDecoder, clsDecoder2, clsNet2
from model.deeplab.deeplab_model import DeepLab
from model.mff import MFF

from model.resnet import ResNet, resnet18, wide_resnet50_2, resnet50
from model.memory import BlockMemory, MemoryUnit, MemoryUnit_prototype
from model.segnet import DiscriminativeSubNetwork, PatchSegNet, SegDecoder
from model.decoder import Decoder, Encoder
from model.loss import ssim
from utils import patch_split, visualize
import numpy as np

class model(torch.nn.Module):
    def __init__(self, in_channel=3, out_channel=3, mem_block_list=[8,16], rec_depth=4, seg_depth=5, device='cuda') -> None:
        super(model, self).__init__()

        # self.featureExtractor = wide_resnet50_2(pretrained=True, progress=True)
        # decoder_input_channel = 2048
        # depth = 5
        # decoder_input_channel = 1024
        # depth = 4

        # expansion = 1

        # self.featureExtractor = resnet50(pretrained=True, progress=True)
        # decoder_input_channel = 512

        self.featureExtractor = Encoder(in_channels=in_channel, base_width=128, depth=rec_depth)
        decoder_input_channel = 128 * (2**(rec_depth-2))
        # decoder_input_channel = 1024

        mem_size_list = [50 for i in mem_block_list]
        self.memory = BlockMemory(block_list = mem_block_list, mem_size_list=mem_size_list, fea_dim=decoder_input_channel,
            shrink_thres=0.02, device=device, pos=True, skip=True)


        self.decoder = Decoder(in_channel=decoder_input_channel, block_list_size=len(mem_block_list), out_channel=out_channel, base_width=128, depth=rec_depth)  # 不拼接原始bottleneck

        self.segnet = DiscriminativeSubNetwork(in_channels=2*in_channel, out_channels=2, depth = seg_depth)

        # self.segnet = DeepLab(backbone='xception', output_stride=8, num_classes=2, sync_bn=False, freeze_bn=False)

        self.detach = False

        pass

    def forward(self, input, ori_img):
        # b, c, h, w
        x5 = self.featureExtractor(input)

        ''' trainable memory bank '''
        x5_hat, compact_loss, distance_loss = self.memory(x5)
        # compact_loss = torch.tensor([1])
        # distance_loss = torch.tensor([1])
        # x5_hat = x5
        
        output = self.decoder(x5_hat)
        
        if self.detach:
            output_detach = output.detach()
        else:
            output_detach = output
        ''' 使用segNet '''
        mask = self.segnet(torch.concat([output_detach, ori_img], dim=1))

        return output, mask, compact_loss, distance_loss

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

    def detach_on(self):
        print('detach on')
        self.detach = True
    
    def detach_off(self):
        print('detach off')
        self.detach = False

if __name__ == '__main__':
    pass