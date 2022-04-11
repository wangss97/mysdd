import imp
import torch
import torch.nn as nn

from torch import Tensor
from typing import List

def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class MFF(torch.nn.Module):
    def __init__(self, expansion=2) -> None:
        super().__init__()

        self.relu = nn.ReLU()

        self.conv1 = conv3x3(512, 1024, 2)
        self.bn1 = nn.BatchNorm2d(1024)
        self.conv2 = conv3x3(in_planes = 1024, out_planes = 1024 * expansion, stride = 2)
        self.bn2 = nn.BatchNorm2d(1024 * expansion)

        self.conv3 = conv3x3(1024, 1024 * expansion, 2)
        self.bn3 = nn.BatchNorm2d(1024 * expansion)

        
        self.conv4 = conv1x1(1024 * expansion * 3, 1024 * expansion, 1)
        self.bn4 = nn.BatchNorm2d(1024 * expansion)


        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _forward_impl(self, x: List[Tensor]) -> Tensor:
        
        l1 = self.relu(self.bn2(self.conv2(self.relu(self.bn1(self.conv1(x[0]))))))
        l2 = self.relu(self.bn3(self.conv3(x[1])))
        feature = torch.concat([l1,l2,x[2]],1)

        output = self.relu(self.bn4(self.conv4(feature)))

        return output.contiguous()

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)