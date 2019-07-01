# author: hellcatzm
# data:   2019/6/28
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import xavier_init

from ..registry import NECKS
from ..utils import ConvModule


@NECKS.register_module
class TridentNeck(nn.Module):
    def __init__(self, fpn_in_channels, rcnn_in_channels, out_channels):
        super(TridentNeck, self).__init__()
        self.rpn_neck = nn.Conv2d(fpn_in_channels, out_channels, kernel_size=1)
        self.rcnn_neck = nn.Conv2d(rcnn_in_channels, out_channels, kernel_size=1)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m, distribution='uniform')

    def forward(self, rpn_input, rcnn_input):
        return self.rpn_neck(rpn_input), self.rcnn_neck(rcnn_input)