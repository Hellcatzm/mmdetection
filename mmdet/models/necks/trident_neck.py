# author: hellcatzm
# data:   2019/6/28
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import xavier_init

from ..registry import NECKS
from ..utils import ConvModule


@NECKS.register_module
class TridentNeck(nn.Module):
    def __init__(self,
                 fpn_in_channels=1024,
                 rcnn_in_channels=2048,
                 fpn_out_channels=256,
                 rcnn_out_channels=256,
                 if_conv=True):
        super(TridentNeck, self).__init__()
        self.if_conv = if_conv
        if if_conv:
            self.rpn_neck = nn.Conv2d(fpn_in_channels, fpn_out_channels, kernel_size=1)
            self.rcnn_neck = nn.Conv2d(rcnn_in_channels, rcnn_out_channels, kernel_size=1)

    def init_weights(self):
        if not self.if_conv:
            return
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m, distribution='uniform')

    def forward(self, x):
        rpn_input, rcnn_input = x
        if self.if_conv:
            return self.rpn_neck(rpn_input), self.rcnn_neck(rcnn_input)
        return rpn_input, rcnn_input