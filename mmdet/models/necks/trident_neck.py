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
                 rpn_neck=True,
                 rcnn_neck=True):
        super(TridentNeck, self).__init__()
        self.rpn_neck = rpn_neck
        self.rcnn_neck = rcnn_neck
        if rpn_neck:
            self.rpn_neck = nn.Conv2d(fpn_in_channels, fpn_out_channels, kernel_size=1)
        if rcnn_neck:
            self.rcnn_neck = nn.Conv2d(rcnn_in_channels, rcnn_out_channels, kernel_size=1)

    def init_weights(self):
        if (not self.rpn_neck) and (not self.rcnn_neck):
            return
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m, distribution='uniform')

    def forward(self, x):
        if len(x)==2:
            rpn_input, rcnn_input = x
        elif len(x)==1:
            rpn_input = rcnn_input = x[0]
        if self.rpn_neck:
            rpn_feat = self.rpn_neck(rpn_input)
        else:
            rpn_feat = rpn_input
        if self.rcnn_neck:
            rcnn_feat = self.rcnn_neck(rcnn_input)
        else:
            rcnn_feat = rcnn_input
        return rpn_feat, rcnn_feat