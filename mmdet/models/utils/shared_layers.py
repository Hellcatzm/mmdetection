# author: hellcatzm
# data:   2019/6/24
import torch
import torch.nn as nn
from mmdet.ops import DeformConv, ModulatedDeformConv


class SharedConv(nn.Module):
    def __init__(self,
                 inplanes,
                 planes,
                 kernel_size=3,
                 with_bias=True,
                 paths=3,
                 stride=(1, 1, 1),
                 dilate=(1, 2, 3),
                 pad=(0, 0, 0),
                 share_weight=True):
        super(SharedConv, self).__init__()
        self.shared_conv = nn.ModuleList()
        for i in range(paths):
            stride_i = stride[i] if isinstance(stride, (list, tuple)) else stride
            dilate_i = dilate[i] if isinstance(dilate, (list, tuple)) else dilate
            pad_i = pad[i] if isinstance(pad, (list, tuple)) else pad
            conv_i = nn.Conv2d(inplanes,
                               planes,
                               kernel_size,
                               bias=with_bias,
                               stride=stride_i,
                               dilation=dilate_i,
                               padding=pad_i)
            if share_weight:
                if i == 0:
                    self.weight, self.bias = conv_i.weight, conv_i.bias
                else:
                    conv_i.weight, conv_i.bias = self.weight, self.bias
            self.shared_conv.append(conv_i)

    def forward(self, x):
        if self.training:
            assert isinstance(x, (list, tuple)), "input should be list of Tensor"
            out = list()
            for x_i, conv_i in zip(x, self.shared_conv):
                feat_i = conv_i(x_i)
                out.append(feat_i)
            return out
        else:
            assert isinstance(x, torch.Tensor), "input should be Tensor"
            return self.shared_conv[1](x)


class SharedBN(nn.Module):
    def __init__(self,
                 inplanes,
                 normalizer=nn.BatchNorm2d,
                 paths=3,
                 share_weight=True):
        super(SharedBN, self).__init__()
        self.shared_norm = nn.ModuleList()
        for i in range(paths):
            norm_i = normalizer(inplanes)
            if normalizer == nn.SyncBatchNorm:
                norm_i._specify_ddp_gpu_num(1)
            if share_weight:
                if i == 0:
                    self.weight, self.bias = norm_i.weight, norm_i.bias
                    running_mean, running_var = norm_i.running_mean, norm_i.running_var
                else:
                    norm_i.weight, norm_i.bias = self.weight, self.bias
                    norm_i.running_mean, norm_i.running_var = running_mean, running_var
            self.shared_norm.append(norm_i)

    def forward(self, x):
        if self.training:
            assert isinstance(x, (list, tuple)), "input should be list of Tensor"
            out = list()
            for x_i, norm_i in zip(x, self.shared_norm):
                feat_i = norm_i(x_i)
                out.append(feat_i)
            return out
        else:
            assert isinstance(x, torch.Tensor), "input should be Tensor"
            self.shared_norm[1].training=self.training
            return self.shared_norm[1](x)


class SharedDeformConv(nn.Module):
    def __init__(self,
                 inplanes,
                 planes,
                 kernel_size=3,
                 conv_op=DeformConv,
                 paths=3,
                 stride=(1, 1, 1),
                 dilate=(1, 2, 3),
                 pad=(0, 0, 0),
                 deformable_groups=4,
                 share_weight=True):
        super(SharedDeformConv, self).__init__()
        self.shared_conv = nn.ModuleList()
        self.offset_conv = nn.ModuleList()

        if conv_op == DeformConv:
            offset_channels = 18
        elif conv_op == ModulatedDeformConv:
            offset_channels = 27
        else:
            assert TypeError, "conv_op must be DeformConv or ModulatedDeformConv"

        for i in range(paths):
            stride_i = stride[i] if isinstance(stride, (list, tuple)) else stride
            dilate_i = dilate[i] if isinstance(dilate, (list, tuple)) else dilate
            pad_i = pad[i] if isinstance(pad, (list, tuple)) else pad
            offset_conv_i = nn.Conv2d(inplanes,
                                      deformable_groups * offset_channels,
                                      kernel_size=3,
                                      stride=stride_i,
                                      dilation=dilate_i,
                                      padding=pad_i,)
            conv_i = conv_op(inplanes,
                             planes,
                             kernel_size,
                             bias=False,
                             stride=stride_i,
                             dilation=dilate_i,
                             padding=pad_i,
                             deformable_groups=deformable_groups)
            if share_weight:
                if i == 0:
                    self.weight = conv_i.weight
                else:
                    conv_i.weight = self.weight
            self.shared_conv.append(conv_i)
            self.offset_conv.append(offset_conv_i)

    def forward(self, x):
        if self.training:
            assert isinstance(x, (list, tuple)), "input should be list of Tensor"
            out = list()
            for x_i, conv_i, offset_conv_i in zip(x, self.shared_conv, self.offset_conv):
                if isinstance(conv_i, ModulatedDeformConv):
                    offset_mask = offset_conv_i(x_i)
                    offset = offset_mask[:, :18, :, :]
                    mask = offset_mask[:, -9:, :, :].sigmoid()
                    feat_i = conv_i(x_i, offset, mask)
                else:
                    offset = offset_conv_i(x_i)
                    feat_i = conv_i(x_i, offset)
                out.append(feat_i)
            return out
        else:
            assert isinstance(x, torch.Tensor), "input should be Tensor"
            if isinstance(self.shared_conv[1], ModulatedDeformConv):
                offset_mask = self.offset_conv[1](x)
                offset = offset_mask[:, :18, :, :]
                mask = offset_mask[:, -9:, :, :].sigmoid()
                feat = self.shared_conv[1](x, offset, mask)
            else:
                offset = self.offset_conv[1](x)
                feat = self.shared_conv[1](x, offset)
            return feat
