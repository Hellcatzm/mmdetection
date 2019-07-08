# author: hellcatzm
# data:   2019/7/6

import math
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from mmdet.ops import DeformConv, ModulatedDeformConv


class SharedBatchNorm(nn.Module):
    def __init__(self, num_features, paths=3, normalizer=nn.SyncBatchNorm,):
        super(SharedBatchNorm, self).__init__()
        self.weight = Parameter(torch.empty(num_features))
        self.bias = Parameter(torch.empty(num_features))
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))
        self.init_weights()

        for i in range(paths):
            norm_i = normalizer(num_features)
            if normalizer == nn.SyncBatchNorm:
                norm_i._specify_ddp_gpu_num(1)
            norm_i.weight = self.weight
            norm_i.bias = self.bias
            norm_i.running_mean = self.running_mean
            norm_i.running_var = self.running_var
            layer_name = 'branch_{}'.format(i + 1)
            self.add_module(layer_name, norm_i)

    def init_weights(self):
        nn.init.uniform_(self.weight)
        nn.init.zeros_(self.bias)

    def forward(self, x):
        if isinstance(x, (list, tuple)):
            norm = list()
            for i in range(len(x)):
                # bn_i = nn.functional.batch_norm(input=x[i],
                #                                 running_mean=self.running_mean,
                #                                 running_var=self.running_var,
                #                                 weight=self.weight,
                #                                 bias=self.bias,
                #                                 training=self.training)
                # norm.append(bn_i)
                branch_i = getattr(self, 'branch_{}'.format(i + 1))
                norm_i = branch_i(x[i])
                norm.append(norm_i)

        elif isinstance(x, torch.Tensor):
            # norm = nn.functional.batch_norm(input=x,
            #                                 running_mean=self.running_mean,
            #                                 running_var=self.running_var,
            #                                 weight=self.weight,
            #                                 bias=self.bias,
            #                                 training=self.training)
            branch = getattr(self, 'branch_{}'.format(1 + 1))
            norm = branch(x)
        else:
            raise NotImplementedError

        return norm


class SharedConv2d(nn.Module):
    def __init__(self,
                 in_channels, out_channels, kernel_size,
                 stride=(1, 1, 1), dilation=(1, 2, 3), padding=(0, 0, 0), groups=1, bias=True):
        super(SharedConv2d, self).__init__()
        if in_channels % groups != 0:
            raise ValueError('in_channels must be divisible by groups')
        if out_channels % groups != 0:
            raise ValueError('out_channels must be divisible by groups')
        self.stride = stride if isinstance(stride , (list, tuple)) else [stride]*3
        self.groups = groups
        self.dilation = dilation
        self.padding = padding
        self.weight = Parameter(torch.empty(out_channels, in_channels // groups, kernel_size, kernel_size))
        if bias:
            self.bias = Parameter(torch.empty(out_channels))
        else:
            self.register_parameter('bias', None)
        self.init_weights()

    def init_weights(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x, branch=1):
        if isinstance(x, (list, tuple)):
            conv = list()
            for i in range(len(x)):
                conv_i = nn.functional.conv2d(input=x[i],
                                              weight=self.weight,
                                              bias=self.bias,
                                              stride=self.stride[i],
                                              padding=self.padding[i],
                                              dilation=self.dilation[i],
                                              groups=self.groups)
                conv.append(conv_i)
        elif isinstance(x, torch.Tensor):
            conv = nn.functional.conv2d(input=x,
                                          weight=self.weight,
                                          bias=self.bias,
                                          stride=self.stride[branch],
                                          padding=self.padding[branch],
                                          dilation=self.dilation[branch],
                                          groups=self.groups)
        else:
            raise NotImplementedError

        return conv


class SharedDeformConv2d(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 conv_op=DeformConv,
                 paths=3,
                 stride=(1, 1, 1),
                 dilation=(1, 2, 3),
                 padding=(1, 2, 3),
                 groups=1,
                 deformable_groups=1):
        super(SharedDeformConv2d, self).__init__()
        self.shared_conv = nn.ModuleList()
        self.offset_conv = nn.ModuleList()
        if conv_op == DeformConv:
            offset_channels = 18
        elif conv_op == ModulatedDeformConv:
            offset_channels = 27
        else:
            assert TypeError, "conv_op must be DeformConv or ModulatedDeformConv"
        # self.weight = Parameter(torch.empty(out_channels, in_channels, kernel_size, kernel_size))

        for i in range(paths):
            stride_i = stride[i] if isinstance(stride, (list, tuple)) else stride
            dilate_i = dilation[i] if isinstance(dilation, (list, tuple)) else dilation
            pad_i = padding[i] if isinstance(padding, (list, tuple)) else padding
            offset_conv_i = nn.Conv2d(in_channels,
                                      deformable_groups * offset_channels,
                                      kernel_size=kernel_size,
                                      stride=stride_i,
                                      dilation=dilate_i,
                                      padding=pad_i,)
            conv_i = conv_op(in_channels,
                             out_channels,
                             kernel_size=kernel_size,
                             bias=False,
                             stride=stride_i,
                             dilation=dilate_i,
                             padding=pad_i,
                             groups=groups,
                             deformable_groups=deformable_groups)

            if i == 0:
                self.register_parameter('weight', conv_i.weight)
            else:
                conv_i.weight = self.weight
            self.shared_conv.append(conv_i)
            self.offset_conv.append(offset_conv_i)
        self.init_weights()

    def init_weights(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        for offset_conv_i in self.offset_conv:
            nn.init.kaiming_uniform_(offset_conv_i.weight, a=math.sqrt(5))
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(offset_conv_i.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(offset_conv_i.bias, -bound, bound)

    def forward(self, x, branch=1):
        if isinstance(x, (list, tuple)):
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
        elif isinstance(x, torch.Tensor):
            if isinstance(self.shared_conv[branch], ModulatedDeformConv):
                offset_mask = self.offset_conv[branch](x)
                offset = offset_mask[:, :18, :, :]
                mask = offset_mask[:, -9:, :, :].sigmoid()
                feat = self.shared_conv[branch](x, offset, mask)
            else:
                offset = self.offset_conv[branch](x)
                feat = self.shared_conv[branch](x, offset)
            return feat


#
#
# class PyramidBatchNorm(nn.Module):
#     def __init__(self, num_features, normalizer=nn.BatchNorm2d, paths=3):
#         super(PyramidBatchNorm, self).__init__()
#         for i in range(paths):
#             norm_i = normalizer(num_features)
#             if normalizer == nn.SyncBatchNorm:
#                 norm_i._specify_ddp_gpu_num(1)
#             layer_name = 'branch_{}'.format(i + 1)
#             self.add_module(layer_name, norm_i)
#
#     def forward(self, x):
#         if isinstance(x, (list, tuple)):
#             norm = list()
#             for i in range(len(x)):
#                 branch_i = getattr(self, 'branch_{}'.format(i + 1))
#                 norm_i = branch_i(x[i])
#                 norm.append(norm_i)
#         else:
#             branch = getattr(self, 'branch_{}'.format(2))
#             norm = branch(x)
#
#         return norm
#
#
# class PyramidConv2d(nn.Module):
#     def __init__(self,
#                  in_channels, out_channels, kernel_size,
#                  stride=(1, 1, 1), dilation=(1, 2, 3), padding=(0, 0, 0), groups=1, bias=True, paths=3):
#         super(PyramidConv2d, self).__init__()
#         if in_channels % groups != 0:
#             raise ValueError('in_channels must be divisible by groups')
#         if out_channels % groups != 0:
#             raise ValueError('out_channels must be divisible by groups')
#         stride = stride if isinstance(stride, (list, tuple)) else [stride] * 3
#         for i in range(paths):
#             conv_i = nn.Conv2d(in_channels, out_channels, kernel_size,
#                                stride=stride[i], dilation=dilation[i], padding=padding[i], groups=groups, bias=bias)
#             layer_name = 'branch_{}'.format(i + 1)
#             self.add_module(layer_name, conv_i)
#
#     def forward(self, x):
#         if isinstance(x, (list, tuple)):
#             conv = list()
#             for i in range(len(x)):
#                 branch_i = getattr(self, 'branch_{}'.format(i + 1))
#                 conv_i = branch_i(x[i])
#                 conv.append(conv_i)
#         else:
#             branch = getattr(self, 'branch_{}'.format(2))
#             conv = branch(x)
#
#         return conv