# author: hellcatzm
# data:   2019/7/6

import math
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter


class SharedBatchNorm(nn.Module):
    def __init__(self, num_features):
        super(SharedBatchNorm, self).__init__()
        self.weight = Parameter(torch.empty(num_features))
        self.bias = Parameter(torch.empty(num_features))
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))

    def init_weights(self):
        nn.init.uniform_(self.weight)
        nn.init.zeros_(self.bias)

    def forward(self, data):
        if isinstance(data, (list, tuple)):
            bn = list()
            for i in range(len(data)):
                bn_i = nn.functional.batch_norm(input=data[i],
                                                running_mean=self.running_mean,
                                                running_var=self.running_var,
                                                weight=self.weight,
                                                bias=self.bias,
                                                training=self.training)
                bn.append(bn_i)
        elif isinstance(data, torch.Tensor):
            bn = nn.functional.batch_norm(input=data,
                                          running_mean=self.running_mean,
                                          running_var=self.running_var,
                                          weight=self.weight,
                                          bias=self.bias,
                                          training=self.training)
        else:
            raise NotImplementedError

        return bn


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

    def init_weights(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, data, shared=True):
        if isinstance(data, (list, tuple)):
            conv = list()
            for i in range(len(data)):
                conv_i = nn.functional.conv2d(input=data[i],
                                              weight=self.weight,
                                              bias=self.bias,
                                              stride=self.stride[i],
                                              padding=self.padding[i],
                                              dilation=self.dilation[i],
                                              groups=self.groups)
                conv.append(conv_i)
        elif isinstance(data, torch.Tensor):
            conv = nn.functional.conv2d(input=data,
                                          weight=self.weight,
                                          bias=self.bias,
                                          stride=self.stride[1],
                                          padding=self.padding[1],
                                          dilation=self.dilation[1],
                                          groups=self.groups)
        else:
            raise NotImplementedError

        return conv


class PyramidBatchNorm(nn.Module):
    def __init__(self, num_features, normalizer=nn.BatchNorm2d, paths=3):
        super(PyramidBatchNorm, self).__init__()
        for i in range(paths):
            norm_i = normalizer(num_features)
            if normalizer == nn.SyncBatchNorm:
                norm_i._specify_ddp_gpu_num(1)
            layer_name = 'branch_{}'.format(i + 1)
            self.add_module(layer_name, norm_i)

    def forward(self, data):
        if isinstance(data, (list, tuple)):
            norm = list()
            for i in range(len(data)):
                branch_i = getattr(self, 'branch_{}'.format(i + 1))
                norm_i = branch_i(data[i])
                norm.append(norm_i)
        else:
            branch = getattr(self, 'branch_{}'.format(2))
            norm = branch(data)

        return norm


class PyramidConv2d(nn.Module):
    def __init__(self,
                 in_channels, out_channels, kernel_size,
                 stride=(1, 1, 1), dilation=(1, 2, 3), padding=(0, 0, 0), groups=1, bias=True, paths=3):
        super(PyramidConv2d, self).__init__()
        if in_channels % groups != 0:
            raise ValueError('in_channels must be divisible by groups')
        if out_channels % groups != 0:
            raise ValueError('out_channels must be divisible by groups')
        stride = stride if isinstance(stride, (list, tuple)) else [stride] * 3
        for i in range(paths):
            conv_i = nn.Conv2d(in_channels, out_channels, kernel_size,
                               stride=stride[i], dilation=dilation[i], padding=padding[i], groups=groups, bias=bias)
            layer_name = 'branch_{}'.format(i + 1)
            self.add_module(layer_name, conv_i)

    def forward(self, data):
        if isinstance(data, (list, tuple)):
            conv = list()
            for i in range(len(data)):
                branch_i = getattr(self, 'branch_{}'.format(i + 1))
                conv_i = branch_i(data[i])
                conv.append(conv_i)
        else:
            branch = getattr(self, 'branch_{}'.format(2))
            conv = branch(data)

        return conv