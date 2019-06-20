import math

import torch.nn as nn
import torch.utils.checkpoint as cp
from mmdet.ops import DeformConv, ModulatedDeformConv
from .resnet import Bottleneck as _Bottleneck
from .resnet import ResNet
from ..registry import BACKBONES
from ..utils import build_conv_layer, build_norm_layer


class SEModule(nn.Module):
    def __init__(self, channels, ratio, conv_cfg=None):
        super(SEModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv1 = build_conv_layer(
            conv_cfg,
            channels,
            int(channels * ratio),
            kernel_size=1,
            padding=0)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = build_conv_layer(
            conv_cfg,
            int(channels * ratio),
            channels,
            kernel_size=1,
            padding=0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        identity = x
        x = self.avg_pool(x)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.sigmoid(x)
        return identity * x


class Bottleneck(_Bottleneck):

    def __init__(self, *args, groups=1, base_width=4, **kwargs):
        """Bottleneck block for ResNeXt.
        If style is "pytorch", the stride-two layer is the 3x3 conv layer,
        if it is "caffe", the stride-two layer is the first 1x1 conv layer.
        """
        super(Bottleneck, self).__init__(*args, **kwargs)

        self.se_module = SEModule(self.planes * self.expansion, ratio=16)

        if groups == 1:
            width = self.planes
        else:
            width = math.floor(self.planes * (base_width / 64)) * groups

        self.norm1_name, norm1 = build_norm_layer(
            self.norm_cfg, width, postfix=1)
        self.norm2_name, norm2 = build_norm_layer(
            self.norm_cfg, width, postfix=2)
        self.norm3_name, norm3 = build_norm_layer(
            self.norm_cfg, self.planes * self.expansion, postfix=3)

        self.conv1 = build_conv_layer(
            self.conv_cfg,
            self.inplanes,
            width,
            kernel_size=1,
            stride=self.conv1_stride,
            bias=False)
        self.add_module(self.norm1_name, norm1)
        fallback_on_stride = False
        self.with_modulated_dcn = False
        if self.with_dcn:
            fallback_on_stride = self.dcn.get('fallback_on_stride', False)
            self.with_modulated_dcn = self.dcn.get('modulated', False)
        if not self.with_dcn or fallback_on_stride:
            self.conv2 = build_conv_layer(
                self.conv_cfg,
                width,
                width,
                kernel_size=3,
                stride=self.conv2_stride,
                padding=self.dilation,
                dilation=self.dilation,
                groups=groups,
                bias=False)
        else:
            assert self.conv_cfg is None, 'conv_cfg must be None for DCN'
            groups = self.dcn.get('groups', 1)
            deformable_groups = self.dcn.get('deformable_groups', 1)
            if not self.with_modulated_dcn:
                conv_op = DeformConv
                offset_channels = 18
            else:
                conv_op = ModulatedDeformConv
                offset_channels = 27
            self.conv2_offset = nn.Conv2d(
                width,
                deformable_groups * offset_channels,
                kernel_size=3,
                stride=self.conv2_stride,
                padding=self.dilation,
                dilation=self.dilation)
            self.conv2 = conv_op(
                width,
                width,
                kernel_size=3,
                stride=self.conv2_stride,
                padding=self.dilation,
                dilation=self.dilation,
                groups=groups,
                deformable_groups=deformable_groups,
                bias=False)
        self.add_module(self.norm2_name, norm2)
        self.conv3 = build_conv_layer(
            self.conv_cfg,
            width,
            self.planes * self.expansion,
            kernel_size=1,
            bias=False)
        self.add_module(self.norm3_name, norm3)

    def forward(self, x):

        def _inner_forward(x):
            identity = x

            out_ = self.conv1(x)
            out_ = self.norm1(out_)
            out_ = self.relu(out_)

            if not self.with_dcn:
                out_ = self.conv2(out_)
            elif self.with_modulated_dcn:
                offset_mask = self.conv2_offset(out_)
                offset = offset_mask[:, :18, :, :]
                mask = offset_mask[:, -9:, :, :].sigmoid()
                out_ = self.conv2(out_, offset, mask)
            else:
                offset = self.conv2_offset(out_)
                out_ = self.conv2(out_, offset)
            out_ = self.norm2(out_)
            out_ = self.relu(out_)

            out_ = self.conv3(out_)
            out_ = self.norm3(out_)

            if self.downsample is not None:
                identity = self.downsample(x)

            out_ = self.se_module(out_) + identity

            return out_

        if self.with_cp and x.requires_grad:
            out = cp.checkpoint(_inner_forward, x)
        else:
            out = _inner_forward(x)
        out = self.relu(out)

        return out


def make_res_layer(block,
                   inplanes,
                   planes,
                   blocks,
                   stride=1,
                   dilation=1,
                   groups=1,
                   base_width=4,
                   style='pytorch',
                   with_cp=False,
                   conv_cfg=None,
                   norm_cfg=dict(type='BN'),
                   dcn=None):
    downsample = None
    if stride != 1 or inplanes != planes * block.expansion:
        downsample = nn.Sequential(
            build_conv_layer(
                conv_cfg,
                inplanes,
                planes * block.expansion,
                kernel_size=1,
                stride=stride,
                bias=False),
            build_norm_layer(norm_cfg, planes * block.expansion)[1],
        )

    layers = list()
    layers.append(
        block(
            inplanes,
            planes,
            stride=stride,
            dilation=dilation,
            downsample=downsample,
            groups=groups,
            base_width=base_width,
            style=style,
            with_cp=with_cp,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            dcn=dcn))
    inplanes = planes * block.expansion
    for i in range(1, blocks):
        layers.append(
            block(
                inplanes,
                planes,
                stride=1,
                dilation=dilation,
                groups=groups,
                base_width=base_width,
                style=style,
                with_cp=with_cp,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                dcn=dcn))

    return nn.Sequential(*layers)


@BACKBONES.register_module
class SEResNeXt(ResNet):
    """ResNeXt backbone.

    Args:
        depth (int): Depth of resnet, from {18, 34, 50, 101, 152}.
        num_stages (int): Resnet stages, normally 4.
        groups (int): Group of resnext.
        base_width (int): Base width of resnext.
        strides (Sequence[int]): Strides of the first block of each stage.
        dilations (Sequence[int]): Dilation of each stage.
        out_indices (Sequence[int]): Output from which stages.
        style (str): `pytorch` or `caffe`. If set to "pytorch", the stride-two
            layer is the 3x3 conv layer, otherwise the stride-two layer is
            the first 1x1 conv layer.
        frozen_stages (int): Stages to be frozen (all param fixed). -1 means
            not freezing any parameters.
        norm_cfg (dict): dictionary to construct and config norm layer.
        norm_eval (bool): Whether to set norm layers to eval mode, namely,
            freeze running stats (mean and var). Note: Effect on Batch Norm
            and its variants only.
        with_cp (bool): Use checkpoint or not. Using checkpoint will save some
            memory while slowing down the training speed.
        zero_init_residual (bool): whether to use zero init for last norm layer
            in resblocks to let them behave as identity.
    """

    arch_settings = {
        50: (Bottleneck, (3, 4, 6, 3)),
        101: (Bottleneck, (3, 4, 23, 3)),
        152: (Bottleneck, (3, 8, 36, 3))
    }

    def __init__(self, groups=1, base_width=4, **kwargs):
        super(SEResNeXt, self).__init__(**kwargs)
        self.groups = groups
        self.base_width = base_width

        self.inplanes = 64
        self.res_layers = []
        for i, num_blocks in enumerate(self.stage_blocks):
            stride = self.strides[i]
            dilation = self.dilations[i]
            dcn = self.dcn if self.stage_with_dcn[i] else None
            planes = 64 * 2**i
            res_layer = make_res_layer(
                self.block,
                self.inplanes,
                planes,
                num_blocks,
                stride=stride,
                dilation=dilation,
                groups=self.groups,
                base_width=self.base_width,
                style=self.style,
                with_cp=self.with_cp,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                dcn=dcn)
            self.inplanes = planes * self.block.expansion
            layer_name = 'layer{}'.format(i + 1)
            self.add_module(layer_name, res_layer)
            self.res_layers.append(layer_name)

        self._freeze_stages()
