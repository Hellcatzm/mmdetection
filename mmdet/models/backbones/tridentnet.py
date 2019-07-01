import logging

import torch
import torch.nn as nn
import torch.utils.checkpoint as cp
from torch.nn.modules.batchnorm import _BatchNorm

from mmcv.cnn import constant_init, kaiming_init
from mmcv.runner import load_checkpoint
from mmdet.ops import DeformConv, ModulatedDeformConv, ContextBlock
from mmdet.models.plugins import GeneralizedAttention

from ..utils import build_conv_layer, build_norm_layer, SharedConv, SharedDeformConv, SharedBN
from ..registry import BACKBONES


class SharedBottleneck(nn.Module):

    expansion = 4

    def __init__(self,
                 inplanes,
                 planes,
                 stride=(1, 1, 1),
                 dilate=(1, 2, 3),
                 downsample=None,
                 with_cp=False,
                 norm_cfg=dict(type='BN'),
                 dcn=None):
        super(SharedBottleneck, self).__init__()
        assert dcn is None or isinstance(dcn, dict)
        self.with_cp = with_cp
        self.with_dcn = dcn is not None
        if norm_cfg.get('type', 'BN') == 'BN':
            normalizer = nn.BatchNorm2d
        else:
            normalizer = nn.SyncBatchNorm

        self.conv1 = SharedConv(inplanes=inplanes, planes=planes,
                                stride=1, dilate=1, kernel_size=1, with_bias=False)
        self.norm1 = SharedBN(inplanes=planes, normalizer=normalizer)

        fallback_on_stride = False
        self.with_modulated_dcn = False
        if self.with_dcn:
            fallback_on_stride = dcn.get('fallback_on_stride', False)
            self.with_modulated_dcn = dcn.get('modulated', False)
        if not self.with_dcn or fallback_on_stride:
            self.conv2 = SharedConv(inplanes=planes, planes=planes,
                                    stride=stride, dilate=dilate, pad=dilate, kernel_size=3, with_bias=False)
        else:
            deformable_groups = dcn.get('deformable_groups', 1)
            self.conv2 = SharedDeformConv(inplanes=inplanes, planes=planes,
                                          stride=stride, dilate=dilate, pad=dilate,
                                          kernel_size=3, deformable_groups=deformable_groups)

        self.norm2 = SharedBN(inplanes=planes, normalizer=normalizer)
        self.conv3 = SharedConv(inplanes=planes, planes=planes*self.expansion,
                                stride=1, dilate=1, kernel_size=1, with_bias=False)
        self.norm3 = SharedBN(inplanes=planes*self.expansion, normalizer=normalizer)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        if self.training:
            def _inner_forward(x):
                identity = x
                out_ = self.conv1(x)
                out_ = self.norm1(out_)
                out_ = [self.relu(out_[i]) for i in range(len(out_))]
                out_ = self.conv2(out_)
                out_ = self.norm2(out_)
                out_ = [self.relu(out_[i]) for i in range(len(out_))]
                out_ = self.conv3(out_)
                out_ = self.norm3(out_)
                if self.downsample is not None:
                    identity = self.downsample(x)
                out_ = [out_[i] + identity[i] for i in range(len(out_))]
                return out_
        else:
            def _inner_forward(x):
                identity = x
                out_ = self.conv1(x)
                out_ = self.norm1(out_)
                out_ = self.relu(out_)
                out_ = self.conv2(out_)
                out_ = self.norm2(out_)
                out_ = self.relu(out_)
                out_ = self.conv3(out_)
                out_ = self.norm3(out_)
                if self.downsample is not None:
                    identity = self.downsample(x)
                out_ = out_ + identity
                return out_
        if self.with_cp and x.requires_grad:
            out = cp.checkpoint(_inner_forward, x)
        else:
            out = _inner_forward(x)

        if self.training:
            return [self.relu(out[i]) for i in range(len(out))]
        else:
            return self.relu(out)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self,
                 inplanes,
                 planes,
                 stride=1,
                 dilation=1,
                 downsample=None,
                 style='pytorch',
                 with_cp=False,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 dcn=None,
                 gcb=None,
                 gen_attention=None):
        """Bottleneck block for ResNet.
        If style is "pytorch", the stride-two layer is the 3x3 conv layer,
        if it is "caffe", the stride-two layer is the first 1x1 conv layer.
        """
        super(Bottleneck, self).__init__()
        assert style in ['pytorch', 'caffe']
        assert dcn is None or isinstance(dcn, dict)
        assert gcb is None or isinstance(gcb, dict)
        assert gen_attention is None or isinstance(gen_attention, dict)

        self.inplanes = inplanes
        self.planes = planes
        self.stride = stride
        self.dilation = dilation
        self.style = style
        self.with_cp = with_cp
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.dcn = dcn
        self.with_dcn = dcn is not None
        self.gcb = gcb
        self.with_gcb = gcb is not None
        self.gen_attention = gen_attention
        self.with_gen_attention = gen_attention is not None

        if self.style == 'pytorch':
            self.conv1_stride = 1
            self.conv2_stride = stride
        else:
            self.conv1_stride = stride
            self.conv2_stride = 1

        self.norm1_name, norm1 = build_norm_layer(norm_cfg, planes, postfix=1)
        self.norm2_name, norm2 = build_norm_layer(norm_cfg, planes, postfix=2)
        self.norm3_name, norm3 = build_norm_layer(
            norm_cfg, planes * self.expansion, postfix=3)

        self.conv1 = build_conv_layer(
            conv_cfg,
            inplanes,
            planes,
            kernel_size=1,
            stride=self.conv1_stride,
            bias=False)
        self.add_module(self.norm1_name, norm1)
        fallback_on_stride = False
        self.with_modulated_dcn = False
        if self.with_dcn:
            fallback_on_stride = dcn.get('fallback_on_stride', False)
            self.with_modulated_dcn = dcn.get('modulated', False)
        if not self.with_dcn or fallback_on_stride:
            self.conv2 = build_conv_layer(
                conv_cfg,
                planes,
                planes,
                kernel_size=3,
                stride=self.conv2_stride,
                padding=dilation,
                dilation=dilation,
                bias=False)
        else:
            assert conv_cfg is None, 'conv_cfg must be None for DCN'
            deformable_groups = dcn.get('deformable_groups', 1)
            if not self.with_modulated_dcn:
                conv_op = DeformConv
                offset_channels = 18
            else:
                conv_op = ModulatedDeformConv
                offset_channels = 27
            self.conv2_offset = nn.Conv2d(
                planes,
                deformable_groups * offset_channels,
                kernel_size=3,
                stride=self.conv2_stride,
                padding=dilation,
                dilation=dilation)
            self.conv2 = conv_op(
                planes,
                planes,
                kernel_size=3,
                stride=self.conv2_stride,
                padding=dilation,
                dilation=dilation,
                deformable_groups=deformable_groups,
                bias=False)
        self.add_module(self.norm2_name, norm2)
        self.conv3 = build_conv_layer(
            conv_cfg,
            planes,
            planes * self.expansion,
            kernel_size=1,
            bias=False)
        self.add_module(self.norm3_name, norm3)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

        if self.with_gcb:
            gcb_inplanes = planes * self.expansion
            self.context_block = ContextBlock(inplanes=gcb_inplanes, **gcb)

        # gen_attention
        if self.with_gen_attention:
            self.gen_attention_block = GeneralizedAttention(
                planes, **gen_attention)

    @property
    def norm1(self):
        return getattr(self, self.norm1_name)

    @property
    def norm2(self):
        return getattr(self, self.norm2_name)

    @property
    def norm3(self):
        return getattr(self, self.norm3_name)

    def forward(self, x):

        def _inner_forward(x):
            identity = x

            out = self.conv1(x)
            out = self.norm1(out)
            out = self.relu(out)

            if not self.with_dcn:
                out = self.conv2(out)
            elif self.with_modulated_dcn:
                offset_mask = self.conv2_offset(out)
                offset = offset_mask[:, :18, :, :]
                mask = offset_mask[:, -9:, :, :].sigmoid()
                out = self.conv2(out, offset, mask)
            else:
                offset = self.conv2_offset(out)
                out = self.conv2(out, offset)
            out = self.norm2(out)
            out = self.relu(out)

            if self.with_gen_attention:
                out = self.gen_attention_block(out)

            out = self.conv3(out)
            out = self.norm3(out)

            if self.with_gcb:
                out = self.context_block(out)

            if self.downsample is not None:
                identity = self.downsample(x)

            out += identity

            return out

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
                   style='pytorch',
                   with_cp=False,
                   conv_cfg=None,
                   norm_cfg=dict(type='BN'),
                   dcn=None,
                   gcb=None,
                   gen_attention=None,
                   gen_attention_blocks=[]):
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

    layers = []
    layers.append(
        block(
            inplanes=inplanes,
            planes=planes,
            stride=stride,
            dilation=dilation,
            downsample=downsample,
            style=style,
            with_cp=with_cp,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            dcn=dcn,
            gcb=gcb,
            gen_attention=gen_attention if
            (0 in gen_attention_blocks) else None))
    inplanes = planes * block.expansion
    for i in range(1, blocks):
        layers.append(
            block(
                inplanes=inplanes,
                planes=planes,
                stride=1,
                dilation=dilation,
                style=style,
                with_cp=with_cp,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                dcn=dcn,
                gcb=gcb,
                gen_attention=gen_attention if
                (i in gen_attention_blocks) else None))

    return nn.Sequential(*layers)


def make_trid_res_layer(block,
                        inplanes,
                        planes,
                        blocks,
                        stride=1,
                        dilation=1,
                        norm_cfg=dict(type='BN'),
                        dcn=None):
    downsample = None
    if stride != 1 or inplanes != planes * block.expansion:
        downsample = nn.Sequential(
            SharedConv(
                inplanes,
                planes * block.expansion,
                kernel_size=1,
                stride=stride,
                with_bias=False),
            SharedBN(planes * block.expansion),
        )
    layers = []
    layers.append(
        block(
            inplanes,
            planes,
            stride,
            dilation,
            downsample,
            norm_cfg=norm_cfg,
            dcn=dcn))
    inplanes = planes * block.expansion
    for i in range(1, blocks):
        layers.append(
            block(
                inplanes,
                planes,
                1,
                dilation,
                norm_cfg=norm_cfg,
                dcn=dcn))
    return nn.Sequential(*layers)


@BACKBONES.register_module
class TridentResNet(nn.Module):

    arch_settings = {
        50: (3, 4, 6, 3),
        101: (3, 4, 23, 3),
        152: (3, 8, 36, 3)
    }

    def __init__(self,
                 depth,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN', requires_grad=True),
                 norm_eval=False,
                 frozen_stages=-1,
                 **kwargs):
        super(TridentResNet, self).__init__()
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.norm_eval = norm_eval
        self.frozen_stages = frozen_stages

        self.c1 = self._resnet_c1()  # [n, 64, h, w]
        self.c2 = self._resnet(block=Bottleneck,
                               inplanes=64,
                               planes=64*2**0,
                               blocks=self.arch_settings[depth][0],
                               conv_cfg=conv_cfg,
                               norm_cfg=norm_cfg,
                               **kwargs)
        self.c3 = self._resnet(block=Bottleneck,
                               inplanes=64*2**0*Bottleneck.expansion,
                               planes=64*2**1,
                               blocks=self.arch_settings[depth][1],
                               conv_cfg=conv_cfg,
                               norm_cfg=norm_cfg,
                               **kwargs)
        self.c4 = self._resnet_trident(block=SharedBottleneck,
                                       inplanes=64*2**1*SharedBottleneck.expansion,
                                       planes=64*2**2,
                                       blocks=self.arch_settings[depth][2],
                                       norm_cfg=norm_cfg)
        self.c5 = self._resnet(Bottleneck,
                               64*2**2*Bottleneck.expansion,
                               64*2**3,
                               blocks=self.arch_settings[depth][3],
                               conv_cfg=conv_cfg,
                               norm_cfg=norm_cfg,
                               **kwargs)

    def _resnet_c1(self):
        return nn.Sequential(
            build_conv_layer(
                self.conv_cfg,
                3,
                64,
                kernel_size=7,
                stride=2,
                padding=3,
                bias=False),
            build_norm_layer(self.norm_cfg, 64, postfix=1)[1],
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

    def _resnet(self, block, inplanes, planes, blocks, **kwargs):
        return make_res_layer(block, inplanes, planes, blocks, **kwargs)

    def _resnet_trident(self, block, inplanes, planes, blocks, norm_cfg):
        return make_trid_res_layer(block,
                                   inplanes,
                                   planes,
                                   blocks,
                                   norm_cfg=norm_cfg)

    def forward(self, x):
        if self.training:
            c1_f = self.c1(x)
            c2_f = self.c2(c1_f)
            c3_f = self.c3(c2_f)
            c3_fs = [c3_f for _ in range(3)]
            c4_fs = self.c4(c3_fs)

            c4_shape = list(c4_fs[0].shape)
            c4_shape[0] *= 3
            c4_f = torch.stack(c4_fs, dim=1)
            c4_f = c4_f.view(c4_shape)

            c5_f = self.c5(c4_f)

            outs = [c4_f, c5_f]
            return outs
        else:
            c1_f = self.c1(x)
            c2_f = self.c2(c1_f)
            c3_f = self.c3(c2_f)
            c4_f = self.c4(c3_f)
            c5_f = self.c5(c4_f)

            outs = [c4_f, c5_f]
            return outs

    def train(self, mode=True):
        super(TridentResNet, self).train(mode)
        self._freeze_stages()
        if mode and self.norm_eval:
            for m in self.modules():
                # trick: eval have effect on BatchNorm only
                if isinstance(m, _BatchNorm):
                    m.eval()

    def _freeze_stages(self):
        for i in range(self.frozen_stages + 1):
            m = getattr(self, 'c{}'.format(i))
            m.eval()
            for param in m.parameters():
                param.requires_grad = False

    def init_weights(self, pretrained=None):
        if isinstance(pretrained, str):
            logger = logging.getLogger()
            load_checkpoint(self, pretrained, strict=False, logger=logger)
        elif pretrained is None:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    kaiming_init(m)
                elif isinstance(m, (_BatchNorm, nn.GroupNorm)):
                    constant_init(m, 1)
