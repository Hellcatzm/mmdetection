# author: hellcatzm
# data:   2019/7/13
from torch.autograd import gradcheck

import os.path as osp
import sys
sys.path.append(osp.abspath(osp.join(__file__, '../../')))
from cc_attention import CrissCrossAttention, ca_weight, ca_map

import torch
t = torch.rand(2, 32, 5, 5, requires_grad=True).cuda()  # dtype=torch.float64
inputs = (t,)
print('Gradcheck for roi pooling...')
test = gradcheck(CrissCrossAttention(32).cuda(), inputs, eps=1e-5, atol=1e-3)
print(test)
