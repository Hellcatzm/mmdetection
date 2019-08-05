# 自用改装型mmdetection(整理中)

本项目基于商汤开源目标检测框架mmdetection，为了适应自己的需求，做出了较多修改。<br>
#### 改动概览：<br>
本部分记录了改动文件,其基础版本为本fork的master分支,后续如果需要更新mmdetection版本的话把下面记录的文件依次放入或酌情修改即可(大概没问题)。<br>
>数据class及增强：<br>
```
mmdet/models/datasets/carbonate.py
```
>辅助工具：<br>
```
tools/hook_model.py              模拟train/inference过程,调bug用
tools/subprocess_train.py        自动训练多个配置文件
TODO 将外部ipython辅助脚本收录进本目录
```
>PAN bottom-up path:<br>
```
mmdet/models/necks/fpn.py       简单的为FPN添加了一个可选路径
```
>SEResNeXt(with GC module) backbone:<br>
```
mmdet/models/backbones/__init__.py
mmdet/models/backbones/senet_raw.py   包含了SENet154/SEResNet/SEResNeXt,没有预训练文件故废弃
mmdet/models/backbones/seresnext.py   基于ResNeXt的SEResNeXt,有GCN选项,可载人ResNeXt的预训练文件
```
>Mask Scoring head:<br>
```
mmdet/models/mask_heads/__init__.py
mmdet/core/mask/mask_target.py
mmdet/models/mask_heads/htc_mask-scoring_head.py
```
>Soft Dice Loss:<br>
```
mmdet/models/losses/soft_dice_loss.py
```
>GIoU Loss:<br>
```
mmdet/models/losses/__init__.py
mmdet/models/losses/giou_loss.py            损失函数本体
mmdet/models/detectors/htc.py               _bbox_forward_train函数修改
mmdet/models/core/bbox/bbox_target.py       添加bbox_iou_target_single函数
mmdet/models/bbox_heads/bbox_head.py        BBoxHead类获取bbox_target方法调整
```
>Trident RCNN/HTC 相关改动：<br>
```
mmdet/models/backbones/sharedresnet.py      基于ResNet的trident backbone,可载人ResNet的预训练文件
mmdet/models/backbones/sharedresnet_raw.py  废弃的trident backbone,调bug方便
mmdet/models/detectors/trident_htc.py       trident htc检测器
mmdet/models/detectors/trident_rcnn.py      trident rcnn检测器
mmdet/models/necks/trident_neck.py          trident的特殊neck,应对显存不够为特征降维
mmdet/ops/shared_layers/__init__.py
mmdet/ops/shared_layers/shared_layers.py    共享卷积/DCN/BN实现
```
>Criss Crioss Attention：<br>
```
mmdet/models/plugins/cc_attention/modules/__init__.py
mmdet/models/plugins/cc_attention/modules/cc_attation.py   封装好的pytorch cca module节点
mmdet/models/plugins/cc_attention/src/__init__.py
mmdet/models/plugins/cc_attention/src/cca_cuda.cpp         cuda核函数
mmdet/models/plugins/cc_attention/src/cca_kernel.cu        forward/backward函数C++封装
mmdet/models/plugins/cc_attention/__init__.py
mmdet/models/plugins/cc_attention/gradcheck.py             梯度检查脚本,直接运行即可
mmdet/models/plugins/cc_attention/setup.py                 setup脚本,编译好mmdetection后需要单独编译本脚本
```
