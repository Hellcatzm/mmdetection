# 自用改装型mmdetection(整理中)

本项目基于商汤开源目标检测框架mmdetection，为了适应自己的需求，做出了较多修改。<br>
#### 改动概览：<br>
本部分记录了改动文件,其基础版本为本fork的master分支,后续如果需要更新mmdetection版本的话把下面记录的文件依次放入或酌情修改即可(大概没问题)。<br>
>辅助工具：<br>
```
tools/hook_model.py
tools/subprocess_train.py
TODO 将外部ipython脚本收录进本目录
```
>PAN bottom-up path:<br>
```
mmdet/models/necks/fpn.py
```
>SEResNeXt(with GC module) backbone:<br>
```
mmdet/models/backbones/__init__.py
mmdet/models/backbones/senet_raw.py
mmdet/models/backbones/seresnext.py
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
mmdet/models/losses/giou_loss.py
mmdet/models/detectors/htc 的_bbox_forward_train函数
mmdet/models/core/bbox/bbox_target.py 添加bbox_iou_target_single函数
mmdet/models/bbox_heads/bbox_head 中BBoxHead类获取bbox_target方法调整
```
>Trident RCNN/HTC 相关改动：<br>
```
mmdet/models/backbones/sharedresnet.py
mmdet/models/backbones/sharedresnet_raw.py
mmdet/models/detectors/trident_htc.py
mmdet/models/detectors/trident_rcnn.py
mmdet/models/necks/trident_neck.py
```
>Criss Crioss Attention：<br>
```
mmdet/models/plugins/cc_attention/modules/__init__.py
mmdet/models/plugins/cc_attention/modules/cc_attation.py
mmdet/models/plugins/cc_attention/src/__init__.py
mmdet/models/plugins/cc_attention/src/cca_cuda.cpp
mmdet/models/plugins/cc_attention/src/cca_kernel.cu
mmdet/models/plugins/cc_attention/__init__.py
mmdet/models/plugins/cc_attention/gradcheck.py
mmdet/models/plugins/cc_attention/setup.py
```
