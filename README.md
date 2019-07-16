# 自用改装型mmdetection(整理中)

本项目基于商汤开源目标检测框架mmdetection，为了适应自己的项目需求，做出了较多修改。<br>
#### 改动概览：<br>
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
>GIoU Loss:<br>
```
mmdet/models/losses/__init__.py
mmdet/models/losses/giou_loss.py
mmdet/models/detectors/htc 的_bbox_forward_train函数
mmdet/models/core/bbox/bbox_target.py 添加bbox_iou_target_single函数
mmdet/models/bbox_heads/bbox_head 中BBoxHead类获取bbox_target方法调整
```
>Trident RCNN/HTC 相关改动：<br>
>Criss Crioss Attention：<br>
