# 自用改装型mmdetection

本项目基于商汤开源目标检测框架mmdetection，为了适应自己的项目需求，做出了较多修改。<br>
#### 改动概览：<br>
>辅助工具：<br>
>PAN top-bottom path:<br>
>SEResNeXt(with GC module) backbone:<br>
>Mask Scoring head:<br>
```
   mmdet/models/mask_heads/__init__.py
   mmdet/core/mask/mask_target.py
   mmdet/models/mask_heads/htc_mask-scoring_head.py
```
>Soft Dice Loss:<br>
>GIoU Loss:<br>
>Trident RCNN/HTC 相关改动：<br>
>Criss Crioss Attention：<br>
