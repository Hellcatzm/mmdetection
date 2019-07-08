import os
import mmcv
import torch
import numpy as np

from mmcv.runner import load_checkpoint
from mmdet.models import build_detector
from mmdet.apis import init_detector, inference_detector
from mmdet.datasets import get_dataset, build_dataloader
import matplotlib.pyplot as plt
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"


def forward_hook(module, data_input, data_output):
    """register_forward_hook(hook)"""
    print(data_input.data.shape)
    print(data_output.data.shape)


def backward_hook(module, grad_input, grad_output):
    """register_backward_hook(hook)"""
    print(grad_input.data.shape)
    print(grad_output.data.shape)


HOOT_MODE = "train"  # "inference" or "train"
ROOT_DIR = '/home/gttintern/mmdetection'
CONFIG_NAME = 'configs/carbonate/trident/faster_rcnn_r50_fpn_1x.py'

config_file = os.path.join(ROOT_DIR, CONFIG_NAME)
cfg = mmcv.Config.fromfile(config_file)

model = build_detector(
        cfg.model, train_cfg=cfg.train_cfg, test_cfg=cfg.test_cfg).cuda()
# checkpoint_file = os.path.join(os.path.join(ROOT_DIR, cfg.work_dir), 'latest.pth')
# load_checkpoint(model, checkpoint_file, map_location='cpu')

cfg.data.train.ann_file = os.path.join(ROOT_DIR,
                                       cfg.data.train.ann_file)
cfg.data.train.img_prefix = os.path.join(ROOT_DIR,
                                         cfg.data.train.img_prefix)
# cfg.data.train.seg_prefix = os.path.join(ROOT_DIR,
#                                          cfg.data.train.seg_prefix)
dataset = get_dataset(cfg.data.train)
dataloader = build_dataloader(
    dataset,
    imgs_per_gpu=1,
    workers_per_gpu=cfg.data.workers_per_gpu,
    num_gpus=1,
    dist=False)
model.CLASSES = dataset.CLASSES
batch_data = next(iter(dataloader))


if HOOT_MODE == "inference":
    model.eval()
    #_____________________________________________________________________
    """
    在感兴趣的层注册钩子查看数据流
    """
    # _____________________________________________________________________
    with torch.no_grad():
        result = model(return_loss=False,
                       rescale=False,
                       img=[batch_data['img'].data[0].cuda()],
                       img_meta=[batch_data['img_meta'].data[0]],
                       )
    plt.imshow(batch_data['img'].data[0][0][1])
    # for i in range(10):
    #     plt.plot(result[0][0][i][[0,2]],result[0][0][i][[1,3]])
elif HOOT_MODE == "train":
    # _____________________________________________________________________
    """
    在感兴趣的层注册钩子查看数据流
    """
    # _____________________________________________________________________
    for i, batch_data in enumerate(iter(dataloader)):
        losses = model(img=batch_data['img'].data[0].cuda(),
                       img_meta=batch_data['img_meta'].data[0],
                       gt_bboxes=[t.cuda() for t in batch_data['gt_bboxes'].data[0]],
                       gt_labels=[t.cuda() for t in batch_data['gt_labels'].data[0]],
                       gt_bboxes_ignore=batch_data['gt_bboxes_ignore'].data[0],
                       # gt_masks=[t for t in batch_data['gt_masks'].data[0]],  # 传入numpy数组即可
                       # gt_semantic_seg=batch_data['gt_semantic_seg'].data[0].cuda()
                       )
        break

    ms_test_mode = ["ms_target", "ms_head", None][2]
    # 如需调试，记得取消htc_mask_scoring_head的98、135行注释
    if ms_test_mode == "ms_target":
        # htc_mask_scoring_head 测试
        mask_pred = model.mask_head[-1].mask_pred.detach().cpu().numpy()
        mask_targets = model.mask_head[-1].mask_targets.cpu().numpy()
        labels = model.mask_head[-1].labels.cpu().numpy()

        mask_pred = mask_pred[range(mask_pred.shape[0]), labels]  # [n_pos_roi, h, w]
        mask_pred = mask_pred > 0

        mask_ovr = mask_pred * mask_targets
        mask_union = np.logical_or(mask_pred, mask_targets)

        inds = 56  # 挑选一个roi可视化
        print("网络标签iou：",model.mask_head[-1].mask_iou_targets[inds])
        print("网络预测iou：", model.mask_head[-1].mask_iou[inds])
        print("外部计算iou：%.4f" % (mask_ovr[inds].sum() / mask_union[inds].sum()))
        plt.subplot(2,2,1)
        plt.imshow(mask_pred[inds])
        plt.subplot(2, 2, 2)
        plt.imshow(mask_targets[inds])
        plt.subplot(2, 2, 3)
        plt.imshow(mask_ovr[inds])
        plt.subplot(2, 2, 4)
        plt.imshow(mask_union[inds])
    elif ms_test_mode == "ms_head":
        mask_pred = model.mask_head[-1].mask_pred.detach()
        mask_targets = model.mask_head[-1].mask_targets
        labels = model.mask_head[-1].labels

        num_rois = mask_pred.size()[0]
        inds = torch.arange(0, num_rois, dtype=torch.long, device=mask_pred.device)
        mask_pred = mask_pred[inds, labels]                   # [n_pos_roi, h, w]

        pred_slice_pooled = mask_pred[:, None, :, :]  # [n_pos_roi, 1, h/2, w/2]
        mask_iou = model.mask_head[-1].mask_iou_head(model.mask_head[-1].roi_feat, pred_slice_pooled)
        mask_iou = mask_iou.squeeze()
        print(mask_iou[:, labels])
    else:
        pass
