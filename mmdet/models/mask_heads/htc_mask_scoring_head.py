# https://github.com/zjhuang22/maskscoring_rcnn
import torch
import torch.nn as nn
from torch.nn import functional as F

from .htc_mask_head import HTCMaskHead
from ..registry import HEADS
from ..utils import ConvModule
from mmdet.core import mask_target


def l2_loss(input, target):
    """
    very similar to the smooth_l1_loss from pytorch, but with
    the extra beta parameter
    """
    pos_inds = torch.nonzero(target > 0.0).squeeze(1)
    if pos_inds.shape[0] > 0:
        cond = torch.abs(input[pos_inds] - target[pos_inds])
        loss = 0.5 * cond**2 / pos_inds.shape[0]
    else:
        loss = input * 0.0
    return loss.sum()


class MSIoUFeatureExtractor(nn.Module):
    def __init__(self, num_classes):
        super(MSIoUFeatureExtractor, self).__init__()
        self.conv1 = nn.Conv2d(257, 256, 3, 1, 1)
        self.conv2 = nn.Conv2d(256, 256, 3, 1, 1)
        self.conv3 = nn.Conv2d(256, 256, 3, 1, 1)
        self.conv4 = nn.Conv2d(256, 256, 3, 2, 1)
        self.fc1 = nn.Linear(256*7*7, 1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.fc3 = nn.Linear(1024, num_classes)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                nn.init.constant_(m.bias, 0)
            if m in (self.fc1, self.fc2):
                nn.init.kaiming_normal_(m.weight, a=1)
                nn.init.constant(m.bias, 0)
        nn.init.normal_(self.fc3.weight, mean=0, std=0.01)
        nn.init.constant_(self.fc3.bias, 0)

    def forward(self, x, mask):
        mask_pool = F.max_pool2d(mask, kernel_size=2, stride=2)
        x = torch.cat((x, mask_pool), dim=1)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))

        return x


@HEADS.register_module
class HTCMaskScoringHead(HTCMaskHead):

    def __init__(self, *arg, **kwargs):
        super(HTCMaskScoringHead, self).__init__(*arg, **kwargs)
        self.roi_feat = None
        self.mask_ratios = None
        self.mask_iou_head = MSIoUFeatureExtractor(self.num_classes)

    def init_weights(self):
        super(HTCMaskScoringHead, self).init_weights()
        self.mask_iou_head.init_weights()

    def forward(self, x, *arg, **kwargs):
        self.roi_feat = x
        return super(HTCMaskScoringHead, self).forward(x, *arg, **kwargs)

    def get_target(self, sampling_results, gt_masks, rcnn_train_cfg):
        pos_proposals = [res.pos_bboxes for res in sampling_results]
        pos_assigned_gt_inds = [res.pos_assigned_gt_inds for res in sampling_results]
        mask_targets, mask_ratios = mask_target(pos_proposals, pos_assigned_gt_inds,
                                                gt_masks, rcnn_train_cfg, if_mask_iou=True)
        self.mask_ratios = mask_ratios
        return mask_targets  # denote only positive propose will be fed in mask head

    def loss(self, mask_pred, mask_targets, labels):
        """

        :param mask_pred:    [n_pos_roi, n_cls, h, w]
        :param mask_targets: [n_pos_roi, h, w], {0, 1}
        :param labels:       [n_pos_roi], int
        :return:
        """
        assert self.roi_feat is not None, 'must forward once before compute loss'
        assert self.mask_ratios is not None, 'must get_target once before compute loss'
        self.mask_pred = mask_pred
        self.mask_targets = mask_targets
        self.labels =labels

        loss = dict()
        if self.class_agnostic:
            loss_mask = self.loss_mask(mask_pred, mask_targets,
                                       torch.zeros_like(labels))
        else:
            loss_mask = self.loss_mask(mask_pred, mask_targets, labels)
        loss['loss_mask'] = loss_mask

        num_rois = mask_pred.size()[0]
        inds = torch.arange(0, num_rois, dtype=torch.long, device=mask_pred.device)
        mask_pred = mask_pred[inds, labels]                   # [n_pos_roi, h, w]

        pred_slice_pooled = mask_pred[:, None, :, :]  # [n_pos_roi, 1, h/2, w/2]
        mask_iou = self.mask_iou_head(self.roi_feat, pred_slice_pooled)  # [n_pos_roi, n_cls, 1, 1]
        mask_iou = mask_iou.squeeze()       # [n_pos_roi, n_cls]
        mask_iou = mask_iou[inds, labels]   # [n_pos_roi]

        # mask_iou为bbox中的交集 / 整个图像中的并集
        mask_pred = (mask_pred > 0).float()  # a[:] = b is a's inplace change
        mask_ovr = mask_pred * mask_targets
        mask_ovr_area = mask_ovr.sum(dim=[1, 2])
        mask_targets_full_area = mask_targets.sum(dim=[1, 2]) / self.mask_ratios
        mask_union_area = mask_pred.sum(dim=[1, 2]) + mask_targets_full_area - mask_ovr_area

        # value_0 = torch.zeros(mask_pred.shape[0], device=labels.device)
        # value_1 = torch.ones(mask_pred.shape[0], device=labels.device)
        # mask_ovr_area = torch.max(mask_ovr_area, value_0)
        # mask_union_area = torch.max(mask_union_area, value_1)
        mask_ovr_area = mask_ovr_area.clamp(min=0)
        mask_union_area = mask_union_area.clamp(min=1)
        mask_iou_targets = mask_ovr_area / mask_union_area
        mask_iou_targets = mask_iou_targets.detach()  # [n_pos_roi]

        self.mask_iou = mask_iou
        self.mask_iou_targets = mask_iou_targets
        loss_mask_iou = l2_loss(mask_iou, mask_iou_targets)
        loss['loss_mask_iou'] = loss_mask_iou

        return loss

