import torch
import torch.nn as nn
from ..registry import LOSSES
from .utils import weighted_loss


@weighted_loss
def generalized_iou_loss(pred, target, mode="GIoU"):
    assert pred.size() == target.size() and target.numel() > 0

    area1 = (pred[:, 2] - pred[:, 0] + 1) * (pred[:, 3] - pred[:, 1] + 1)
    area2 = (target[:, 2] - target[:, 0] + 1) * (target[:, 3] - target[:, 1] + 1)

    overlap_lt = torch.max(pred[:, :2], target[:, :2])  # [n, 2]
    overlap_rb = torch.min(pred[:, 2:], target[:, 2:])
    overlap_wh = (overlap_rb - overlap_lt + 1).clamp(min=0)
    overlap = overlap_wh[:, 0] * overlap_wh[:, 1]

    convex_lt = torch.min(pred[:, :2], target[:, :2])
    convex_rb = torch.max(pred[:, 2:], target[:, 2:])
    convex_wh = (convex_rb - convex_lt + 1).clamp(min=0)
    convex = convex_wh[:, 0] * convex_wh[:, 1]

    unions = area1 + area2 - overlap
    ious = overlap / unions
    if mode == "GIoU":
        gious = ious - (convex - unions) / convex.clamp(min=1e-10)  # [n]
        return 1 - gious  # (1 - gious).sum()
    elif mode == "IoU":
        return 1 - ious
    else:
        raise NotImplementedError


# def weighted_generalized_iou_loss(pred,
#                                   target,
#                                   weight,
#                                   avg_factor=None):
#     """
#
#     :param pred:   [n, 4]
#     :param target: [n, 4]
#     :param weight: [n, 4] 掩码数组，用于区分正负样本（仅正样本贡献reg loss）
#     :param avg_factor: scalar
#     :return:
#     """
#     inds = torch.nonzero(weight[:, 0] > 0)
#     if inds.numel() > 0:
#         inds = inds.squeeze(1)
#     if avg_factor is None:
#         avg_factor = inds.numel() + 1e-6
#     loss = generalized_iou_loss(pred[inds], target[inds])
#
#     return loss[None] / avg_factor


@LOSSES.register_module
class GIoULoss(nn.Module):

    def __init__(self, loss_weight=1.0, mode="GIoU"):
        super(GIoULoss, self).__init__()
        self.mode = mode  # ("GIoU, IoU")
        self.loss_weight = loss_weight

    def forward(self, pred, target, weight, avg_factor=None, **kwargs):
        weight = weight[:, 0]
        loss = self.loss_weight * generalized_iou_loss(pred, target, weight, mode=self.mode, **kwargs)
        return loss
