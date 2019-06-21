import torch
import torch.nn as nn

from ..registry import LOSSES


def soft_dice_loss(pred, target, label, reduction='mean', avg_factor=None):
    """
    using for mask loss
    :param pred:   [n_rois, n_cls, h, w]
    :param target: [n_rois, h, w]
    :param label:  [n_rois]
    :return:
    """
    assert reduction == 'mean' and avg_factor is None
    num_rois = pred.size()[0]
    inds = torch.arange(0, num_rois, dtype=torch.long, device=pred.device)
    pred_slice = pred[inds, label].squeeze(1)

    ovr = (pred_slice * target).sum(dim=[1, 2])
    union = (pred_slice**2 + target**2).sum(dim=[1, 2]).clamp(1e-5)
    return (1 - ovr*2 / union).mean()[None]


@LOSSES.register_module
class SoftDiceLoss(nn.Module):
    def __init__(self, loss_weight=1.0):
        super(SoftDiceLoss, self).__init__()
        self.loss_weight = loss_weight

    def forward(self, cls_score, label, weight=None, avg_factor=None,
                **kwargs):
        return self.loss_weight * soft_dice_loss(cls_score, label, weight)

