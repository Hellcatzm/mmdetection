import torch
import numpy as np
import mmcv


def mask_target(pos_proposals_list, pos_assigned_gt_inds_list, gt_masks_list,
                cfg, if_mask_iou=False):
    cfg_list = [cfg for _ in range(len(pos_proposals_list))]
    if if_mask_iou:
        swittch = [True for _ in range(len(pos_proposals_list))]
        mask_targets_total = map(mask_target_single, pos_proposals_list,
                                 pos_assigned_gt_inds_list, gt_masks_list, cfg_list, swittch)

        mask_targets, mask_ratios = [list() for _ in range(2)]
        [(mask_targets.append(item[0]),
          mask_ratios.append(item[1])) for item in mask_targets_total]

        mask_ratios = torch.cat(list(mask_ratios))
        mask_targets = torch.cat(list(mask_targets))
        return mask_targets, mask_ratios
    else:
        mask_targets = map(mask_target_single, pos_proposals_list,
                           pos_assigned_gt_inds_list, gt_masks_list, cfg_list)
        mask_targets = torch.cat(list(mask_targets))
        return mask_targets


def mask_target_single(pos_proposals, pos_assigned_gt_inds, gt_masks, cfg, if_mask_iou=False):
    mask_size = cfg.mask_size
    num_pos = pos_proposals.size(0)
    mask_targets = []
    mask_ratios = []
    if num_pos > 0:
        proposals_np = pos_proposals.cpu().numpy()
        pos_assigned_gt_inds = pos_assigned_gt_inds.cpu().numpy()
        for i in range(num_pos):
            gt_mask = gt_masks[pos_assigned_gt_inds[i]]
            bbox = proposals_np[i, :].astype(np.int32)
            x1, y1, x2, y2 = bbox
            w = np.maximum(x2 - x1 + 1, 1)
            h = np.maximum(y2 - y1 + 1, 1)
            # mask is uint8 both before and after resizing
            target = mmcv.imresize(gt_mask[y1:y1 + h, x1:x1 + w],
                                   (mask_size, mask_size))
            mask_targets.append(target)

            if if_mask_iou:
                gt_area = np.maximum(1, np.sum(gt_mask))
                tar_area = np.sum(gt_mask[y1:y1 + h, x1:x1 + w])
                mask_ratios.append(tar_area / gt_area)

        mask_targets = torch.from_numpy(np.stack(mask_targets)).float().to(
            pos_proposals.device)
        if if_mask_iou:
            mask_ratios = torch.from_numpy(np.stack(mask_ratios)).float().to(
                pos_proposals.device)
    else:
        mask_targets = pos_proposals.new_zeros((0, mask_size, mask_size))

    if if_mask_iou:
        return mask_targets, mask_ratios
    else:
        return mask_targets

