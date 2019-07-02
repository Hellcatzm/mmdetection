# author: hellcatzm
# data:   2019/6/28
import torch

from ..registry import DETECTORS
from .htc import HybridTaskCascade
from mmdet.core import (bbox2roi, bbox2result, build_assigner, build_sampler,
                        merge_aug_masks)
from functools import reduce


@DETECTORS.register_module
class TridentRCNN(HybridTaskCascade):

    def __init__(self, val_range=((0, 10), (5, 15), (15, -1)), **kwargs):
        super(TridentRCNN, self).__init__(**kwargs)
        self.val_range = val_range

    def extract_feat(self, img):
        x = self.backbone(img)
        return self.neck(*x)

    def forward_train(self,
                      img,
                      img_meta,   # [n, dict]
                      gt_bboxes,  # [n, [gts, 4]]
                      gt_labels,  # [n, [gts]]
                      gt_bboxes_ignore=None,
                      gt_masks=None,  # [n, np.array[gts, h, w]]
                      gt_semantic_seg=None,
                      proposals=None):
        rpn_feat, rcnn_feat = self.extract_feat(img)  # [3n, 256, h, w]
        self.sr = []
        # convert data from dim imgs to dim branches*imgs
        if not isinstance(gt_bboxes_ignore, (list, tuple)):
            gt_bboxes_ignore = [None]

        img_meta_ = list()
        gt_bboxes_ = list()
        gt_labels_ = list()
        gt_masks_ = list()
        gt_bboxes_ignore_ = list()
        val_index = torch.zeros(rpn_feat.shape[0])
        val_pointer = -1
        for img_num in range(img.size(0)):
            for lo, hi in self.val_range:
                val_pointer += 1
                bboxes_hw = gt_bboxes[img_num][:, 2:] - gt_bboxes[img_num][:, :2]
                boxes_area = (bboxes_hw[:, 0] * bboxes_hw[:, 1]).clamp(min=0)
                if hi >= 0:
                    boxes_index = (boxes_area > lo ** 2) & (boxes_area < hi ** 2)
                else:
                    boxes_index = boxes_area > lo ** 2
                if boxes_index.int().sum():
                    val_index[val_pointer] = 1
                    arg_idx = torch.nonzero(boxes_index).squeeze()
                    img_meta_.append(img_meta[img_num])
                    gt_bboxes_.append(gt_bboxes[img_num][arg_idx])  # [3n, [gts_v, 4]]
                    gt_labels_.append(gt_labels[img_num][arg_idx])  # [3n, [gts_v]]
                    gt_masks_.append(gt_masks[img_num][arg_idx.cpu()])
                    gt_bboxes_ignore_.append(gt_bboxes_ignore[img_num])
                    if arg_idx.numel() == 1:
                        gt_bboxes_[-1].unsqueeze_(dim=0)
                        gt_labels_[-1].unsqueeze_(dim=0)
                        gt_masks_[-1] = gt_masks_[-1][None]
        idx = torch.nonzero(val_index).squeeze()
        rpn_feat = rpn_feat[idx][None]
        rcnn_feat = rcnn_feat[idx][None]
        if idx.numel() == 1:
            rpn_feat = rpn_feat[None]
            rcnn_feat = rcnn_feat[None]

        # forward as normal
        losses = dict()

        # RPN part, the same as normal two-stage detectors
        assert self.with_rpn, "Trident Net must have RPN head"

        rpn_outs = self.rpn_head(rpn_feat)
        # [list: num_levels, [3n, anchors*2, h, w]]
        # [list: num_levels, [3n, anchors*4, h, w]]

        rpn_loss_inputs = rpn_outs + (gt_bboxes_, img_meta_, self.train_cfg.rpn)
        self.rpn = rpn_loss_inputs
        rpn_losses = self.rpn_head.loss(
            *rpn_loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore_)

        losses.update(rpn_losses)

        proposal_cfg = self.train_cfg.get('rpn_proposal',
                                          self.test_cfg.rpn)
        proposal_inputs = rpn_outs + (img_meta_, proposal_cfg)
        proposal_list = self.rpn_head.get_bboxes(*proposal_inputs)  # [list(num imgs) [2000(配置文件指定), 5]]

        assert not self.with_semantic, "Not supply yet"
        semantic_feat = None

        num_imgs = len(img_meta_)
        for i in range(self.num_stages):
            self.current_stage = i
            rcnn_train_cfg = self.train_cfg.rcnn[i]
            lw = self.train_cfg.stage_loss_weights[i]

            # assign gts and sample proposals
            sampling_results = []
            bbox_assigner = build_assigner(rcnn_train_cfg.assigner)
            bbox_sampler = build_sampler(rcnn_train_cfg.sampler, context=self)

            for j in range(num_imgs):
                assign_result = bbox_assigner.assign(
                    proposal_list[j],
                    gt_bboxes_[j],
                    gt_bboxes_ignore_[j],
                    gt_labels_[j])
                sampling_result = bbox_sampler.sample(
                    assign_result,
                    proposal_list[j],
                    gt_bboxes_[j],
                    gt_labels_[j],
                    feats=rcnn_feat)
                sampling_results.append(sampling_result)

            # bbox head forward and loss
            loss_bbox, rois, bbox_targets, bbox_pred = \
                self._bbox_forward_train(
                    i, rcnn_feat, sampling_results, gt_bboxes_, gt_labels_,
                    rcnn_train_cfg, semantic_feat)
            roi_labels = bbox_targets[0]

            for name, value in loss_bbox.items():
                losses['s{}.{}'.format(i, name)] = (
                    value * lw if 'loss' in name else value)

            # mask head forward and loss
            if self.with_mask:
                # interleaved execution: use regressed bboxes by the box branch
                # to train the mask branch
                if self.interleaved:
                    pos_is_gts = [res.pos_is_gt for res in sampling_results]
                    with torch.no_grad():
                        proposal_list = self.bbox_head[i].refine_bboxes(
                            rois, roi_labels, bbox_pred, pos_is_gts, img_meta_)
                        # re-assign and sample 512 RoIs from 512 RoIs
                        sampling_results = []
                        for j in range(num_imgs):
                            assign_result = bbox_assigner.assign(
                                proposal_list[j],
                                gt_bboxes_[j],
                                gt_bboxes_ignore_[j],
                                gt_labels_[j])
                            sampling_result = bbox_sampler.sample(
                                assign_result,
                                proposal_list[j],
                                gt_bboxes_[j],
                                gt_labels_[j],
                                feats=rcnn_feat)
                            sampling_results.append(sampling_result)
                self.sr.append(sampling_results)
                loss_mask = self._mask_forward_train(i, rcnn_feat, sampling_results,
                                                     gt_masks_, rcnn_train_cfg,
                                                     semantic_feat)
                for name, value in loss_mask.items():
                    losses['s{}.{}'.format(i, name)] = (
                        value * lw if 'loss' in name else value)

            # refine bboxes (same as Cascade R-CNN)
            if i < self.num_stages - 1 and not self.interleaved:
                pos_is_gts = [res.pos_is_gt for res in sampling_results]
                with torch.no_grad():
                    proposal_list = self.bbox_head[i].refine_bboxes(
                        rois, roi_labels, bbox_pred, pos_is_gts, img_meta_)

        return losses

    def simple_test(self, img, img_meta, proposals=None, rescale=False):
        rpn_feat, rcnn_feat = self.extract_feat(img)  # [n, 256, h, w], [n, 256, h, w]
        proposal_list = self.simple_test_rpn(
            rpn_feat[None], img_meta, self.test_cfg.rpn) if proposals is None else proposals

        assert not self.with_semantic, "Not supply yet"
        semantic_feat = None

        img_shape = img_meta[0]['img_shape']
        ori_shape = img_meta[0]['ori_shape']
        scale_factor = img_meta[0]['scale_factor']

        # "ms" in variable names means multi-stage
        ms_bbox_result = {}
        ms_segm_result = {}
        ms_scores = []
        rcnn_test_cfg = self.test_cfg.rcnn

        rois = bbox2roi(proposal_list)
        for i in range(self.num_stages):
            bbox_head = self.bbox_head[i]
            cls_score, bbox_pred = self._bbox_forward_test(
                i, rcnn_feat[None], rois, semantic_feat=semantic_feat)
            ms_scores.append(cls_score)

            if self.test_cfg.keep_all_stages:
                det_bboxes, det_labels = bbox_head.get_det_bboxes(
                    rois,
                    cls_score,
                    bbox_pred,
                    img_shape,
                    scale_factor,
                    rescale=rescale,
                    nms_cfg=rcnn_test_cfg)
                bbox_result = bbox2result(det_bboxes, det_labels,
                                          bbox_head.num_classes)
                ms_bbox_result['stage{}'.format(i)] = bbox_result

                if self.with_mask:
                    mask_head = self.mask_head[i]
                    if det_bboxes.shape[0] == 0:
                        segm_result = [
                            [] for _ in range(mask_head.num_classes - 1)
                        ]
                    else:
                        _bboxes = (
                            det_bboxes[:, :4] * scale_factor
                            if rescale else det_bboxes)
                        mask_pred = self._mask_forward_test(
                            i, rcnn_feat[None], _bboxes, semantic_feat=semantic_feat)
                        segm_result = mask_head.get_seg_masks(
                            mask_pred, _bboxes, det_labels, rcnn_test_cfg,
                            ori_shape, scale_factor, rescale)
                    ms_segm_result['stage{}'.format(i)] = segm_result

            if i < self.num_stages - 1:
                bbox_label = cls_score.argmax(dim=1)
                rois = bbox_head.regress_by_class(rois, bbox_label, bbox_pred,
                                                  img_meta[0])

        cls_score = sum(ms_scores) / float(len(ms_scores))
        det_bboxes, det_labels = self.bbox_head[-1].get_det_bboxes(
            rois,
            cls_score,
            bbox_pred,
            img_shape,
            scale_factor,
            rescale=rescale,
            cfg=rcnn_test_cfg)
        bbox_result = bbox2result(det_bboxes, det_labels,
                                  self.bbox_head[-1].num_classes)
        ms_bbox_result['ensemble'] = bbox_result

        if self.with_mask:
            if det_bboxes.shape[0] == 0:
                segm_result = [
                    [] for _ in range(self.mask_head[-1].num_classes - 1)
                ]
            else:
                _bboxes = (
                    det_bboxes[:, :4] * scale_factor
                    if rescale else det_bboxes)

                mask_rois = bbox2roi([_bboxes])
                aug_masks = []
                mask_roi_extractor = self.mask_roi_extractor[-1]
                mask_feats = mask_roi_extractor(
                    rcnn_feat[None][:len(mask_roi_extractor.featmap_strides)], mask_rois)
                if self.with_semantic and 'mask' in self.semantic_fusion:
                    mask_semantic_feat = self.semantic_roi_extractor(
                        [semantic_feat], mask_rois)
                    mask_feats += mask_semantic_feat
                last_feat = None
                for i in range(self.num_stages):
                    mask_head = self.mask_head[i]
                    if self.mask_info_flow:
                        mask_pred, last_feat = mask_head(mask_feats, last_feat)
                    else:
                        mask_pred = mask_head(mask_feats)
                    aug_masks.append(mask_pred.sigmoid().cpu().numpy())
                merged_masks = merge_aug_masks(aug_masks,
                                               [img_meta] * self.num_stages,
                                               self.test_cfg.rcnn)
                segm_result = self.mask_head[-1].get_seg_masks(
                    merged_masks, _bboxes, det_labels, rcnn_test_cfg,
                    ori_shape, scale_factor, rescale)
            ms_segm_result['ensemble'] = segm_result

        if not self.test_cfg.keep_all_stages:
            if self.with_mask:
                results = (ms_bbox_result['ensemble'],
                           ms_segm_result['ensemble'])
            else:
                results = ms_bbox_result['ensemble']
        else:
            if self.with_mask:
                results = {
                    stage: (ms_bbox_result[stage], ms_segm_result[stage])
                    for stage in ms_bbox_result
                }
            else:
                results = ms_bbox_result

        return results

    def aug_test(self, img, img_meta, proposals=None, rescale=False):
        raise NotImplementedError