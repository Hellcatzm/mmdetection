import torch
import torch.nn as nn
from mmdet.models.utils import SharedConv
from .. import builder
from ..registry import DETECTORS
from .htc import HybridTaskCascade
from mmdet.core import (bbox2roi, bbox2result, build_assigner, build_sampler,
                        merge_aug_masks)
from mmdet.core.bbox.samplers.sampling_result import sample_nms
from functools import reduce


@DETECTORS.register_module
class TridentRCNN(HybridTaskCascade):

    def __init__(self, **kwargs):
        super(TridentRCNN, self).__init__(**kwargs)
        self.rpn_neck = SharedConv(1024, 256, kernel_size=1)
        self.rcnn_neck = nn.Conv2d(2048, 256, kernel_size=1)

    def extract_feat(self, img):
        x = self.backbone(img)
        return x

    def forward_train(self,
                      img,
                      img_meta,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      gt_masks=None,
                      gt_semantic_seg=None,
                      proposals=None):
        x = self.extract_feat(img)
        rpn_feat = self.rpn_neck(x[0])        # [3, [n, 256, h, w]]
        rcnn_feat = self.rcnn_neck(x[1])      # [3n, 256, h, w]

        losses = dict()

        # RPN part, the same as normal two-stage detectors
        assert self.with_rpn, "Trident Net must have RPN head"
        rpn_outs = self.rpn_head(rpn_feat)
        # [list: num_branches, [n, anchors*class, h, w]]
        # [list: num_branches, [n, anchors*4, h, w]]
        rpn_loss_inputs = rpn_outs + (gt_bboxes, img_meta,
                                      self.train_cfg.rpn)
        rpn_losses = self.rpn_head.loss(
            *rpn_loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)
        losses.update(rpn_losses)

        proposal_cfg = self.train_cfg.get('rpn_proposal',
                                          self.test_cfg.rpn)

        proposal_branch = list()
        gtbboxes_branch = list()
        gtlabels_branch = list()
        val_range = ((0, 10), (5, 15), (15, -1))
        for branch, (lo, hi) in enumerate(val_range):  # loop for branches
            # get_bboxes 输入 [list: num_levels, [n, anchors*class, h, w]],使用None模拟最外层list
            proposal_inputs = (rpn_outs[0][branch][None], rpn_outs[1][branch][None]) + (img_meta, proposal_cfg)
            proposal_list = self.rpn_head.get_bboxes(*proposal_inputs)  # [imgs, [2000(配置文件指定), 5]]

            proposal_val = list()
            gtbboxes_val = list()
            gtlabels_val = list()
            for img_num in range(len(proposal_list)):  # loop for images
                proposal_hw = proposal_list[img_num][:, 2:4] - proposal_list[img_num][:, :2]
                proposal_area = (proposal_hw[:, 0] * proposal_hw[:, 1]).clamp(min=0)
                if hi >= 0:
                    proposal_index = (proposal_area > lo ** 2) & (proposal_area < hi ** 2)
                else:
                    proposal_index = proposal_area > lo ** 2
                proposal_val.append(proposal_list[img_num][proposal_index])

                bboxes_hw = gt_bboxes[img_num][:, 2:] - gt_bboxes[img_num][:, :2]
                boxes_area = (bboxes_hw[:, 0] * bboxes_hw[:, 1]).clamp(min=0)
                if hi >= 0:
                    boxes_index = (boxes_area > lo ** 2) & (boxes_area < hi ** 2)
                else:
                    boxes_index = boxes_area > lo ** 2
                gtbboxes_val.append(gt_bboxes[img_num][boxes_index])
                gtlabels_val.append(gt_labels[img_num][boxes_index])

            proposal_branch.append(proposal_val)  # [branchs, [img_nums, [num_val_pro, 5]]]
            gtbboxes_branch.append(gtbboxes_val)  # [branchs, [img_nums, [num_val_gt, 4]]]
            gtlabels_branch.append(gtlabels_val)  # [branchs, [img_nums, [num_val_gt]]]

        self.proposal_branch = proposal_branch
        self.gtbboxes_branch = gtbboxes_branch
        self.gtlabels_branch = gtlabels_branch

        # semantic segmentation part
        # 2 outputs: segmentation prediction and embedded features
        assert not self.with_semantic, "Not supply yet"
        semantic_feat = None

        num_imgs = img.size(0)
        sampling_tmp = [[] for _ in range(num_imgs)]
        rcnn_train_cfg = self.train_cfg.rcnn[0]
        for branch in range(len(val_range)):
            proposal = proposal_branch[branch]
            gtbboxes = gtbboxes_branch[branch]
            gtlabels = gtlabels_branch[branch]

            bbox_assigner = build_assigner(rcnn_train_cfg.assigner)
            bbox_sampler = build_sampler(rcnn_train_cfg.sampler, context=self)

            for img_num in range(num_imgs):
                if (not proposal[img_num].numel()) or (not gtlabels[img_num].numel()):
                    continue
                assign_result = bbox_assigner.assign(
                    proposal[img_num],
                    gtbboxes[img_num],
                    None,
                    gtlabels[img_num])
                sampling_result = bbox_sampler.sample(
                    assign_result,
                    proposal[img_num],  # [num_val_pro, 5]
                    gtbboxes[img_num],  # [num_val_pro, 4]
                    gtlabels[img_num],  # [num_val_pro,]
                    if_scale=True,
                    feats=[rcnn_feat[img_num][None]])  # [n, c, h, w] feat参数是为了特定的sample方式提供
                sampling_tmp[img_num].append(sampling_result)  # [images, [branches of sample_obj]]
        # TODO "we select ground truth boxes which are valid for this branch
        #  according to Eq1 during anchor label assiginment for RPN"
        # sampling_results = [reduce(sample_nms, sampling_results[img_num]) for img_num in range(num_imgs)]  # [imgs]
        sampling_results = list()
        [sampling_results.extend(samples) for samples in sampling_tmp]
        self.sr = sampling_results
        img_meta_ = list()
        for meta in img_meta:
            img_meta_.extend([meta]*len(val_range))
        img_meta = img_meta_

        for i in range(self.num_stages):
            self.current_stage = i
            rcnn_train_cfg = self.train_cfg.rcnn[i]
            lw = self.train_cfg.stage_loss_weights[i]

            if i != 0:
                # TODO "we sample valid proposals for each branch during the training of RCNN"
                sampling_results = []
                bbox_assigner = build_assigner(rcnn_train_cfg.assigner)
                bbox_sampler = build_sampler(rcnn_train_cfg.sampler, context=self)
                num_imgs = img.size(0)
                if gt_bboxes_ignore is None:
                    gt_bboxes_ignore = [None for _ in range(num_imgs)]

                for j in range(num_imgs):
                    assign_result = bbox_assigner.assign(
                        proposal_list[j],
                        gt_bboxes[j],
                        gt_bboxes_ignore[j],
                        gt_labels[j])
                    sampling_result = bbox_sampler.sample(
                        assign_result,
                        proposal_list[j],
                        gt_bboxes[j],
                        gt_labels[j],
                        feats=[rcnn_feat[j][None]])
                    sampling_results.append(sampling_result)  # [images]

            # bbox head forward and loss
            loss_bbox, rois, bbox_targets, bbox_pred = \
                self._bbox_forward_train(
                    i, rcnn_feat[None], sampling_results, rcnn_train_cfg, semantic_feat)
            roi_labels = bbox_targets[0]
            print(rois.shape, roi_labels.shape)

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
                            rois, roi_labels, bbox_pred, pos_is_gts, img_meta)
                        # re-assign and sample 512 RoIs from 512 RoIs
                        sampling_results = []
                        for j in range(num_imgs):
                            assign_result = bbox_assigner.assign(
                                proposal_list[j],
                                gt_bboxes[j],
                                gt_bboxes_ignore[j],
                                gt_labels[j])
                            sampling_result = bbox_sampler.sample(
                                assign_result,
                                proposal_list[j],
                                gt_bboxes[j],
                                gt_labels[j],
                                feats=[rcnn_feat[j][None]])
                            sampling_results.append(sampling_result)
                loss_mask = self._mask_forward_train(i, rcnn_feat[None], sampling_results,
                                                     gt_masks, rcnn_train_cfg,
                                                     semantic_feat)
                for name, value in loss_mask.items():
                    losses['s{}.{}'.format(i, name)] = (
                        value * lw if 'loss' in name else value)

            # refine bboxes (same as Cascade R-CNN)
            if i < self.num_stages - 1 and not self.interleaved:
                pos_is_gts = [res.pos_is_gt for res in sampling_results]
                with torch.no_grad():
                    proposal_list = self.bbox_head[i].refine_bboxes(
                        rois, roi_labels, bbox_pred, pos_is_gts, img_meta)
        return losses

    def simple_test(self, img, img_meta, proposals=None, rescale=False):
        x = self.extract_feat(img)
        rpn_feat = self.rpn_neck(x[0])    # [n, 256, h, w]
        rcnn_feat = self.rcnn_neck(x[1])  # [n, 256, h, w]  TODO neck提取调整

        proposal_list = self.simple_test_rpn(
            rpn_feat[None], img_meta, self.test_cfg.rpn) if proposals is None else proposals

        self.pro = proposal_list
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
