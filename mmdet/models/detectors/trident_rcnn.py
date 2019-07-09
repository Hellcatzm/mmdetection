# author: hellcatzm
# data:   2019/7/02
import torch
from .two_stage import TwoStageDetector
from ..registry import DETECTORS
from mmdet.core import bbox2roi, bbox2result, build_assigner, build_sampler

# gradient = []
# gradient1 = []
# def hook(grad):
#     print("*"*100)
#     print('x')
#     print(grad.shape)
#     print([g.sum() for g in grad])
#     gradient.append(grad)
#     print("*" * 100)
# def backward_hook(module, grad_input, grad_output):
#     gradient1.append(grad_output)
#     gradient1.append(grad_input)
#     for g in grad_output:
#         print('out: ', g.shape)
#         print(g.sum())
#     for g in grad_input:
#         if g is not None:
#             print('in: ', g.shape)
#             print(g.sum())
#     print("*" * 100)

@DETECTORS.register_module
class TridentRCNN(TwoStageDetector):

    def __init__(self,
                 backbone,
                 rpn_head,
                 bbox_roi_extractor,
                 bbox_head,
                 train_cfg,
                 test_cfg,
                 neck=None,
                 shared_head=None,
                 pretrained=None,
                 scale_aware=True,
                 valid_range=((0, 90), (30, 160), (90, -1)),
                 **kwargs):
        super(TridentRCNN, self).__init__(
            backbone=backbone,
            neck=neck,
            shared_head=shared_head,
            rpn_head=rpn_head,
            bbox_roi_extractor=bbox_roi_extractor,
            bbox_head=bbox_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            pretrained=pretrained)
        self.scale_aware = scale_aware
        self.valid_range = valid_range

    def extract_feat(self, img):
        x = self.backbone(img)
        if self.with_neck:
            x = self.neck(x)
        return x

    def forward_train(self,
                      img,
                      img_meta,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      gt_masks=None,
                      proposals=None):
        x = self.extract_feat(img)

        # c4_shape = list(x[0].shape)
        # c4_shape[0] *= 3
        # x = torch.stack([x[0]] * 3, dim=1)
        # x = [x.view(c4_shape)]

        if not self.backbone.shared:
            # x[0].register_hook(hook)
            c4_shape = list(x[0].shape)
            c4_shape[0] *= 3
            x = torch.stack([x[0]] * 3, dim=1)
            x = [x.view(c4_shape)]

        img_meta_ = list()
        gt_bboxes_ = list()
        gt_labels_ = list()
        gt_masks_ = list()
        gt_bboxes_ignore_ = list()
        val_index = torch.zeros(x[0].shape[0])
        val_pointer = -1
        for img_num in range(img.size(0)):
            for lo, hi in self.valid_range:
                val_pointer += 1
                bboxes_hw = gt_bboxes[img_num][:, 2:] - gt_bboxes[img_num][:, :2]
                boxes_area = (bboxes_hw[:, 0] * bboxes_hw[:, 1]).clamp(min=0)
                if hi >= 0:
                    boxes_index = (boxes_area > lo ** 2) & (boxes_area < hi ** 2)
                else:
                    boxes_index = boxes_area > lo ** 2
                if boxes_index.int().sum():
                    if self.scale_aware:
                        val_index[val_pointer] = 1
                        arg_idx = torch.nonzero(boxes_index).squeeze()
                        img_meta_.append(img_meta[img_num])
                        gt_bboxes_.append(gt_bboxes[img_num][arg_idx])  # [3n, [gts_v, 4]]
                        gt_labels_.append(gt_labels[img_num][arg_idx])  # [3n, [gts_v]]
                        # gt_masks_.append(gt_masks[img_num][arg_idx.cpu()])
                        gt_bboxes_ignore_.append(gt_bboxes_ignore[img_num])
                        if arg_idx.numel() == 1:
                            gt_bboxes_[-1].unsqueeze_(dim=0)
                            gt_labels_[-1].unsqueeze_(dim=0)
                            # gt_masks_[-1] = gt_masks_[-1][None]
                    else:
                        img_meta_.append(img_meta[img_num])
                        gt_bboxes_.append(gt_bboxes[img_num])  # [3n, [gts_v, 4]]
                        gt_labels_.append(gt_labels[img_num])  # [3n, [gts_v]]
                        # gt_masks_.append(gt_masks[img_num][arg_idx.cpu()])
                        gt_bboxes_ignore_.append(gt_bboxes_ignore[img_num])

        img_meta = img_meta_
        gt_bboxes = gt_bboxes_
        gt_labels = gt_labels_
        gt_bboxes_ignore = gt_bboxes_ignore_

        if len(x) != 1:
            rpn_ids, rcnn_ids = 0, 1
        else:
            rpn_ids = rcnn_ids = 0
        if self.scale_aware:
            idx = torch.nonzero(val_index).squeeze()
            rpn_feat = [x[rpn_ids][idx][None]] if idx.numel() == 1 else [x[rpn_ids][idx]]
            rcnn_feat = [x[rcnn_ids][idx][None]] if idx.numel() == 1 else [x[rcnn_ids][idx]]
        else:
            rpn_feat = [x[rpn_ids]]
            rcnn_feat = [x[rcnn_ids]]
        # rpn_feat = [x[0]]
        # rcnn_feat = [x[0]]
        losses = dict()

        # RPN forward and loss
        if self.with_rpn:
            rpn_outs = self.rpn_head(rpn_feat)
            rpn_loss_inputs = rpn_outs + (gt_bboxes, img_meta,
                                          self.train_cfg.rpn)
            rpn_losses = self.rpn_head.loss(
                *rpn_loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)
            losses.update(rpn_losses)

            proposal_cfg = self.train_cfg.get('rpn_proposal',
                                              self.test_cfg.rpn)
            proposal_inputs = rpn_outs + (img_meta, proposal_cfg)
            proposal_list = self.rpn_head.get_bboxes(*proposal_inputs)
        else:
            proposal_list = proposals

        # assign gts and sample proposals
        if self.with_bbox or self.with_mask:
            bbox_assigner = build_assigner(self.train_cfg.rcnn.assigner)
            bbox_sampler = build_sampler(
                self.train_cfg.rcnn.sampler, context=self)
            num_imgs = len(img_meta)
            if gt_bboxes_ignore is None:
                gt_bboxes_ignore = [None for _ in range(num_imgs)]
            sampling_results = []
            for i in range(num_imgs):
                assign_result = bbox_assigner.assign(
                    proposal_list[i], gt_bboxes[i], gt_bboxes_ignore[i],
                    gt_labels[i])
                sampling_result = bbox_sampler.sample(
                    assign_result,
                    proposal_list[i],
                    gt_bboxes[i],
                    gt_labels[i],
                    feats=[lvl_feat[i][None] for lvl_feat in rcnn_feat])
                sampling_results.append(sampling_result)

        # bbox head forward and loss
        if self.with_bbox:
            self.sr = sampling_results
            rois = bbox2roi([res.bboxes for res in sampling_results])
            # TODO: a more flexible way to decide which feature maps to use

            # self.bbox_roi_extractor.register_backward_hook(backward_hook)
            bbox_feats = self.bbox_roi_extractor(
                rcnn_feat[:self.bbox_roi_extractor.num_inputs], rois)
            # bbox_feats.register_hook(hook)
            if self.with_shared_head:
                bbox_feats = self.shared_head(bbox_feats)
            self.bf = bbox_feats, rois
            # bbox_feats.register_hook(hook)
            cls_score, bbox_pred = self.bbox_head(bbox_feats)

            bbox_targets = self.bbox_head.get_target(
                sampling_results, gt_bboxes, gt_labels, self.train_cfg.rcnn)
            self.bt = bbox_targets, cls_score, bbox_pred
            s = torch.argmax(cls_score, dim=1)
            print("ACC:",
                  (s[bbox_targets[0]>0]==bbox_targets[0][bbox_targets[0]>0]).sum().float()/float(s[bbox_targets[0]>0].numel()))

            loss_bbox = self.bbox_head.loss(cls_score, bbox_pred,
                                            *bbox_targets)
            losses.update(loss_bbox)

        # mask head forward and loss
        if self.with_mask:
            if not self.share_roi_extractor:
                pos_rois = bbox2roi(
                    [res.pos_bboxes for res in sampling_results])
                mask_feats = self.mask_roi_extractor(
                    rcnn_feat[:self.mask_roi_extractor.num_inputs], pos_rois)
                if self.with_shared_head:
                    mask_feats = self.shared_head(mask_feats)
            else:
                pos_inds = []
                device = bbox_feats.device
                for res in sampling_results:
                    pos_inds.append(
                        torch.ones(
                            res.pos_bboxes.shape[0],
                            device=device,
                            dtype=torch.uint8))
                    pos_inds.append(
                        torch.zeros(
                            res.neg_bboxes.shape[0],
                            device=device,
                            dtype=torch.uint8))
                pos_inds = torch.cat(pos_inds)
                mask_feats = bbox_feats[pos_inds]
            mask_pred = self.mask_head(mask_feats)

            mask_targets = self.mask_head.get_target(
                sampling_results, gt_masks, self.train_cfg.rcnn)
            pos_labels = torch.cat(
                [res.pos_gt_labels for res in sampling_results])
            loss_mask = self.mask_head.loss(mask_pred, mask_targets,
                                            pos_labels)
            losses.update(loss_mask)

        return losses

    def simple_test(self, img, img_meta, proposals=None, rescale=False):
        """Test without augmentation."""
        assert self.with_bbox, "Bbox head must be implemented."

        x = self.extract_feat(img)
        if len(x) != 1:
            rpn_ids, rcnn_ids = 0, 1
        else:
            rpn_ids = rcnn_ids = 0
        rpn_feat, rcnn_feat = [x[rpn_ids]], [x[rcnn_ids]]

        proposal_list = self.simple_test_rpn(
            rpn_feat, img_meta, self.test_cfg.rpn) if proposals is None else proposals

        det_bboxes, det_labels = self.simple_test_bboxes(
            rcnn_feat, img_meta, proposal_list, self.test_cfg.rcnn, rescale=rescale)
        bbox_results = bbox2result(det_bboxes, det_labels,
                                   self.bbox_head.num_classes)

        if not self.with_mask:
            return bbox_results
        else:
            segm_results = self.simple_test_mask(
                rcnn_feat, img_meta, det_bboxes, det_labels, rescale=rescale)
            return bbox_results, segm_results