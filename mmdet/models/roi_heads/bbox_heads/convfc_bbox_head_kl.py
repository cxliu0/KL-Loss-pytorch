import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule

from mmdet.models.builder import HEADS
from .bbox_head import BBoxHead

from mmdet.models.losses import accuracy
from mmdet.core import (auto_fp16, build_bbox_coder, force_fp32, multi_apply,
                        multiclass_nms)

@HEADS.register_module()
class ConvFCBBoxHeadKL(BBoxHead):
    r"""More general bbox head, with shared conv and fc layers and two optional
    separated branches.

    .. code-block:: none

                                    /-> cls convs -> cls fcs -> cls
        shared convs -> shared fcs
                                    \-> reg convs -> reg fcs -> reg
                                                            \-> reg variance                                 
    """  # noqa: W605

    def __init__(self,
                 num_shared_convs=0,
                 num_shared_fcs=0,
                 num_cls_convs=0,
                 num_cls_fcs=0,
                 num_reg_convs=0,
                 num_reg_fcs=0,
                 conv_out_channels=256,
                 fc_out_channels=1024,
                 conv_cfg=None,
                 norm_cfg=None,
                 *args,
                 **kwargs):
        super(ConvFCBBoxHeadKL, self).__init__(*args, **kwargs)
        assert (num_shared_convs + num_shared_fcs + num_cls_convs +
                num_cls_fcs + num_reg_convs + num_reg_fcs > 0)
        if num_cls_convs > 0 or num_reg_convs > 0:
            assert num_shared_fcs == 0
        if not self.with_cls:
            assert num_cls_convs == 0 and num_cls_fcs == 0
        if not self.with_reg:
            assert num_reg_convs == 0 and num_reg_fcs == 0
        self.num_shared_convs = num_shared_convs
        self.num_shared_fcs = num_shared_fcs
        self.num_cls_convs = num_cls_convs
        self.num_cls_fcs = num_cls_fcs
        self.num_reg_convs = num_reg_convs
        self.num_reg_fcs = num_reg_fcs
        self.conv_out_channels = conv_out_channels
        self.fc_out_channels = fc_out_channels
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg

        # add shared convs and fcs
        self.shared_convs, self.shared_fcs, last_layer_dim = \
            self._add_conv_fc_branch(
                self.num_shared_convs, self.num_shared_fcs, self.in_channels,
                True)
        self.shared_out_channels = last_layer_dim

        # add cls specific branch
        self.cls_convs, self.cls_fcs, self.cls_last_dim = \
            self._add_conv_fc_branch(
                self.num_cls_convs, self.num_cls_fcs, self.shared_out_channels)

        # add reg specific branch
        self.reg_convs, self.reg_fcs, self.reg_last_dim = \
            self._add_conv_fc_branch(
                self.num_reg_convs, self.num_reg_fcs, self.shared_out_channels)

        if self.num_shared_fcs == 0 and not self.with_avg_pool:
            if self.num_cls_fcs == 0:
                self.cls_last_dim *= self.roi_feat_area
            if self.num_reg_fcs == 0:
                self.reg_last_dim *= self.roi_feat_area

        self.relu = nn.ReLU(inplace=True)
        # reconstruct fc_cls and fc_reg since input channels are changed
        if self.with_cls:
            self.fc_cls = nn.Linear(self.cls_last_dim, self.num_classes + 1)
        if self.with_reg:
            out_dim_reg = (4 if self.reg_class_agnostic else 4 *
                           self.num_classes)
            self.fc_reg = nn.Linear(self.reg_last_dim, out_dim_reg)

            self.fc_reg_kl = nn.Linear(self.reg_last_dim, out_dim_reg)
        
        # kl params
        self.loss_kl_coef = kwargs['loss_kl_coef'] if 'loss_kl_coef' in kwargs else 1.0
        self.gau_init_std = kwargs['gau_init_std'] if 'gau_init_std' in kwargs else 1e-5

        # inference params
        self.softnms = kwargs['softnms'] if 'softnms' in kwargs else False
        self.softnms_sigma = kwargs['softnms_sigma'] if 'softnms_sigma' in kwargs else 0.5
        self.var_vote = kwargs['var_vote'] if 'var_vote' in kwargs else False
        self.var_sigma_t = kwargs['var_sigma_t'] if 'var_sigma_t' in kwargs else 0.01

    def _add_conv_fc_branch(self,
                            num_branch_convs,
                            num_branch_fcs,
                            in_channels,
                            is_shared=False):
        """Add shared or separable branch.

        convs -> avg pool (optional) -> fcs
        """
        last_layer_dim = in_channels
        # add branch specific conv layers
        branch_convs = nn.ModuleList()
        if num_branch_convs > 0:
            for i in range(num_branch_convs):
                conv_in_channels = (
                    last_layer_dim if i == 0 else self.conv_out_channels)
                branch_convs.append(
                    ConvModule(
                        conv_in_channels,
                        self.conv_out_channels,
                        3,
                        padding=1,
                        conv_cfg=self.conv_cfg,
                        norm_cfg=self.norm_cfg))
            last_layer_dim = self.conv_out_channels
        # add branch specific fc layers
        branch_fcs = nn.ModuleList()
        if num_branch_fcs > 0:
            # for shared branch, only consider self.with_avg_pool
            # for separated branches, also consider self.num_shared_fcs
            if (is_shared
                    or self.num_shared_fcs == 0) and not self.with_avg_pool:
                last_layer_dim *= self.roi_feat_area
            for i in range(num_branch_fcs):
                fc_in_channels = (
                    last_layer_dim if i == 0 else self.fc_out_channels)
                branch_fcs.append(
                    nn.Linear(fc_in_channels, self.fc_out_channels))
            last_layer_dim = self.fc_out_channels
        return branch_convs, branch_fcs, last_layer_dim

    def init_weights(self):
        super(ConvFCBBoxHeadKL, self).init_weights()
        # conv layers are already initialized by ConvModule
        for module_list in [self.shared_fcs, self.cls_fcs, self.reg_fcs]:
            for m in module_list.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    nn.init.constant_(m.bias, 0)

        # KL-Loss: initialize fc (for alpha prediction) with gaussian distribution
        std = self.gau_init_std
        for m in self.fc_reg_kl.modules():
            nn.init.normal_(m.weight, mean=0.0, std=std)
            nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # shared part
        if self.num_shared_convs > 0:
            for conv in self.shared_convs:
                x = conv(x)

        if self.num_shared_fcs > 0:
            if self.with_avg_pool:
                x = self.avg_pool(x)

            x = x.flatten(1)

            for fc in self.shared_fcs:
                x = self.relu(fc(x))
        # separate branches
        x_cls = x
        x_reg = x

        for conv in self.cls_convs:
            x_cls = conv(x_cls)
        if x_cls.dim() > 2:
            if self.with_avg_pool:
                x_cls = self.avg_pool(x_cls)
            x_cls = x_cls.flatten(1)
        for fc in self.cls_fcs:
            x_cls = self.relu(fc(x_cls))

        for conv in self.reg_convs:
            x_reg = conv(x_reg)
        if x_reg.dim() > 2:
            if self.with_avg_pool:
                x_reg = self.avg_pool(x_reg)
            x_reg = x_reg.flatten(1)
        for fc in self.reg_fcs:
            x_reg = self.relu(fc(x_reg))
        
        cls_score = self.fc_cls(x_cls) if self.with_cls else None
        bbox_pred = self.fc_reg(x_reg) if self.with_reg else None
        bbox_var = self.fc_reg_kl(x_reg) if self.with_reg else None
        return cls_score, bbox_pred, bbox_var


@HEADS.register_module()
class Shared2FCBBoxHeadKL(ConvFCBBoxHeadKL):

    def __init__(self, fc_out_channels=1024, *args, **kwargs):
        super(Shared2FCBBoxHeadKL, self).__init__(
            num_shared_convs=0,
            num_shared_fcs=2,
            num_cls_convs=0,
            num_cls_fcs=0,
            num_reg_convs=0,
            num_reg_fcs=0,
            fc_out_channels=fc_out_channels,
            *args,
            **kwargs)
        
    def _get_target_single(self, pos_bboxes, neg_bboxes, pos_gt_bboxes,
                           pos_gt_labels, pos_gt_inds, cfg):
        num_pos = pos_bboxes.size(0)
        num_neg = neg_bboxes.size(0)
        num_samples = num_pos + num_neg
                
        # original implementation uses new_zeros since BG are set to be 0
        # now use empty & fill because BG cat_id = num_classes,
        # FG cat_id = [0, num_classes-1]
        labels = pos_bboxes.new_full((num_samples, ),
                                     self.num_classes,
                                     dtype=torch.long)
        label_weights = pos_bboxes.new_zeros(num_samples)
        bbox_targets = pos_bboxes.new_zeros(num_samples, 4)
        bbox_weights = pos_bboxes.new_zeros(num_samples, 4)
        if num_pos > 0:
            labels[:num_pos] = pos_gt_labels
            pos_weight = 1.0 if cfg.pos_weight <= 0 else cfg.pos_weight
            label_weights[:num_pos] = pos_weight
            if not self.reg_decoded_bbox:
                pos_bbox_targets = self.bbox_coder.encode(
                    pos_bboxes, pos_gt_bboxes)
            else:
                pos_bbox_targets = pos_gt_bboxes
            bbox_targets[:num_pos, :] = pos_bbox_targets
            bbox_weights[:num_pos, :] = 1
        if num_neg > 0:
            label_weights[-num_neg:] = 1.0

        return labels, label_weights, bbox_targets, bbox_weights

    def get_targets(self,
                    sampling_results,
                    gt_bboxes,
                    gt_labels,
                    rcnn_train_cfg,
                    concat=True,
                    **kwargs):           
        pos_bboxes_list = [res.pos_bboxes for res in sampling_results]
        neg_bboxes_list = [res.neg_bboxes for res in sampling_results]
        pos_gt_bboxes_list = [res.pos_gt_bboxes for res in sampling_results]
        pos_gt_labels_list = [res.pos_gt_labels for res in sampling_results]
        pos_gt_inds_list = [res.pos_assigned_gt_inds for res in sampling_results]
        labels, label_weights, bbox_targets, bbox_weights = multi_apply(
            self._get_target_single,
            pos_bboxes_list,
            neg_bboxes_list,
            pos_gt_bboxes_list,
            pos_gt_labels_list,
            pos_gt_inds_list,
            cfg=rcnn_train_cfg,
            )

        if concat:
            labels = torch.cat(labels, 0)
            label_weights = torch.cat(label_weights, 0)
            bbox_targets = torch.cat(bbox_targets, 0)
            bbox_weights = torch.cat(bbox_weights, 0)
        return labels, label_weights, bbox_targets, bbox_weights
    
    def loss(self,
             cls_score,
             bbox_pred,
             bbox_var,
             rois,
             labels,
             label_weights,
             bbox_targets,
             bbox_weights,
             reduction_override=None,
             ):
        losses = dict()
        if cls_score is not None:
            avg_factor = max(torch.sum(label_weights > 0).float().item(), 1.)
            if cls_score.numel() > 0:
                losses['loss_cls'] = self.loss_cls(
                    cls_score,
                    labels,
                    label_weights,
                    avg_factor=avg_factor,
                    reduction_override=reduction_override)
                losses['acc'] = accuracy(cls_score, labels)

        if bbox_pred is not None:
            bg_class_ind = self.num_classes
            # 0~self.num_classes-1 are FG, self.num_classes is BG
            pos_inds = (labels >= 0) & (labels < bg_class_ind)
            # do not perform bounding box regression for BG anymore.            

            if pos_inds.any():
                if self.reg_decoded_bbox:
                    bbox_pred = self.bbox_coder.decode(rois[:, 1:], bbox_pred)
                if self.reg_class_agnostic:
                    pos_bbox_pred = bbox_pred.view(
                        bbox_pred.size(0), 4)[pos_inds.type(torch.bool)]
                    
                    pos_bbox_var = bbox_var.view(
                        bbox_var.size(0), 4)[pos_inds.type(torch.bool)]
                else:
                    pos_bbox_pred = bbox_pred.view(
                        bbox_pred.size(0), -1,
                        4)[pos_inds.type(torch.bool),
                           labels[pos_inds.type(torch.bool)]]
                    
                    pos_bbox_var = bbox_var.view(
                        bbox_var.size(0), -1,
                        4)[pos_inds.type(torch.bool),
                           labels[pos_inds.type(torch.bool)]]

                l1_loss = self.loss_bbox(
                    pos_bbox_pred,
                    bbox_targets[pos_inds.type(torch.bool)],
                    bbox_weights[pos_inds.type(torch.bool)],
                    reduction_override='none'
                )

                losses['loss_bbox'] = (torch.exp(-pos_bbox_var) * l1_loss + 0.5 * pos_bbox_var).sum() / bbox_targets.size(0) * self.loss_kl_coef                
                losses['pos_bbox_var'] = pos_bbox_var
            else:
                losses['loss_bbox'] = bbox_pred.sum() * 0
        return losses


    def get_bboxes(self,
                   rois,
                   cls_score,
                   bbox_pred,
                   img_shape,
                   scale_factor,
                   rescale=False,
                   cfg=None,
                   variance=None):        
        if isinstance(cls_score, list):
            cls_score = sum(cls_score) / float(len(cls_score))
        scores = F.softmax(cls_score, dim=1) if cls_score is not None else None

        if bbox_pred is not None:
            bboxes = self.bbox_coder.decode(
                rois[:, 1:], bbox_pred, max_shape=img_shape)
        else:
            bboxes = rois[:, 1:].clone()
            if img_shape is not None:
                bboxes[:, [0, 2]].clamp_(min=0, max=img_shape[1])
                bboxes[:, [1, 3]].clamp_(min=0, max=img_shape[0])

        if rescale and bboxes.size(0) > 0:
            if isinstance(scale_factor, float):
                bboxes /= scale_factor
            else:
                scale_factor = bboxes.new_tensor(scale_factor)
                bboxes = (bboxes.view(bboxes.size(0), -1, 4) /
                          scale_factor).view(bboxes.size()[0], -1)

        if cfg is None:
            return bboxes, scores
        else:
            # KL-loss NMS
            var_det, var_labels = self.kl_nms(bboxes, scores, cfg.score_thr, cfg.nms, cfg.max_per_img, variance)
            return var_det, var_labels

    def kl_nms(self, bboxes, scores, score_thres, nms_cfg, max_num=-1, variance=None):
        nms_iou = nms_cfg['iou_threshold']
        bboxes = bboxes.view(-1, self.num_classes, 4)
        if not variance is None:
            variance = variance.view(-1, self.num_classes, 4)

        def compute_iou(boxes1, boxes2):
            """
            compute IoU between boxes1 and boxes2
            """
            boxes1_area = (boxes1[..., 2] - boxes1[..., 0]) * (boxes1[..., 3] - boxes1[..., 1])
            boxes2_area = (boxes2[..., 2] - boxes2[..., 0]) * (boxes2[..., 3] - boxes2[..., 1])

            left_up = torch.max(boxes1[..., :2], boxes2[..., :2])
            right_down = torch.min(boxes1[..., 2:], boxes2[..., 2:])

            inter_section = torch.max(right_down - left_up, torch.zeros_like(right_down))
            inter_area = inter_section[..., 0] * inter_section[..., 1]
            union_area = boxes1_area + boxes2_area - inter_area
            iou = 1.0 * inter_area / union_area
            return iou

        def nms_class(cls_boxes, nms_iou):
            """
            Algorithm 1 of the original paper
            """
            assert cls_boxes.shape[1] == 5 or cls_boxes.shape[1] == 9
            keep = []
            while cls_boxes.shape[0] > 0:
                # get bbox with max score
                max_idx = torch.argmax(cls_boxes[:, 4])
                max_box = cls_boxes[max_idx].unsqueeze(0)
                
                # compute iou between max_box and other bboxes
                cls_boxes = torch.cat((cls_boxes[:max_idx], cls_boxes[max_idx + 1:]), 0)
                iou = compute_iou(max_box[:, :4], cls_boxes[:, :4])

                # KL var voting
                if variance is not None:
                    # get overlpapped bboxes
                    iou_mask = iou > 0
                    kl_bboxes = cls_boxes[iou_mask]
                    kl_bboxes = torch.cat((kl_bboxes, max_box), dim=0)
                    kl_ious = iou[iou_mask]
                    
                    # recover var to sigma**2
                    kl_var = kl_bboxes[:, -4:]
                    kl_var = torch.exp(kl_var)

                    # compute weighted bbox
                    p_i = torch.exp(-1 * torch.pow((1 - kl_ious), 2) / self.var_sigma_t)
                    p_i = torch.cat((p_i, torch.ones(1).cuda()), 0).unsqueeze(1)
                    p_i = p_i / kl_var
                    p_i = p_i / p_i.sum(dim=0)
                    max_box[0, :4] = (p_i * kl_bboxes[:, :4]).sum(dim=0)
                keep.append(max_box)

                # apply soft-NMS
                weight = torch.ones_like(iou)
                if not self.softnms:
                    weight[iou > nms_iou] = 0
                else:
                    weight = torch.exp(-1.0 * (iou ** 2 / self.softnms_sigma))
                cls_boxes[:, 4] = cls_boxes[:, 4] * weight
                
                # filter bboxes with low scores
                filter_idx = (cls_boxes[:, 4] >= score_thres).nonzero().squeeze(-1)
                cls_boxes = cls_boxes[filter_idx]
            return torch.cat(keep, 0).to(cls_boxes.device)

        # perform NMS
        num_cls = self.num_classes
        output_boxes, output_scores, output_labels = [], [], []
        for i in range(num_cls):
            filter_idx = (scores[:, i] >= score_thres).nonzero().squeeze(-1)
            if len(filter_idx) == 0:
                continue

            filter_boxes = bboxes[filter_idx, i]
            filter_scores = scores[:, i][filter_idx].unsqueeze(1)
            if variance is not None:
                filter_variance = variance[filter_idx, i]
                out_bboxes = nms_class(torch.cat((filter_boxes, filter_scores, filter_variance), 1), nms_iou)
            else:
                out_bboxes = nms_class(torch.cat((filter_boxes, filter_scores), 1), nms_iou)
            if out_bboxes.shape[0] > 0:
                output_boxes.append(out_bboxes[:, :4])
                output_scores.append(out_bboxes[:, 4])
                output_labels.extend([torch.ByteTensor([i]) for _ in range(len(out_bboxes))])

        # output results
        if len(output_boxes) == 0:
            return torch.empty(0,5).cuda(), torch.empty(0).cuda()
        else:
            output_boxes, output_scores, output_labels = torch.cat(output_boxes), torch.cat(output_scores), torch.cat(output_labels)

            # sort prediction
            sort_inds = torch.argsort(output_scores, descending=True)
            output_boxes, output_scores, output_labels = output_boxes[sort_inds], output_scores[sort_inds], output_labels[sort_inds]

            det_boxes = torch.cat([output_boxes, output_scores.view(-1, 1)], dim=1)
            if max_num > 0:
                det_boxes = det_boxes[:max_num]
                output_labels = output_labels[:max_num]
            return det_boxes, output_labels
