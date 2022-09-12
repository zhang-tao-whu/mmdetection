# Copyright (c) OpenMMLab. All rights reserved.
import copy
from abc import ABCMeta, abstractmethod

import torch
from mmcv.cnn.utils.weight_init import constant_init
from mmcv.ops import batched_nms
from mmcv.runner import BaseModule, force_fp32
from mmcv.cnn import build_plugin_layer, xavier_init
from mmcv.cnn.bricks.transformer import build_positional_encoding
from mmdet.core import build_assigner, build_sampler
from ..builder import HEADS, build_loss, build_roi_extractor
from mmdet.core.utils import filter_scores_and_topk, select_single_mlvl
import torch.nn as nn
from functools import partial

@HEADS.register_module(force=True)
class LineFormerHead(BaseModule, metaclass=ABCMeta):
    def __init__(self,
                 in_channels=64,
                 out_channel=4,
                 num_query=30,
                 roi_wh=(40, 40),
                 expand_scale=1.1,
                 roi_extractor=None,
                 positional_encoding=None,
                 line_predictor=None,
                 line_loss=None,
                 train_cfg=None,
                 test_cfg=None,
                 init_cfg=None,
                 with_line_refine=False,
                 **kwargs
                 ):
        super(LineFormerHead, self).__init__(init_cfg)
        self.in_channels = in_channels
        self.num_query = num_query
        self.roi_w = roi_wh[0]
        self.roi_h = roi_wh[1]
        self.expand_scale = expand_scale
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.with_line_refine = with_line_refine

        self.roi_extractor = build_roi_extractor(roi_extractor)
        self.positional_encoding = build_positional_encoding(positional_encoding)
        line_predictor_ = copy.deepcopy(line_predictor)
        line_predictor_.update(num_query=num_query)
        line_predictor_.update(out_channel=out_channel)
        line_predictor_.update(roi_wh=roi_wh)
        self.line_predictor = build_plugin_layer(line_predictor_)[1]
        self.hidden_dim = self.line_predictor.hidden_dim
        self.num_layers = self.line_predictor.num_layers

        self.line_loss = build_plugin_layer(line_loss)[1]

        self.query_embeds = nn.Embedding(num_query, self.hidden_dim)
        self.pos_embeds = nn.Embedding(num_query, self.hidden_dim)

        self._init_layers()

    def _init_layers(self):
        self.fusion_1 = nn.Sequential(
            nn.Conv2d(self.in_channels, 256, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
        )
        self.fusion_2 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
        )
        self.fusion_3 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
        )
        self.out_proj_1 = nn.Conv2d(256, self.hidden_dim, kernel_size=1, stride=1, bias=True)
        self.out_proj_2 = nn.Conv2d(256, self.hidden_dim, kernel_size=1, stride=1, bias=True)
        self.out_proj_3 = nn.Conv2d(256, self.hidden_dim, kernel_size=1, stride=1, bias=True)

    def init_weights(self):
        """Initialize weights of the LineFormerHead."""
        self.line_predictor.init_weights()
        for m in self.modules():
            if hasattr(m, 'weight') and m.weight.dim() > 1:
                xavier_init(m, distribution='uniform')
        self._is_init = True

    def forward(self, cnn_features, img_metas, contours, bboxes=None, inds=None, use_fpn_level=0):
        """Forward function.

        Args:
            cnn_features (tuple[Tensor]): Multi-scale features from the upstream
                network, each is a 4D-tensor with shape (N, C, H, W).
            contours (Tensor): Shape (total_num_gts, 128, 2).
            bboxes (Tensor/list[Tensor]): Bounding boxes of instances, which are gt_bboxes /
                contour_derived_bboxes when training / testing.
                Shape (total_num_gts, 4).
            inds (Tensor): Indicating which image the indexed bbox belong to.
                Shape (total_num_gts, ).
        """
        cnn_feature = cnn_features[use_fpn_level]
        device = cnn_feature.device
        assert contours is not None
        if bboxes is None:
            # In testing mode, using contour derived bboxes
            max_coor = torch.max(contours, dim=1)[0]
            min_coor = torch.min(contours, dim=1)[0]
            ct = (max_coor + min_coor) / 2.
            roi_wh = (max_coor - min_coor) * self.expand_scale
            inds = torch.zeros((contours.size(0),), dtype=torch.int32, device=device)
        else:
            assert inds is not None
            if isinstance(bboxes, list):
                bboxes = torch.cat(bboxes, 0)
            # In training mode, using gt bboxes
            ct, roi_wh = bboxes[..., :2], bboxes[..., 2:] * self.expand_scale

        rois = torch.cat([ct[..., :1] - roi_wh[..., :1] / 2.,
                          ct[..., 1:] - roi_wh[..., 1:] / 2.,
                          ct[..., :1] + roi_wh[..., :1] / 2.,
                          ct[..., 1:] + roi_wh[..., 1:] / 2.], dim=1) # (n, 4)
        rois = torch.cat([inds[:, None].float(), rois], dim=1) # (n, 5)

        roi = self.roi_extractor([cnn_feature], rois) # (ngt, c, roi_h, roi_w)
        roi_1 = self.fusion_1(roi) # (ngt, c, roi_h, roi_w)
        roi_2 = self.fusion_2(roi_1) # (ngt, c, roi_h / 2, roi_w / 2)
        roi_3 = self.fusion_3(roi_2) # (ngt, c, roi_h / 4, roi_w / 4)
        multi_scale_roi = [self.out_proj_3(roi_3),
                           self.out_proj_2(roi_2),
                           self.out_proj_1(roi_1)]
        multi_scale_pos = []
        for x in multi_scale_roi:
            mask = x.new_zeros(size=(x.shape[0], x.shape[2], x.shape[3]))
            multi_scale_pos.append(self.positional_encoding(mask))

        # centered coords in ct, rescaled in (-roi_wh/2, +roi_wh/2)
        roi_coords = self.get_roi_coords(contours, ct, roi_wh) # (n, 128, 2)

        num_gts = roi_coords.size(0)
        query_embeds = self.query_embeds.weight.unsqueeze(1).repeat(1, num_gts, 1) #nq, ngt, c
        pos_embeds = self.pos_embeds.weight.unsqueeze(1).repeat(1, num_gts, 1)

        multi_scale_memory = [roi.flatten(2).permute(2, 0, 1)
                              for roi in multi_scale_roi]
        multi_scale_memory_pos = [pos_embed.flatten(2).permute(2, 0, 1)
                                  for pos_embed in multi_scale_pos] # each (h*w, ngt, c)

        lines, normed_lines, line_scores = \
            self.line_predictor(query=query_embeds,
                                query_pos=pos_embeds,
                                mlvl_feats=multi_scale_memory,
                                mlvl_pos_embeds=multi_scale_memory_pos,
                                ct=ct,
                                roi_wh=roi_wh,)
        # lines: list of ((ngt, nq, 2), (ngt, nq, 2)), length of the list is num_layers.
        # line_scores: list of (ngt, nq, 2), length of the list is num_layers.
        return lines, normed_lines, line_scores

    def forward_train(self,
                      x,
                      img_metas,
                      contours,
                      bboxes,
                      inds,
                      targets,
                      **kwargs):
        lines, normed_lines, line_scores = self(x, img_metas, contours, bboxes, inds, **kwargs)
        line_loss, score_loss, center_loss, angle_loss = \
            self.line_loss(normed_lines, line_scores, contours, targets)
        loss = 0.
        loss += line_loss * 5.
        loss += score_loss
        loss += angle_loss * 1.
        loss += center_loss * 1.
        ret = dict()
        ret.update({'line_loss': loss})
        return ret

    def simple_test(self,
                    x,
                    img_metas,
                    contours,
                    bboxes,
                    **kwargs):
        rets = self(x, img_metas, contours, bboxes, **kwargs)
        return rets

    def get_roi_coords(self, contours, cts, whs):
        num_p = contours.shape[1]
        cts = cts.unsqueeze(1).repeat(1, num_p, 1)
        whs = whs.unsqueeze(1).repeat(1, num_p, 1)
        contours = (contours - cts) / whs + 0.5
        x, y = contours[..., 0], contours[..., 1]
        x = x * self.roi_w
        y = y * self.roi_h
        return torch.cat([x[..., None], y[..., None]], dim=-1)

