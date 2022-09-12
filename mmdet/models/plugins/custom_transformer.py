import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.optimize as opt

from mmcv.runner import BaseModule
from mmcv.cnn import xavier_init, PLUGIN_LAYERS, build_norm_layer
from mmcv.cnn.bricks.registry import TRANSFORMER_LAYER_SEQUENCE
from mmcv.cnn.bricks.transformer import (TransformerLayerSequence,
                                         build_transformer_layer_sequence)


@TRANSFORMER_LAYER_SEQUENCE.register_module(force=True)
class LineDecoder(TransformerLayerSequence):
    """Implements the decoder in DETR transformer.

    Args:
        return_intermediate (bool): Whether to return intermediate outputs.
        post_norm_cfg (dict): Config of last normalization layer. Default：
            `LN`.
    """

    def __init__(self,
                 *args,
                 post_norm_cfg=dict(type='LN'),
                 return_intermediate=True,
                 **kwargs):

        super(LineDecoder, self).__init__(*args, **kwargs)
        self.return_intermediate = return_intermediate
        if post_norm_cfg is not None:
            self.post_norm = build_norm_layer(post_norm_cfg,
                                              self.embed_dims)[1]
        else:
            self.post_norm = None

    def forward(self,
                query,
                *args,
                key,
                value,
                key_pos,
                **kwargs):
        """Forward function for `TransformerDecoder`.

        Args:
            query (Tensor): Input query with shape
                `(num_query, bs, embed_dims)`.

        Returns:
            Tensor: Results with shape [1, num_query, bs, embed_dims] when
                return_intermediate is `False`, otherwise it has shape
                [num_layers, num_query, bs, embed_dims].
        """
        num_level = len(value)
        intermediate = []
        for lid, layer in enumerate(self.layers):
            query = layer(query,
                          *args,
                          key=value[lid % num_level],
                          value=value[lid % num_level],
                          key_pos=key_pos[lid % num_level],
                          **kwargs)
            if self.return_intermediate:
                if self.post_norm is not None:
                    intermediate.append(self.post_norm(query))
                else:
                    intermediate.append(query)
        return torch.stack(intermediate)

@PLUGIN_LAYERS.register_module(force=True)
class LinePredictor(BaseModule):
    def __init__(self,
                 num_query=30,
                 out_channel=4,
                 roi_wh=(40, 40),
                 line_decoder=None,
                 init_cfg=None,
                 **kwargs):
        super(LinePredictor, self).__init__(init_cfg=init_cfg)
        self.num_query = num_query
        self.out_channel = out_channel
        self.roi_w = roi_wh[0]
        self.roi_h = roi_wh[1]
        self.line_decoder = build_transformer_layer_sequence(line_decoder)
        self.num_layers = self.line_decoder.num_layers
        self.hidden_dim = self.line_decoder.embed_dims

        self._init_layers()

    def _init_layers(self):
        self.predictor_line = MLP(self.hidden_dim * 2, self.hidden_dim * 2,
                                  self.out_channel, 3)
        self.predictor_score = nn.Linear(self.hidden_dim, 2)
        self.pos_off_predictor = MLP(self.hidden_dim, self.hidden_dim * 2,
                                     self.hidden_dim, 3)

    def init_weights(self):
        # follow the official DETR to init parameters
        for m in self.modules():
            if hasattr(m, 'weight') and m.weight.dim() > 1:
                xavier_init(m, distribution='uniform')
        self._is_init = True

    def get_image_coords(self, roi_coords, ct, roi_wh, ro=4.):
        """
        ct (Tensor): Shape (ngt, 2).
        roi_wh (Tensor): Original wh before RoIAlign.
            Shape (ngt, 2).
        """
        x, y = roi_coords[..., :1], roi_coords[..., 1:] # nl, ngt, nq, 1
        x = x / self.roi_w
        y = y / self.roi_h
        coords = torch.cat([x, y], dim=-1) # nl, ngt, nq, 2
        coords = (coords - 0.5) * roi_wh[None, :, None, :] + ct[None, :, None, :]
        return coords * ro

    def forward(self,
                query,
                query_pos,
                mlvl_feats,
                mlvl_pos_embeds,
                ct,
                roi_wh,
                **kwargs):
        """Forward function for 'LinePredictor'.

        Args:
            query (Tensor): Initialized queries. Shape [nq, ngt, c].
            query_pos (Tensor): Initialized queries' pos_embeds.
                Shape [nq, ngt, c].
            mlvl_feats (list[Tensor]): Input multi-level roi features.
                Each element has shape [h*w, ngt, c].
            mlvl_pos_embeds (list[Tensor]): Corresponding pos_embeds of
                mlvl_feats. Each element has shape [h*w, ngt, c].
        """
        # hs (num_layers, nq, ngt, c)
        hs = self.line_decoder(
            query=query,
            key=mlvl_feats,
            value=mlvl_feats,
            key_pos=mlvl_pos_embeds,
            query_pos=query_pos,
            key_padding_mask=None)

        query_pos = query_pos.unsqueeze(0).expand(hs.shape[0], -1, -1, -1)
        queries_with_pos = torch.cat([hs, query_pos], dim=-1) # (nl, nq, ngt, 2c)
        line_offsets = self.predictor_line(queries_with_pos).transpose(1, 2) # (nl, ngt, nq, 4)
        line_offsets = line_offsets.sigmoid()
        line_scores = self.predictor_score(hs).transpose(1, 2) # (nl, ngt, nq, 2)
        # pos_off = self.pos_off_predictor(hs) # (nl, nq, ngt, c)
        stride = torch.tensor([self.roi_w, self.roi_h, self.roi_w, self.roi_h], device=query.device)
        normed_lines = (line_offsets[..., :2], line_offsets[..., 2:])
        line_offsets = line_offsets * stride
        p1 = line_offsets[..., :2] # (nl, ngt, nq, 2)
        p2 = line_offsets[..., 2:] # (nl, ngt, nq, 2)
        p1 = self.get_image_coords(p1, ct, roi_wh)
        p2 = self.get_image_coords(p2, ct, roi_wh)

        lines = [(p1[i], p2[i]) for i in range(len(hs))]
        normed_lines = [(normed_lines[0][i], normed_lines[1][i]) for i in range(len(hs))]
        line_scores = [line_scores[i] for i in range(len(hs))]

        return lines, normed_lines, line_scores#, query_pos + pos_off

@PLUGIN_LAYERS.register_module(force=True)
class LineCriter(BaseModule):
    def __init__(self,
                 score_weight=1.,
                 line_weight=5.,
                 point_weight=1.):
        super(LineCriter, self).__init__()
        self.l1 = torch.nn.functional.l1_loss
        self.matcher = Matcher(score_weight, line_weight, point_weight)
        empty_weight = torch.ones(2)
        empty_weight[-1] = 0.1
        self.register_buffer("empty_weight", empty_weight)

    def line_loss(self, pred_lines, targets, idxs):
        pred_lines = torch.cat([pred_lines[0], pred_lines[1]], dim=-1)
        matched_pred_lines = []
        matched_target = []
        for pred_line, target, idx in zip(pred_lines, targets, idxs):
            pred_idx, gt_idx = idx
            matched_pred_lines.append(pred_line[pred_idx])
            matched_target.append(target[gt_idx])
        matched_pred_lines = torch.cat(matched_pred_lines, dim=0)
        matched_target = torch.cat(matched_target, dim=0)
        loss = torch.mean(self.l1(matched_pred_lines, matched_target, reduction='none'), dim=-1)
        loss_ = torch.sum(self.l1(matched_pred_lines, torch.cat([matched_target[..., -2:],
                                                                 matched_target[..., :2]], dim=-1), reduction='none'),
                          dim=-1)
        n_items = loss.size(0) + 1
        loss = torch.sum(torch.minimum(loss, loss_)) / n_items
        return loss

    def center_loss(self, pred_lines, targets, idxs):
        pred_lines = torch.cat([pred_lines[0], pred_lines[1]], dim=-1)
        matched_pred_lines = []
        matched_target = []
        for pred_line, target, idx in zip(pred_lines, targets, idxs):
            pred_idx, gt_idx = idx
            matched_pred_lines.append(pred_line[pred_idx])
            matched_target.append(target[gt_idx])
        matched_pred_lines = torch.cat(matched_pred_lines, dim=0)
        matched_target = torch.cat(matched_target, dim=0)
        matched_pred_centers = (matched_pred_lines[..., :2] + matched_pred_lines[..., 2:]) / 2.
        matched_target_centers = (matched_target[..., :2] + matched_target[..., 2:]) / 2.
        loss = torch.mean(self.l1(matched_pred_centers, matched_target_centers, reduction='none'), dim=-1)
        loss = torch.mean(loss)
        return loss

    def normalize_vector(self, v):
        distance = (v[..., :1] ** 2 + v[..., 1:] ** 2) ** 0.5 + 1e-4
        return v / distance

    def angle_loss(self, pred_lines, targets, idxs):
        pred_lines = torch.cat([pred_lines[0], pred_lines[1]], dim=-1)
        matched_pred_lines = []
        matched_target = []
        for pred_line, target, idx in zip(pred_lines, targets, idxs):
            pred_idx, gt_idx = idx
            matched_pred_lines.append(pred_line[pred_idx])
            matched_target.append(target[gt_idx])
        matched_pred_lines = torch.cat(matched_pred_lines, dim=0)
        matched_target = torch.cat(matched_target, dim=0)
        matched_pred_vectors = matched_pred_lines[..., :2] - matched_pred_lines[..., 2:]
        matched_target_vectors = matched_target[..., :2] - matched_target[..., 2:]
        normed_pred_vector = self.normalize_vector(matched_pred_vectors)
        normed_target_vector = self.normalize_vector(matched_target_vectors)
        loss = normed_pred_vector[..., 0] * normed_target_vector[..., 1] - \
               normed_pred_vector[..., 1] * normed_target_vector[..., 0]
        return torch.mean(torch.abs(loss))

    def score_loss(self, pred_scores, idxs):
        target = torch.full(pred_scores.shape[:2], 1,
                            dtype=torch.int64, device=pred_scores.device)
        device = pred_scores.device
        for i, idx in enumerate(idxs):
            pos_idx, _ = idx
            target[i][pos_idx] = 0

        # loss = self.label_focal_loss(pred_scores.transpose(1, 2), target)
        loss = F.cross_entropy(pred_scores.transpose(1, 2), target, self.empty_weight)
        return loss

    def label_focal_loss(self, input, target, gamma=2.0):
        """ Focal loss for label prediction. """
        prob = F.softmax(input, 1)
        ce_loss = F.cross_entropy(input, target, self.empty_weight, reduction='none')
        p_t = prob[:, 1, :] * target + prob[:, 0, :] * (1 - target)
        loss = ce_loss * ((1 - p_t) ** gamma)
        loss = loss.mean()
        return loss

    def forward_single(self, pred_lines, pred_scores, targets, idxs):
        score_loss = self.score_loss(pred_scores, idxs)
        line_loss = self.line_loss(pred_lines, targets, idxs)
        center_loss = self.center_loss(pred_lines, targets, idxs)
        angle_loss = self.angle_loss(pred_lines, targets, idxs)
        return line_loss, score_loss, center_loss, angle_loss

    def forward(self, aux_pred_lines, aux_pred_scores, pred_contours, targets):
        assert len(aux_pred_scores) == len(aux_pred_lines)
        line_loss, score_loss, center_loss, angle_loss = 0., 0., 0., 0.
        idxs = self.matcher.match(aux_pred_lines[-1], aux_pred_scores[-1], pred_contours, targets)
        for pred_lines, pred_scores in zip(aux_pred_lines, aux_pred_scores):
            single_line_loss, single_score_loss, single_center_loss, single_angle_loss = \
                self.forward_single(pred_lines, pred_scores, targets, idxs)
            line_loss += single_line_loss
            score_loss += single_score_loss
            center_loss += single_center_loss
            angle_loss += single_angle_loss
        return line_loss, score_loss, center_loss, angle_loss

class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x

class Matcher:
    def __init__(self, score_weight=1., line_weight=5., point_weight=1.):
        self.score_weight = score_weight
        self.line_weight = line_weight
        self.point_weight = point_weight

    def line_cost(self, pred_line, target_line):
        pl_pred, pr_pred = pred_line
        pred_line = torch.cat([pl_pred, pr_pred], dim=-1)
        n_pred, n_gt = pred_line.size(0), target_line.size(0)
        pred_line = pred_line.unsqueeze(1).repeat(1, n_gt, 1)
        target_line = target_line.unsqueeze(0).repeat(n_pred, 1, 1)
        cost = torch.sum((pred_line - target_line) ** 2, dim=-1) ** 0.5
        cost_ = torch.sum((pred_line - torch.cat([target_line[..., -2:], target_line[..., :2]],
                          dim=-1)) ** 2, dim=-1) ** 0.5
        return torch.minimum(cost_, cost)
        #return cost

    def point_cost(self, pred_contour, target_line):
        n_pred = pred_contour.size(0)
        n_gt = target_line.size(0)
        target_center = (target_line[..., :2] + target_line[..., -2:]) / 2.
        pred_contour = pred_contour.unsqueeze(1).repeat(1, n_gt, 1)
        target_center = target_center.unsqueeze(0).repeat(n_pred, 1, 1)
        cost = torch.sum((pred_contour - target_center) ** 2, dim=-1)

        return cost

    def match_single(self, pred_line, pred_score, pred_contour, target_line):
        cost_line = self.line_cost(pred_line, target_line)
        #cost_point = self.point_cost(pred_contour, target_line)
        n_gt = target_line.size(0)
        pred_score = pred_score[..., 0].unsqueeze(1).repeat(1, n_gt)
        cost = cost_line * self.line_weight - pred_score * self.score_weight
        cost = cost.detach().cpu().numpy()
        pred_idx, gt_idx = opt.linear_sum_assignment(cost)
        return pred_idx, gt_idx

    def match(self, pred_lines, pred_scores, pred_contours, target_lines):
        #pred lines(Tensor(n, 128, 2), Tensor(n, 128, 2))
        #pred_scores Tensor(n, 128, 1)
        #pred_contours Tensor(n, 128, 2)
        #target_lines [Tensor(n1, 2), ...]
        n = pred_contours.size(0)
        idxs = []
        for i in range(n):
            idxs.append(self.match_single((pred_lines[0][i], pred_lines[1][i]),
                                          pred_scores[i].softmax(-1), pred_contours[i], target_lines[i]))
        return idxs