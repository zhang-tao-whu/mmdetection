# import copy
# import inspect
import math
# import warnings
#
# import cv2
# import mmcv
import numpy as np
# from numpy import random
# from numpy.fft import fft
#
# from mmdet.core import BitmapMasks, PolygonMasks, find_inside_bboxes
# from mmdet.core.evaluation.bbox_overlaps import bbox_overlaps
# from mmdet.utils import log_img_scale
from ..builder import PIPELINES
from shapely.geometry import Polygon
from shapely.ops import clip_by_rect

from mmcv.parallel import DataContainer as DC
from .formatting import to_tensor
try:
    from imagecorruptions import corrupt
except ImportError:
    corrupt = None

try:
    import albumentations
    from albumentations import Compose
except ImportError:
    albumentations = None
    Compose = None

@PIPELINES.register_module(force=True)
class AlignSampleBoundaryLine:
    def __init__(self,
                 point_nums=128,
                 reset_bbox=True):
        self.point_nums = point_nums
        self.reset_bbox = reset_bbox
        self.d = Douglas()
        self.Line = Matcher_Point_Line()

    def __call__(self, results):
        gt_masks = results['gt_masks']
        gt_labels = results['gt_labels']
        gt_polys = gt_masks.masks
        # height, width = gt_masks.height, gt_masks.width
        sampled_polys, keyPointsMask, key_points_list = [], [], []
        sampled_lines, sampled_normed_lines, num_lines = [], [], []
        if self.reset_bbox:
            reset_bboxes = []
            reset_labels = []

        for gt_poly, label in zip(gt_polys, gt_labels):
            for comp_poly in gt_poly:
                poly = comp_poly.reshape(-1, 2).astype(np.float32)
                if len(poly) < 3:
                    continue
                bbox = np.concatenate([np.min(poly, axis=0), np.max(poly, axis=0)], axis=0)
                h, w = bbox[3] - bbox[1], bbox[2] - bbox[0]
                if h <= 1 or w <= 1:
                    continue
                succeed = self.prepare_evolution(poly, sampled_polys, keyPointsMask, key_points_list,
                                                 sampled_lines, sampled_normed_lines, num_lines)
                if succeed and self.reset_bbox:
                    reset_labels.append(label)
                    reset_bboxes.append(bbox)

        if len(sampled_polys) != 0:
            results['gt_polys'] = np.stack(sampled_polys, axis=0)
            results['key_points_masks'] = np.stack(keyPointsMask, axis=0)
            results['key_points'] = np.stack(key_points_list, axis=0)
            results['gt_lines'] = np.concatenate(sampled_normed_lines, axis=0)
            results['num_lines'] = np.array(num_lines, dtype=np.int64)
            if self.reset_bbox:
                results['gt_labels'] = np.stack(reset_labels, axis=0)
                results['gt_bboxes'] = np.stack(reset_bboxes, axis=0)
        else:
            results['gt_polys'] = np.zeros((0, 128, 2), dtype=np.float32)
            results['key_points_masks'] = np.zeros((0, 128, ), dtype=np.int64)
            results['key_points'] = np.zeros((0, 128, 2), dtype=np.float32)
            results['gt_lines'] = np.zeros((0, 128, 2), dtype=np.float32)
            results['num_lines'] = np.zeros((0, ), dtype=np.int64)
            if self.reset_bbox:
                results['gt_labels'] = np.zeros((0, ), dtype=np.int64)
                results['gt_bboxes'] = np.zeros((0, 4), dtype=np.float32)
        return results

    def ignore_poly(self, idxs):
        idxs = np.array(idxs)
        max_idx = np.argmax(idxs)
        idxs = np.roll(idxs, -max_idx - 1)
        ret = idxs[1] > idxs[0] and idxs[2] > idxs[1] and idxs[3] > idxs[2]
        return not ret

    def unique(self, poly):
        poly_ = np.roll(poly, 1)
        dis = np.sum((poly - poly_) ** 2, axis=1) ** 0.5
        valid = dis >= 0.1
        return poly[valid]

    def prepare_evolution(self, poly, img_gt_polys, keyPointsMask, key_points_list,
                          sampled_lines, sampled_normed_lines, num_lines):
        poly = self.unique(poly)
        poly = self.get_cw_polys(poly)
        ori_nodes = len(poly)
        key_points = poly
        img_gt_poly = self.uniformsample(poly, ori_nodes * self.point_nums)
        idx = self.four_idx(img_gt_poly)
        if self.ignore_poly(idx):
            return False
        img_gt_poly = self.get_img_gt(img_gt_poly, idx, t=self.point_nums)
        key_mask = self.get_keypoints_mask(key_points)
        key_points = key_points[key_mask.astype(np.bool)]
        if len(key_points) >= self.point_nums:
            key_points = key_points[:self.point_nums]
            item = self.Line.get(img_gt_poly, key_points)
            key_mask = np.ones((self.point_nums, ), dtype=np.int64)
        else:
            item = self.Line.get(img_gt_poly, key_points)
            temp = np.zeros((self.point_nums, 2), dtype=np.float32)
            temp[:len(key_points)] = key_points
            key_mask = np.zeros((self.point_nums,), dtype=np.int64)
            key_mask[:len(key_points)] = 1
            key_points = temp

        keyPointsMask.append(key_mask)
        img_gt_polys.append(img_gt_poly)
        key_points_list.append(key_points)
        sampled_lines.append(item['lines'])
        sampled_normed_lines.append(item['normed_lines']) # each (nl, 2)
        num_lines.append(len(item['normed_lines']))
        return True

    def get_cw_polys(self, poly):
        return poly[::-1] if Polygon(poly).exterior.is_ccw else poly

    @staticmethod
    def uniformsample(pgtnp_px2, newpnum):
        pnum, cnum = pgtnp_px2.shape
        assert cnum == 2

        idxnext_p = (np.arange(pnum, dtype=np.int32) + 1) % pnum
        pgtnext_px2 = pgtnp_px2[idxnext_p]
        edgelen_p = np.sqrt(np.sum((pgtnext_px2 - pgtnp_px2) ** 2, axis=1))
        edgeidxsort_p = np.argsort(edgelen_p)

        # two cases
        # we need to remove gt points
        # we simply remove shortest paths
        if pnum > newpnum:
            edgeidxkeep_k = edgeidxsort_p[pnum - newpnum:]
            edgeidxsort_k = np.sort(edgeidxkeep_k)
            pgtnp_kx2 = pgtnp_px2[edgeidxsort_k]
            assert pgtnp_kx2.shape[0] == newpnum
            return pgtnp_kx2
        # we need to add gt points
        # we simply add it uniformly
        else:
            edgenum = np.round(edgelen_p * newpnum / np.sum(edgelen_p)).astype(np.int32)
            for i in range(pnum):
                if edgenum[i] == 0:
                    edgenum[i] = 1

            # after round, it may has 1 or 2 mismatch
            edgenumsum = np.sum(edgenum)
            if edgenumsum != newpnum:

                if edgenumsum > newpnum:

                    id = -1
                    passnum = edgenumsum - newpnum
                    while passnum > 0:
                        edgeid = edgeidxsort_p[id]
                        if edgenum[edgeid] > passnum:
                            edgenum[edgeid] -= passnum
                            passnum -= passnum
                        else:
                            passnum -= edgenum[edgeid] - 1
                            edgenum[edgeid] -= edgenum[edgeid] - 1
                            id -= 1
                else:
                    id = -1
                    edgeid = edgeidxsort_p[id]
                    edgenum[edgeid] += newpnum - edgenumsum

            assert np.sum(edgenum) == newpnum

            psample = []
            for i in range(pnum):
                pb_1x2 = pgtnp_px2[i:i + 1]
                pe_1x2 = pgtnext_px2[i:i + 1]

                pnewnum = edgenum[i]
                wnp_kx1 = np.arange(edgenum[i], dtype=np.float32).reshape(-1, 1) / edgenum[i]

                pmids = pb_1x2 * (1 - wnp_kx1) + pe_1x2 * wnp_kx1
                psample.append(pmids)

            psamplenp = np.concatenate(psample, axis=0)
            return psamplenp

    @staticmethod
    def four_idx(img_gt_poly):
        x_min, y_min = np.min(img_gt_poly, axis=0)
        x_max, y_max = np.max(img_gt_poly, axis=0)
        center = [(x_min + x_max) / 2., (y_min + y_max) / 2.]
        can_gt_polys = img_gt_poly.copy()
        can_gt_polys[:, 0] -= center[0]
        can_gt_polys[:, 1] -= center[1]
        distance = np.sum(can_gt_polys ** 2, axis=1, keepdims=True) ** 0.5 + 1e-6
        can_gt_polys /= np.repeat(distance, axis=1, repeats=2)
        idx_bottom = np.argmax(can_gt_polys[:, 1])
        idx_top = np.argmin(can_gt_polys[:, 1])
        idx_right = np.argmax(can_gt_polys[:, 0])
        idx_left = np.argmin(can_gt_polys[:, 0])
        return [idx_bottom, idx_right, idx_top, idx_left]

    @staticmethod
    def get_img_gt(img_gt_poly, idx, t=128):
        align = len(idx)
        pointsNum = img_gt_poly.shape[0]
        r = []
        k = np.arange(0, t / align, dtype=float) / (t / align)
        for i in range(align):
            begin = idx[i]
            end = idx[(i + 1) % align]
            if begin > end:
                end += pointsNum
            r.append((np.round(((end - begin) * k).astype(int)) + begin) % pointsNum)
        r = np.concatenate(r, axis=0)
        return img_gt_poly[r, :]

    def get_keypoints_mask(self, img_gt_poly):
        key_mask = self.d.sample(img_gt_poly)
        return key_mask

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(poly_nums={self.point_nums}, '
        return repr_str

class Douglas:
    D = 1
    def sample(self, poly):
        mask = np.zeros((poly.shape[0],), dtype=int)
        mask[0] = 1
        endPoint = poly[0: 1, :] + poly[-1:, :]
        endPoint /= 2
        poly_append = np.concatenate([poly, endPoint], axis=0)
        self.compress(0, poly.shape[0], poly_append, mask)
        return mask

    def compress(self, idx1, idx2, poly, mask):
        p1 = poly[idx1, :]
        p2 = poly[idx2, :]
        A = (p1[1] - p2[1])
        B = (p2[0] - p1[0])
        C = (p1[0] * p2[1] - p2[0] * p1[1])

        m = idx1
        n = idx2
        if (n == m + 1):
            return
        d = abs(A * poly[m + 1: n, 0] + B * poly[m + 1: n, 1] + C) / math.sqrt(math.pow(A, 2) + math.pow(B, 2) + 1e-4)
        max_idx = np.argmax(d)
        dmax = d[max_idx]
        max_idx = max_idx + m + 1

        if dmax > self.D:
            mask[max_idx] = 1
            self.compress(idx1, max_idx, poly, mask)
            self.compress(max_idx, idx2, poly, mask)


class Matcher_Point_Line:
    def __init__(self):
        pass

    def get_lines_from_polygon(self, poly):
        pl = poly
        pr = np.roll(poly, 1, axis=0)
        return (pl, pr)

    def distance_points_lines(self, points, line):
        pl, pr = line
        num_p, num_l = points.shape[0], pl.shape[0]
        points_ = np.expand_dims(points, axis=1).repeat(num_l, axis=1)
        pl_ = np.expand_dims(pl, axis=0).repeat(num_p, axis=0)
        pr_ = np.expand_dims(pr, axis=0).repeat(num_p, axis=0)

        vpl = pl_ - points_
        vpr = pr_ - points_
        area = vpl[..., 0] * vpr[..., 0] + vpl[..., 1] * vpr[..., 1]
        length = np.sum((pl_ - pr_) ** 2, axis=-1) ** 0.5
        distance = area / (length + 1e-8)
        return distance

    def get_nearet_line(self, points, line):
        distance = self.distance_points_lines(points, line)
        # distance (np, nl)
        idx = np.argmin(distance, axis=1)
        pl, pr = line
        pl_, pr_ = pl[idx], pr[idx]
        return (pl_, pr_)

    def centerness_score(self, points, line):
        pl, pr = line
        ll = np.sum((points - pl) ** 2, axis=-1) ** 0.5
        lr = np.sum((points - pr) ** 2, axis=-1) ** 0.5
        l = np.sum((pl - pr) ** 2, axis=-1) ** 0.5
        score = 1.0 - np.abs(ll - lr) / (l + 1e-8)
        return score

    def get_bbox_from_poly(self, poly):
        min_coor = np.min(poly, axis=0)
        max_coor = np.max(poly, axis=0)
        center = (min_coor + max_coor) / 2.
        wh = max_coor - min_coor
        return np.concatenate([center, wh], axis=0)

    def get_normalized_lines(self, poly, expand_ratio=1.1):
        min_coor = np.min(poly, axis=0)
        max_coor = np.max(poly, axis=0)
        poly[..., 0] = (poly[..., 0] - min_coor[0]) / (max_coor[0] - min_coor[0] + 1e-4)
        poly[..., 1] = (poly[..., 1] - min_coor[1]) / (max_coor[1] - min_coor[1] + 1e-4)
        poly = (poly - 0.5) / expand_ratio + 0.5
        pl = poly
        pr = np.roll(poly, 1, axis=0)
        return (pl, pr)

    def format_line(self, lines):
        pl, pr = lines
        return np.concatenate([pl, pr], axis=-1)

    def get(self, points, ori_polygon):
        # points (128, 2), ori_polygon (n, 2)
        lines = self.get_lines_from_polygon(ori_polygon)
        bbox = self.get_bbox_from_poly(ori_polygon)
        normed_lines = self.get_normalized_lines(ori_polygon)
        ret = {'bbox': bbox, 'lines': self.format_line(lines) * 4.,
               'normed_lines': self.format_line(normed_lines)}
        return ret

@PIPELINES.register_module(force=True)
class ContourLineDefaultFormatBundle:
    """Default formatting bundle.

    It simplifies the pipeline of formatting common fields, including "img",
    "proposals", "gt_bboxes", "gt_labels", "gt_masks" and "gt_semantic_seg".
    These fields are formatted as follows.

    - img: (1)transpose, (2)to tensor, (3)to DataContainer (stack=True)
    - proposals: (1)to tensor, (2)to DataContainer
    - gt_bboxes: (1)to tensor, (2)to DataContainer
    - gt_bboxes_ignore: (1)to tensor, (2)to DataContainer
    - gt_labels: (1)to tensor, (2)to DataContainer
    - gt_masks: (1)to tensor, (2)to DataContainer (cpu_only=True)
    - gt_semantic_seg: (1)unsqueeze dim-0 (2)to tensor, \
                       (3)to DataContainer (stack=True)

    Args:
        img_to_float (bool): Whether to force the image to be converted to
            float type. Default: True.
        pad_val (dict): A dict for padding value in batch collating,
            the default value is `dict(img=0, masks=0, seg=255)`.
            Without this argument, the padding value of "gt_semantic_seg"
            will be set to 0 by default, which should be 255.
    """

    def __init__(self,
                 img_to_float=True,
                 pad_val=dict(img=0, masks=0, seg=255)):
        self.img_to_float = img_to_float
        self.pad_val = pad_val

    def __call__(self, results):
        """Call function to transform and format common fields in results.

        Args:
            results (dict): Result dict contains the data to convert.

        Returns:
            dict: The result dict contains the data that is formatted with \
                default bundle.
        """

        if 'img' in results:
            img = results['img']
            if self.img_to_float is True and img.dtype == np.uint8:
                # Normally, image is of uint8 type without normalization.
                # At this time, it needs to be forced to be converted to
                # flot32, otherwise the model training and inference
                # will be wrong. Only used for YOLOX currently .
                img = img.astype(np.float32)
            # add default meta keys
            results = self._add_default_meta_keys(results)
            if len(img.shape) < 3:
                img = np.expand_dims(img, -1)
            img = np.ascontiguousarray(img.transpose(2, 0, 1))
            results['img'] = DC(
                to_tensor(img), padding_value=self.pad_val['img'], stack=True)
        for key in ['proposals', 'gt_bboxes', 'gt_bboxes_ignore', 'gt_labels',
                    'gt_polys', 'key_points_masks', 'key_points', 'gt_lines', 'num_lines']:
            if key not in results:
                continue
            results[key] = DC(to_tensor(results[key]))
        if 'gt_masks' in results:
            results['gt_masks'] = DC(
                results['gt_masks'],
                padding_value=self.pad_val['masks'],
                cpu_only=True)
        if 'gt_semantic_seg' in results:
            results['gt_semantic_seg'] = DC(
                to_tensor(results['gt_semantic_seg'][None, ...]),
                padding_value=self.pad_val['seg'],
                stack=True)
        return results

    def _add_default_meta_keys(self, results):
        """Add default meta keys.

        We set default meta keys including `pad_shape`, `scale_factor` and
        `img_norm_cfg` to avoid the case where no `Resize`, `Normalize` and
        `Pad` are implemented during the whole pipeline.

        Args:
            results (dict): Result dict contains the data to convert.

        Returns:
            results (dict): Updated result dict contains the data to convert.
        """
        img = results['img']
        results.setdefault('pad_shape', img.shape)
        results.setdefault('scale_factor', 1.0)
        num_channels = 1 if len(img.shape) < 3 else img.shape[2]
        results.setdefault(
            'img_norm_cfg',
            dict(
                mean=np.zeros(num_channels, dtype=np.float32),
                std=np.ones(num_channels, dtype=np.float32),
                to_rgb=False))
        return results

    def __repr__(self):
        return self.__class__.__name__ + \
               f'(img_to_float={self.img_to_float})'