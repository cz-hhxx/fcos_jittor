import jittor as jt
from jittor import nn
import jittor.nn as F

from utils import cls2onehot


class LocLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def execute(self, loc_map_pred, loc_map_gt):
        # 回归损失，采用IoU Loss
        l_pred, t_pred, r_pred, b_pred = loc_map_pred[:, 0], loc_map_pred[:, 1], loc_map_pred[:, 2], loc_map_pred[:, 3]
        l_gt, t_gt, r_gt, b_gt = loc_map_gt[:, 0], loc_map_gt[:, 1], loc_map_gt[:, 2], loc_map_gt[:, 3]
        l_max, t_max, r_max, b_max = jt.maximum(l_pred, l_gt), jt.maximum(t_pred, t_gt), jt.maximum(r_pred,
                                                                                                    r_gt), jt.maximum(
            b_pred, b_gt)

        area_pred = (l_pred + r_pred) * (t_pred + b_pred)
        area_gt = (l_gt + r_gt) * (t_gt + b_gt)

        w_union = jt.minimum(r_pred, r_gt) + jt.minimum(l_pred, l_gt)
        h_union = jt.minimum(b_pred, b_gt) + jt.minimum(t_pred, t_gt)

        area_intersect = w_union * h_union
        area_union = area_gt + area_pred - area_intersect

        iou = area_intersect / area_union.clamp(1e-6)
        iou_loss = 1 - iou

        return iou_loss.sum()


class CenterLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def execute(self, center_map_pred, center_map_gt):
        """
        中心度损失，采用 BCE Loss（Jittor 实现，与 PyTorch 对齐）
        :param center_map_pred <jt.Var>: 模型输出的中心度预测 (未经过 Sigmoid，shape 与 gt 一致)
        :param center_map_gt <jt.Var>: 真实中心度标签 (shape 与 pred 一致)
        """
        # 直接使用 jittor 官方 BCEWithLogits 函数，自动处理 Sigmoid + BCE 计算
        return nn.binary_cross_entropy_with_logits(
            center_map_pred,
            center_map_gt,
            weight=None,
            pos_weight=None,
            size_average=False  # 与 PyTorch 中 reduction='sum' 对齐
        )


class ClsLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def execute(self, cls_map_pred, cls_map_gt):
        # 类别损失，采用Focal Loss
        n_clses = cls_map_pred.shape[1]
        cls_map_gt = cls2onehot(cls_map_gt, n_clses)

        gamma = 2.
        alpha = .25

        pt = cls_map_pred * cls_map_gt + (1. - cls_map_pred) * (1. - cls_map_gt)
        # pt = jt.clamp(pt, min_v=1e-8, max_v=1 - 1e-8)
        w = alpha * cls_map_gt + (1. - alpha) * (1 - cls_map_gt)
        loss = -w * jt.pow((1.0 - pt), gamma) * jt.log(pt)

        return loss.sum()


class Loss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loc_criterion = LocLoss()
        self.center_criterion = CenterLoss()
        self.cls_criterion = ClsLoss()

    def execute(self, loc_maps_pred, loc_maps_gt,
                center_maps_pred, center_maps_gt,
                cls_maps_pred, cls_maps_gt, masks):
        n_layers = len(loc_maps_pred)

        loc_map_pred_f, loc_map_gt_f = [], []
        center_map_pred_f, center_map_gt_f = [], []
        cls_map_pred_f, cls_map_gt_f = [], []
        mask_f = []

        for l in range(n_layers):
            loc_map_pred, loc_map_gt = loc_maps_pred[l], loc_maps_gt[l]
            loc_map_pred = loc_map_pred.permute(0, 2, 3, 1).reshape(-1, 4)
            loc_map_gt = loc_map_gt.permute(0, 2, 3, 1).reshape(-1, 4)
            loc_map_pred_f.append(loc_map_pred)
            loc_map_gt_f.append(loc_map_gt)

            center_map_pred, center_map_gt = center_maps_pred[l], center_maps_gt[l]
            center_map_pred = center_map_pred.reshape(-1)
            center_map_gt = center_map_gt.reshape(-1)
            center_map_pred_f.append(center_map_pred)
            center_map_gt_f.append(center_map_gt)

            cls_map_pred, cls_map_gt = cls_maps_pred[l], cls_maps_gt[l]
            n_classes = cls_map_pred.shape[1]
            cls_map_pred = cls_map_pred.permute(0, 2, 3, 1).reshape(-1, n_classes)
            cls_map_gt = cls_map_gt.reshape(-1)
            cls_map_pred_f.append(cls_map_pred)
            cls_map_gt_f.append(cls_map_gt)

            mask = masks[l].reshape(-1)
            mask_f.append(mask)

        loc_map_pred_f, loc_map_gt_f = jt.concat(loc_map_pred_f), jt.concat(loc_map_gt_f)
        center_map_pred_f, center_map_gt_f = jt.concat(center_map_pred_f), jt.concat(center_map_gt_f)
        cls_map_pred_f, cls_map_gt_f = jt.concat(cls_map_pred_f), jt.concat(cls_map_gt_f)
        mask_f = jt.concat(mask_f)
        n_pos = jt.sum(mask_f).clamp(1)

        loc_map_pred_f, loc_map_gt_f = loc_map_pred_f[mask_f], loc_map_gt_f[mask_f]
        center_map_pred_f, center_map_gt_f = center_map_pred_f[mask_f], center_map_gt_f[mask_f]

        loc_loss = self.loc_criterion(loc_map_pred_f, loc_map_gt_f) / n_pos
        center_loss = self.center_criterion(center_map_pred_f, center_map_gt_f) / n_pos
        cls_loss = self.cls_criterion(cls_map_pred_f, cls_map_gt_f) / n_pos

        return loc_loss, center_loss, cls_loss
