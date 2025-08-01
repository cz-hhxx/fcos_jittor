import numpy as np
import jittor as jt


def cls2onehot(cls_map, n_clses):
    hw = False
    if len(cls_map.shape) == 2:
        hw = True
        h, w = cls_map.shape
        cls_map = cls_map.flatten()
    bhw = False
    if len(cls_map.shape) == 3:
        bhw = True
        b, h, w = cls_map.shape
        cls_map = cls_map.flatten()
    cls_idx = jt.arange(1, n_clses + 1).unsqueeze(0)
    cls_map = cls_map.unsqueeze(1)
    cls_map = (cls_map == cls_idx).float()
    if hw:
        cls_map = cls_map.view(h, w, n_clses).permute(2, 0, 1)
    if bhw:
        cls_map = cls_map.view(b, h, w, n_clses).permute(0, 3, 1, 2)
    return cls_map


@jt.no_grad()
def decode_heatmap(loc_map, center_map, cls_map, scale=1., K=100, thres=0.3):  # 提高阈值到0.3
    loc_map = jt.array(loc_map).detach()
    center_map = jt.array(center_map).detach()
    cls_map = jt.array(cls_map).detach()

    if len(cls_map.shape) == 2:
        cls_map = cls_map.unsqueeze(0)
    C, H, W = cls_map.shape

    yv, xv = jt.meshgrid([jt.arange(H), jt.arange(W)])
    yv = jt.array(yv)
    xv = jt.array(xv)

    argmax_result = jt.argmax(cls_map, dim=0)
    if isinstance(argmax_result, tuple):
        cls = jt.array(argmax_result[0])
    else:
        cls = jt.array(argmax_result)
    cls = cls.long()
    yv = yv.long()
    xv = xv.long()

    # 关键修改：使用类别分数 × 中心度分数作为综合分数
    cls_score = cls_map[cls, yv, xv]
    center_score = center_map  # 中心度已通过sigmoid激活
    score = cls_score * center_score  # 综合评分

    # 使用综合分数过滤
    mask = score > thres
    indices = jt.where(mask)

    if len(indices[0]) == 0:
        return jt.empty((0, 4)), jt.empty((0,)), jt.empty((0,))

    y_indices, x_indices = indices[0], indices[1]
    flat_indices = y_indices * W + x_indices
    flat_indices = flat_indices.long()

    def take(x, indices):
        if not isinstance(x, jt.Var):
            x = jt.array(x)
        return jt.index_select(x.flatten(), 0, indices)

    x = take(xv, flat_indices)
    y = take(yv, flat_indices)
    score = take(score, flat_indices)  # 使用综合分数排序
    cls = take(cls, flat_indices)

    l = take(loc_map[0], flat_indices)
    t = take(loc_map[1], flat_indices)
    r = take(loc_map[2], flat_indices)
    b = take(loc_map[3], flat_indices)

    cx = (x.float() + 0.5) * scale
    cy = (y.float() + 0.5) * scale
    x1 = cx - l
    y1 = cy - t
    x2 = cx + r
    y2 = cy + b

    boxes = jt.stack([x1, y1, x2, y2], dim=1)

    if len(score) > K:
        sorted_indices, _ = score.argsort(descending=True)
        sorted_indices = sorted_indices[:K]
        boxes = boxes[sorted_indices]
        score = score[sorted_indices]
        cls = cls[sorted_indices]

    return boxes.numpy(), score.numpy(), cls.numpy()


@jt.no_grad()
def decode_heatmaps(loc_maps, center_maps, cls_maps, scales, thresh=.3, use_nms=True, nms_thresh=.5):  # 默认启用NMS
    boxes = []
    scores = []
    clses = []

    for i in range(len(loc_maps)):
        loc_map, center_map, cls_map = loc_maps[i], center_maps[i], cls_maps[i]
        loc_map = loc_map.squeeze()
        center_map = center_map.squeeze()
        cls_map = cls_map.squeeze()

        boxes_, scores_, clses_ = decode_heatmap(
            loc_map, center_map, cls_map, scale=scales[i], K=100, thres=thresh)

        if len(boxes_) > 0:
            boxes.append(boxes_)
            scores.append(scores_)
            clses.append(clses_)

    if len(boxes) == 0:
        return jt.empty((0, 6))

    boxes = np.concatenate(boxes, axis=0)
    scores = np.concatenate(scores, axis=0)
    clses = np.concatenate(clses, axis=0)

    results = np.hstack([np.array(clses).reshape(-1, 1),
                         np.array(scores).reshape(-1, 1),
                         boxes])

    if use_nms:
        from jittor.misc import nms as jt_nms
        dets_np = np.concatenate([
            results[:, 2:6],
            results[:, 1:2]
        ], axis=1).astype(np.float32)
        dets = jt.array(dets_np)
        keep = jt_nms(dets, nms_thresh)
        results = results[keep.numpy()]

    return jt.array(results)
