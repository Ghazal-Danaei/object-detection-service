"""Detection metrics and post-processing utilities (pure NumPy).

Deliberately dependency-light (NumPy only) so the metric / NMS logic can be
unit-tested without torch or onnxruntime installed. These are the building
blocks behind object-detection evaluation: IoU, non-maximum suppression, and
single-class Average Precision.
"""
from __future__ import annotations

import numpy as np


def iou_xyxy(box_a, box_b) -> float:
    """Intersection-over-Union of two boxes in [x1, y1, x2, y2] format."""
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter = inter_w * inter_h

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    if union <= 0.0:
        return 0.0
    return float(inter / union)


def iou_matrix(boxes_a, boxes_b) -> np.ndarray:
    """Pairwise IoU between two sets of xyxy boxes -> array of shape (Na, Nb)."""
    boxes_a = np.asarray(boxes_a, dtype=np.float64).reshape(-1, 4)
    boxes_b = np.asarray(boxes_b, dtype=np.float64).reshape(-1, 4)
    if len(boxes_a) == 0 or len(boxes_b) == 0:
        return np.zeros((len(boxes_a), len(boxes_b)), dtype=np.float64)

    area_a = ((boxes_a[:, 2] - boxes_a[:, 0]).clip(min=0)
              * (boxes_a[:, 3] - boxes_a[:, 1]).clip(min=0))
    area_b = ((boxes_b[:, 2] - boxes_b[:, 0]).clip(min=0)
              * (boxes_b[:, 3] - boxes_b[:, 1]).clip(min=0))

    inter_x1 = np.maximum(boxes_a[:, None, 0], boxes_b[None, :, 0])
    inter_y1 = np.maximum(boxes_a[:, None, 1], boxes_b[None, :, 1])
    inter_x2 = np.minimum(boxes_a[:, None, 2], boxes_b[None, :, 2])
    inter_y2 = np.minimum(boxes_a[:, None, 3], boxes_b[None, :, 3])

    inter_w = (inter_x2 - inter_x1).clip(min=0)
    inter_h = (inter_y2 - inter_y1).clip(min=0)
    inter = inter_w * inter_h

    union = area_a[:, None] + area_b[None, :] - inter
    with np.errstate(divide="ignore", invalid="ignore"):
        iou = np.where(union > 0, inter / union, 0.0)
    return iou


def nms(boxes, scores, iou_threshold: float = 0.45):
    """Greedy non-maximum suppression. Returns indices of the kept boxes."""
    boxes = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    if len(boxes) == 0:
        return []

    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(int(i))
        if order.size == 1:
            break
        rest = order[1:]
        ious = iou_matrix(boxes[i:i + 1], boxes[rest]).reshape(-1)
        order = rest[ious <= iou_threshold]
    return keep


def average_precision(pred_boxes, pred_scores, gt_boxes,
                      iou_threshold: float = 0.5) -> float:
    """Single-class Average Precision (area under the PR curve, VOC all-points).

    Args:
        pred_boxes: (N, 4) predicted xyxy boxes for one class.
        pred_scores: (N,) confidence scores.
        gt_boxes: (M, 4) ground-truth xyxy boxes for the same class.
        iou_threshold: IoU needed to count a prediction as a true positive.

    Returns:
        AP in [0, 1].
    """
    pred_boxes = np.asarray(pred_boxes, dtype=np.float64).reshape(-1, 4)
    pred_scores = np.asarray(pred_scores, dtype=np.float64).reshape(-1)
    gt_boxes = np.asarray(gt_boxes, dtype=np.float64).reshape(-1, 4)

    n_gt = len(gt_boxes)
    if len(pred_boxes) == 0:
        return 0.0 if n_gt > 0 else 1.0

    order = pred_scores.argsort()[::-1]
    pred_boxes = pred_boxes[order]

    matched = np.zeros(n_gt, dtype=bool)
    tp = np.zeros(len(pred_boxes), dtype=np.float64)
    fp = np.zeros(len(pred_boxes), dtype=np.float64)

    ious = iou_matrix(pred_boxes, gt_boxes) if n_gt > 0 else None

    for i in range(len(pred_boxes)):
        if n_gt == 0:
            fp[i] = 1
            continue
        gt_idx = int(np.argmax(ious[i]))
        best_iou = ious[i, gt_idx]
        if best_iou >= iou_threshold and not matched[gt_idx]:
            tp[i] = 1
            matched[gt_idx] = True
        else:
            fp[i] = 1

    tp_cum = np.cumsum(tp)
    fp_cum = np.cumsum(fp)
    recalls = tp_cum / (n_gt + 1e-12)
    precisions = tp_cum / (tp_cum + fp_cum + 1e-12)

    # VOC all-points interpolation: area under the (monotonic) PR curve.
    mrec = np.concatenate(([0.0], recalls, [1.0]))
    mpre = np.concatenate(([0.0], precisions, [0.0]))
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = max(mpre[i - 1], mpre[i])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    ap = float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))
    return ap
