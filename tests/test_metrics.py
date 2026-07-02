"""Unit tests for the detection metrics / post-processing (NumPy only)."""
import numpy as np

from src.metrics import average_precision, iou_matrix, iou_xyxy, nms


def test_iou_identical_boxes():
    assert iou_xyxy([0, 0, 10, 10], [0, 0, 10, 10]) == 1.0


def test_iou_disjoint_boxes():
    assert iou_xyxy([0, 0, 10, 10], [20, 20, 30, 30]) == 0.0


def test_iou_partial_overlap():
    # boxes [0,0,2,2] and [1,0,3,2]: intersection = 2, union = 6
    assert abs(iou_xyxy([0, 0, 2, 2], [1, 0, 3, 2]) - (2 / 6)) < 1e-9


def test_iou_matrix_shape_and_values():
    a = [[0, 0, 2, 2]]
    b = [[0, 0, 2, 2], [10, 10, 12, 12]]
    m = iou_matrix(a, b)
    assert m.shape == (1, 2)
    assert abs(m[0, 0] - 1.0) < 1e-9
    assert m[0, 1] == 0.0


def test_iou_matrix_empty():
    assert iou_matrix([], [[0, 0, 1, 1]]).shape == (0, 1)


def test_nms_suppresses_overlapping():
    boxes = [[0, 0, 10, 10], [1, 1, 11, 11], [100, 100, 110, 110]]
    scores = [0.9, 0.8, 0.95]
    keep = nms(boxes, scores, iou_threshold=0.5)
    assert 2 in keep          # highest score, isolated box
    assert 0 in keep          # best of the overlapping pair
    assert 1 not in keep      # suppressed by box 0
    assert len(keep) == 2


def test_nms_empty():
    assert nms([], [], 0.5) == []


def test_average_precision_perfect():
    preds = [[0, 0, 10, 10], [50, 50, 60, 60]]
    scores = [0.9, 0.8]
    gts = [[0, 0, 10, 10], [50, 50, 60, 60]]
    assert abs(average_precision(preds, scores, gts, 0.5) - 1.0) < 1e-9


def test_average_precision_all_wrong():
    preds = [[0, 0, 10, 10]]
    scores = [0.9]
    gts = [[100, 100, 110, 110]]
    assert average_precision(preds, scores, gts, 0.5) == 0.0


def test_average_precision_no_predictions():
    assert average_precision([], [], [[0, 0, 10, 10]], 0.5) == 0.0


def test_average_precision_half_correct():
    # one correct (high score), one false positive (low score) -> AP = 0.5
    preds = [[0, 0, 10, 10], [200, 200, 210, 210]]
    scores = [0.9, 0.3]
    gts = [[0, 0, 10, 10], [50, 50, 60, 60]]
    ap = average_precision(preds, scores, gts, 0.5)
    assert abs(ap - 0.5) < 1e-9
