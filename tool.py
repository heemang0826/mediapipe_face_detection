from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np


def save_annotations(detection_dict, save_path):
    """
    Unlike the original WIDER FACE ground truth format: (x, y, w, h, blur, expression, illumination, invalid, occlusion, pose),
    this code saves detections in the simplified format: (x, y, w, h, score).
    """
    with open(save_path, "w") as f:
        for img_path, detections in detection_dict.items():
            f.write(f"{img_path}\n")
            f.write(f"{len(detections)}\n")
            for x, y, w, h, score in detections:
                f.write(f"{int(x)} {int(y)} {int(w)} {int(h)} {score:.6f}\n")


def parse(path, type: str):
    """
    Parses annotation files for ground truth or prediction results.

    Expected input file format:
        Each image has:
        - One line with image path
        - One line with the number of bounding boxes
        - N lines of bounding boxes in the format:
            For ground truth (type="gt")  : x, y, w, h
            For predictions (type="pred"): x, y, w, h, score

    Returns:
        dict: {
            "img_path1": [
                [x1, y1, x2, y2],               # if type == "gt"
                [x1, y1, x2, y2, score],        # if type == "pred"
                ...
            ],
            ...
        }
    """
    with open(path, "r") as f:
        lines = [line.strip() for line in f]

    data = {}
    i = 0
    while i < len(lines):
        img_path = lines[i]
        i += 1
        num_boxes = int(lines[i])
        i += 1

        boxes = []
        for _ in range(num_boxes):
            parts = list(map(float, lines[i].split()))
            x, y, w, h = parts[:4]
            x2, y2 = x + w, y + h
            if type == "gt":
                boxes.append([int(x), int(y), int(x2), int(y2)])
            elif type == "pred":
                score = parts[4]
                boxes.append([x, y, x2, y2, score])
            else:
                raise ValueError("Invalid type. Use 'gt' or 'pred'.")
            i += 1

        data[img_path] = boxes

    return data


def iou(box1, box2):
    x1, y1, x2, y2 = box1
    x1g, y1g, x2g, y2g = box2

    xi1 = max(x1, x1g)
    yi1 = max(y1, y1g)
    xi2 = min(x2, x2g)
    yi2 = min(y2, y2g)
    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    if inter_area == 0:
        return 0.0
    union_area = (x2 - x1) * (y2 - y1) + (x2g - x1g) * (y2g - y1g) - inter_area
    return inter_area / union_area


def evaluate(
    gt_boxes: Dict[str, List[List[float]]],
    pred_boxes: Dict[str, List[List[float]]],
    iou_thre=0.4,
    conf_thre=0.01,
):
    all_detections = []
    total_gt = 0

    for img_id, gt in gt_boxes.items():
        preds = pred_boxes.get(img_id, [])
        total_gt += len(gt)
        gt_matched = [False] * len(gt)

        preds_sorted = sorted(
            [p for p in preds if p[4] >= conf_thre], key=lambda x: -x[4]
        )

        for pred in preds_sorted:
            px1, py1, px2, py2, score = pred
            matched = False
            for i, gt_box in enumerate(gt):
                if not gt_matched[i] and iou([px1, py1, px2, py2], gt_box) >= iou_thre:
                    gt_matched[i] = True
                    matched = True
                    break
            all_detections.append(
                {"score": score, "tp": 1 if matched else 0, "fp": 0 if matched else 1}
            )

    all_detections.sort(key=lambda x: -x["score"])
    tp = np.array([d["tp"] for d in all_detections])
    fp = np.array([d["fp"] for d in all_detections])

    if total_gt == 0:
        return np.array([]), np.array([]), 0.0

    cum_tp = np.cumsum(tp)
    cum_fp = np.cumsum(fp)

    recall = cum_tp / total_gt
    precision = cum_tp / (cum_tp + cum_fp + 1e-6)

    for i in range(len(precision) - 2, -1, -1):
        precision[i] = max(precision[i], precision[i + 1])

    ap = np.trapz(precision, recall)

    print(f"AP@IoU={iou_thre:.2f}={ap:.2f}")
    return precision, recall, ap


def plot_pr_curve(precision, recall, ap, save_path):
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, label=f"AP = {ap:.4f}", color="blue", linewidth=2)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.grid(True)
    plt.legend()
    plt.xlim([0, max(recall)])
    plt.ylim([0, max(precision)])
    plt.savefig(save_path)
