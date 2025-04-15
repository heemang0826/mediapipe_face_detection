import os
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix


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
                [x1, y1, x2, y2],                 # if type == "gt"
                [x1, y1, x2, y2, score],          # if type == "pred"
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


def evaluate(gt_boxes, pred_boxes, iou_threshold=0.5):
    all_scores = []
    all_tp = []
    all_fp = []
    total_gt = 0
    y_true = []
    y_pred = []

    for img, gt in gt_boxes.items():
        preds = pred_boxes.get(img, [])
        gt_flags = [False] * len(gt)
        matched_flags = [False] * len(preds)
        total_gt += len(gt)

        preds_sorted = sorted(preds, key=lambda x: -x[4])  # sort by score

        for pred in preds_sorted:
            px1, py1, px2, py2, score = pred
            matched = False
            for i, gt_box in enumerate(gt):
                if (
                    not gt_flags[i]
                    and iou([px1, py1, px2, py2], gt_box) >= iou_threshold
                ):
                    gt_flags[i] = True
                    matched = True
                    matched_flags[matched_flags.index(False)] = True
                    break
            all_scores.append(score)
            all_tp.append(1 if matched else 0)
            all_fp.append(0 if matched else 1)

        # For confusion matrix
        tp = sum(matched_flags)
        fp = len(preds) - tp
        fn = len(gt) - sum(gt_flags)

        y_true += [1] * tp + [0] * fp + [1] * fn
        y_pred += [1] * tp + [1] * fp + [0] * fn

    cm = confusion_matrix(y_true, y_pred)
    print()
    print("                Pred: Face    Pred: No Face")
    print(f"GT: Face         {cm[1,1]:>6}          {cm[1,0]:>6}")
    print(f"GT: No Face      {cm[0,1]:>6}          {cm[0,0]:>6}")
    print()

    # Now, compute the metrics
    scores = np.array(all_scores)
    tp = np.array(all_tp)
    fp = np.array(all_fp)

    # sort by descending score
    indices = np.argsort(-scores)
    tp = tp[indices]
    fp = fp[indices]

    cum_tp = np.cumsum(tp)
    cum_fp = np.cumsum(fp)
    recall = cum_tp / total_gt
    precision = cum_tp / (cum_tp + cum_fp + 1e-6)

    # 11-point interpolated AP
    ap = 0
    for t in np.linspace(0, 1, 11):
        p = precision[recall >= t]
        ap += np.max(p) if p.size else 0
    ap /= 11

    print(f"AP: {ap:.4f}")

    return precision, recall, ap


def save_pr_curve(precision, recall, ap, save_dir):
    timestamp = datetime.now().strftime("%m%d_%H%M%S")
    filename = f"{timestamp}.png"
    save_path = os.path.join(save_dir, filename)

    plt.figure(figsize=(7, 5))
    plt.plot(
        recall, precision, marker=".", color="blue", linewidth=1, label=f"AP = {ap:.4f}"
    )
    plt.title("Precision-Recall Curve")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.grid(True)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
