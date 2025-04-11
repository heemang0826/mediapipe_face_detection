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
    with open(path, "r") as f:
        lines = [line.strip() for line in f.readlines()]

    data = {}
    i = 0
    while i < len(lines):
        img = lines[i]
        i += 1
        num = int(lines[i])
        i += 1
        boxes = []
        for _ in range(num):
            if type == "gt":
                parts = list(map(int, lines[i].split()[:4]))
                x1, y1, w, h = parts
                boxes.append([x1, y1, x1 + w, y1 + h])
            elif type == "pred":
                parts = list(map(float, lines[i].split()[:5]))
                x1, y1, w, h, score = parts
                boxes.append([x1, y1, x1 + w, y1 + h, score])
            else:
                raise ValueError("Invalid type. Use 'gt' or 'pred'.")

            i += 1
        data[img] = boxes
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

    for img, gt in gt_boxes.items():
        preds = pred_boxes.get(img, [])
        gt_flags = [False] * len(gt)
        total_gt += len(gt)

        preds_sorted = sorted(preds, key=lambda x: -x[4])  # sort by score

        for pred in preds_sorted:
            px1, py1, pw, ph, score = pred
            px2, py2 = px1 + pw, py1 + ph
            matched = False
            for i, gt_box in enumerate(gt):
                if (
                    not gt_flags[i]
                    and iou([px1, py1, px2, py2], gt_box) >= iou_threshold
                ):
                    gt_flags[i] = True
                    matched = True
                    break
            all_scores.append(score)
            all_tp.append(1 if matched else 0)
            all_fp.append(0 if matched else 1)

    return all_scores, all_tp, all_fp, total_gt


def compute_metrics(scores, tp, fp, total_gt):
    scores = np.array(scores)
    tp = np.array(tp)
    fp = np.array(fp)

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

    return precision, recall, ap
