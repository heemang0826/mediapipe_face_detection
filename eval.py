import argparse
import os

from tool import compute_metrics, evaluate, parse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ground_truth",
        type=str,
        default="wider_face_val_bbx_gt.txt",
        help="Path to model file",
    )
    parser.add_argument(
        "--prediction",
        type=str,
        default="WIDER_val/annotations.txt",
        help="Relative path to image folder inside dataset",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(os.path.dirname(current_dir), "wider_face")
    gt_dir = os.path.join(data_dir, "wider_face_split", args.ground_truth)
    pred_dir = os.path.join(current_dir, "outputs", args.prediction)

    gt = parse(gt_dir, type="gt")
    pred = parse(pred_dir, type="pred")

    scores, tp, fp, total = evaluate(gt, pred, iou_threshold=0.5)
    precision, recall, ap = compute_metrics(scores, tp, fp, total)

    print(f"AP: {ap:.4f}")
    print(f"Final Precision: {precision[-1]:.4f}")
    print(f"Final Recall: {recall[-1]:.4f}")
