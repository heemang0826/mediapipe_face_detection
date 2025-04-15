import argparse
import os

from tool import *


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
    output_dir = os.path.join(current_dir, "outputs")
    gt_dir = os.path.join(data_dir, "wider_face_split", args.ground_truth)
    pred_dir = os.path.join(output_dir, args.prediction)

    gt = parse(gt_dir, type="gt")
    pred = parse(pred_dir, type="pred")

    precision, recall, ap = evaluate(gt, pred, iou_threshold=0.5)
    save_pr_curve(precision, recall, ap, output_dir)
