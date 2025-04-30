import argparse
import os
import time
from collections import defaultdict

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from tqdm import tqdm

from tool import save_annotations
from visualize import visualize


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        default="detector.tflite",
        help="Path to model file",
    )
    parser.add_argument(
        "--input",
        type=str,
        default="WIDER_val/images",
        help="Relative path to image folder inside dataset",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="output/WIDER_val",
        help="Path to save annotated images and predictions",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(current_dir, args.model)
    input_dir = os.path.join("wider_face", args.input)
    output_dir = os.path.join(current_dir, args.output)
    os.makedirs(os.path.join(output_dir, "images"), exist_ok=True)

    base_options = python.BaseOptions(model_asset_path=model_path)
    options = vision.FaceDetectorOptions(base_options=base_options)
    detector = vision.FaceDetector.create_from_options(options)

    image_paths = []
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.endswith(".jpg"):
                image_paths.append(os.path.join(root, file))

    total_time = 0
    outputs = defaultdict(list)

    for image_path in tqdm(image_paths, desc="Processing images"):
        file = os.path.basename(image_path)
        image = mp.Image.create_from_file(image_path)

        start = time.time()
        detection_result = detector.detect(image)
        end = time.time()
        total_time += (end - start) * 1000

        image_copy = np.copy(image.numpy_view())
        annotated_image = visualize(image_copy, detection_result)
        rgb_annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
        save_path = os.path.join(output_dir, "images", file)
        cv2.imwrite(save_path, rgb_annotated_image)

        boxes = []
        for det in detection_result.detections:
            bbox = det.bounding_box
            x1, y1, w, h = bbox.origin_x, bbox.origin_y, bbox.width, bbox.height
            score = det.categories[0].score if det.categories else 0.0
            boxes.append((x1, y1, w, h, score))
        outputs[os.path.relpath(image_path, input_dir)] = boxes

    save_annotations(outputs, os.path.join(output_dir, "annotations.txt"))
    print(f"✅ Avg. Inference Time: {total_time / len(image_paths):.2f} ms")
