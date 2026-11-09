#!/usr/bin/env python
"""
File Name: pose_coco_eval_fast.py
Author: wux024
Email: wux024@nenu.edu.cn
Created On: 2024/7/28
Last Modified: 2024/10/25
Version: 1.0.

Overview:
    Provide a concise summary of the file's functionality, objectives, or primary logic implemented.

Notes:
    - Modifications should be documented in the "Revision History" section beneath this.
    - Ensure compliance with project coding standards.

Revision History:
    - [2024/7/28] wux024: Initial file creation
    - [2024/10/25] wux024: Added support for spipose models and specific path configurations
"""

import argparse
import numpy as np
import yaml
import os
from xtcocotools.coco import COCO
from xtcocotools.cocoeval import COCOeval

def parse_args():
    parser = argparse.ArgumentParser(description="Validate pose estimation models on a specified dataset.")
    parser.add_argument("--model-type", type=str, default="animalrtpose", help="Type of the model (e.g., animalrtpose, yolov8, yolo11, spipose).")
    parser.add_argument("--models", type=str, default="n", help="Comma-separated list of model codes to evaluate (e.g., n,s,m).")
    parser.add_argument("--dataset", type=str, default="ap10k", help="Name of the dataset.")
    parser.add_argument("--split", type=str, default="test", help="Split of the dataset to evaluate on (e.g., train, val, test).")
    return parser.parse_args()

def load_dataset_config(dataset_name):
    """Load the dataset configuration file."""
    with open(f"configs/data/{dataset_name}.yaml") as f:
        dataset_cfg = yaml.load(f, Loader=yaml.FullLoader)
        oks_sigmas = np.array(dataset_cfg.get("oks_sigmas", [1.0 / dataset_cfg["kpt_shape"][0]] * dataset_cfg["kpt_shape"][0]))
    return oks_sigmas

def load_coco_annotations(dataset_name, split):
    """Load the COCO annotations file."""
    return COCO(f"datasets/{dataset_name}/annotations/{split}.json")

def main():
    args = parse_args()
    oks_sigmas = load_dataset_config(args.dataset)
    coco_gt = load_coco_annotations(args.dataset, args.split)
    for root, _, files in os.walk(f"runs/{args.model_type}/eval/{args.dataset}"):
        for file in files:
            if file.endswith('.json'):
                print(f"Evaluating root: {root}")
                coco_dt = coco_gt.loadRes(os.path.join(root, file))
                # Evaluate the predictions
                coco_eval = COCOeval(coco_gt, coco_dt, iouType="keypoints", sigmas=oks_sigmas, use_area=True)
                coco_eval.evaluate()
                coco_eval.accumulate()
                coco_eval.summarize()

if __name__ == "__main__":
    main()