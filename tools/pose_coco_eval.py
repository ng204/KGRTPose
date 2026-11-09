#!/usr/bin/env python
"""
File Name: pose_coco_eval.py
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
from xtcocotools.coco import COCO
from xtcocotools.cocoeval import COCOeval

def parse_models(models_str, model_type="animalrtpose"):
    """Parse the comma-separated list of model codes into a list of model YAML files."""
    
    # Define mappings from model codes to YAML file names for each model type
    model_mappings = {
        "animalrtpose": {
            "n": "animalrtpose-n.yaml",
            "s": "animalrtpose-s.yaml",
            "m": "animalrtpose-m.yaml",
            "l": "animalrtpose-l.yaml",
            "x": "animalrtpose-x.yaml"
        },
        "yolov8-pose": {
            "n": "yolov8n-pose.yaml",
            "s": "yolov8s-pose.yaml",
            "m": "yolov8m-pose.yaml",
            "l": "yolov8l-pose.yaml",
            "x": "yolov8x-pose.yaml"
        },
        "yolo11-pose": {
            "n": "yolo11n-pose.yaml",
            "s": "yolo11s-pose.yaml",
            "m": "yolo11m-pose.yaml",
            "l": "yolo11l-pose.yaml",
            "x": "yolo11x-pose.yaml"
        },
        "spipose": {
            "n": "spipose-n.yaml",
            "s": "spipose-s.yaml",
            "m": "spipose-m.yaml",
            "l": "spipose-l.yaml",
            "x": "spipose-x.yaml"
        }
    }
    
    # Check if the provided model type is valid
    if model_type not in model_mappings:
        raise ValueError(f"Invalid model type: {model_type}. Valid types are {', '.join(model_mappings.keys())}.")
    
    # Get the mapping for the specific model type
    mapping = model_mappings[model_type]
    
    # Split the input string into individual model codes and map them to YAML file names
    models = [mapping[code.strip()] for code in models_str.split(",") if code.strip() in mapping]
    
    return models

def parse_args():
    parser = argparse.ArgumentParser(description="Validate pose estimation models on a specified dataset.")
    parser.add_argument("--model-type", type=str, default="animalrtpose", help="Type of the model (e.g., animalrtpose, yolov8, yolo11, spipose).")
    parser.add_argument("--models", type=str, default="n", help="Comma-separated list of model codes to evaluate (e.g., n,s,m).")
    parser.add_argument("--dataset", type=str, default="ap10k", help="Name of the dataset.")
    parser.add_argument("--split", type=str, default="test", help="Split of the dataset to evaluate on (e.g., train, val, test).")
    parser.add_argument("--optical-field-sizes", type=int, default=None, help="Optical field size for the entire image (for spipose).")
    parser.add_argument("--sub-optical-field-sizes", type=int, default=None, help="Optical field size for sub-regions of the image (for spipose).")
    parser.add_argument("--window-size", nargs=2, type=int, default=None, help="Window size for sub-regions of the image (for spipose).")
    parser.add_argument("--inverse", action="store_true", help="Order the images by their size before splitting into sub-regions (for spipose).")
    parser.add_argument("--imgsz-hadamard", type=int, default=None, help="Image size for the Hadamard transform (for spipose).")
    parser.add_argument("--aliasing", action="store_true", help="Use aliasing for the Hadamard transform.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
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

def build_output_dir(
    base_dir, 
    optical_field_sizes=None, 
    sub_optical_field_sizes=None, 
    window_size=None, 
    inverse=False, 
    imgsz_hadamard=None,
    aliasing=False,
    seed=42
):
    """Build the save directory based on the provided arguments."""
    base_dir = f"{base_dir}"
    
    if optical_field_sizes is not None:
        base_dir += f"-{optical_field_sizes}x{optical_field_sizes}"
    
    if sub_optical_field_sizes is not None:
        base_dir += f"-{sub_optical_field_sizes}x{sub_optical_field_sizes}"
    
    if window_size is not None:
        base_dir += f"-{window_size[0]}x{window_size[1]}"
    
    if inverse:
        base_dir += "-inverse"

    if aliasing:
        base_dir += "-aliasing"
    
    if imgsz_hadamard is not None:
        base_dir += f"-{imgsz_hadamard}"
    
    if seed is not None:
        base_dir += f"-{seed}"
    
    return base_dir

def main():
    args = parse_args()
    oks_sigmas = load_dataset_config(args.dataset)
    models = parse_models(args.models, model_type=args.model_type)
    
    for model in models:
        if args.model_type == "spipose":
            model_name = build_output_dir(
                model[:-5],
                args.optical_field_sizes,
                args.sub_optical_field_sizes,
                args.window_size,
                args.inverse,
                args.imgsz_hadamard,
                args.aliasing,
                args.seed
            )
        else:
            model_name = model[:-5]

        coco_gt = load_coco_annotations(args.dataset, args.split)
        coco_dt = coco_gt.loadRes(f"runs/{args.model_type}/eval/{args.dataset}/{model_name}/predictions.json")
        
        # Evaluate the predictions
        coco_eval = COCOeval(coco_gt, coco_dt, iouType="keypoints", sigmas=oks_sigmas, use_area=True)
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

if __name__ == "__main__":
    main()