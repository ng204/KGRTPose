#!/usr/bin/env python
"""
File Name: pose_modify_categories_id.py
Author: wux024
Email: wux024@nenu.edu.cn
Created On: 2024/7/28
Last Modified: 2024/7/28
Version: 1.0.

Overview:
    Provide a concise summary of the file's functionality, objectives, or primary logic implemented.

Notes:
    - Modifications should be documented in the "Revision History" section beneath this.
    - Ensure compliance with project coding standards.

Revision History:
    - [2024/7/28] wux024: Initial file creation
"""

import argparse
import json
import os

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
    parser = argparse.ArgumentParser(description="Modify categories id in COCO json file.")
    parser.add_argument("--dataset", type=str, default="ap10k", help="Name of the dataset.")
    parser.add_argument("--model-type", type=str, default="animalrtpose", help="Type of the model.")
    parser.add_argument("--models", type=str, default="n,s,m,l,x", help="Comma-separated list of model codes.")
    parser.add_argument("--optical-field-sizes", type=int, default=None, help="Optical field size for the entire image (for spipose).")
    parser.add_argument("--sub-optical-field-sizes", type=int, default=None, help="Optical field size for sub-regions of the image (for spipose).")
    parser.add_argument("--window-size", nargs=2, type=int, default=None, help="Window size for sub-regions of the image (for spipose).")
    parser.add_argument("--inverse", action="store_true", help="Order the images by their size before splitting into sub-regions (for spipose).")
    parser.add_argument("--imgsz-hadamard", type=int, default=None, help="Image size for the Hadamard transform (for spipose).")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    return parser.parse_args()

def build_output_dir(
    base_dir, 
    optical_field_sizes=None, 
    sub_optical_field_sizes=None, 
    window_size=None, 
    inverse=False, 
    imgsz_hadamard=None,
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
    
    if imgsz_hadamard is not None:
        base_dir += f"-{imgsz_hadamard}"
    
    if seed is not None:
        base_dir += f"-{seed}"
    

    
    return base_dir

def modify_categories_id(json_file):
    with open(json_file) as f:
        data = json.load(f)
    
    cat_id_set = set([ann["category_id"] for ann in data])
    
    if 0 in cat_id_set:
        for ann in data:
            ann["category_id"] += 1
    
    with open(json_file, "w") as f:
        json.dump(data, f, indent=4)

if __name__ == "__main__":
    args = parse_args()
    dataset = args.dataset
    base_dir = os.path.join("runs", args.model_type, "eval", dataset)

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
                args.seed
            )
        else:
            model_name = model[:-5]

        model_dir = os.path.join(base_dir, model_name)
        
        for dirpath, _, files in os.walk(model_dir):
            for file in files:
                if file.endswith(".json"):
                    json_file = os.path.join(dirpath, file)
                    modify_categories_id(json_file)
