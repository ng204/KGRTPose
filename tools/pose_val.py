#!/usr/bin/env python
"""
File Name: pose_val_combined.py
Author: wux024
Email: wux024@nenu.edu.cn
Created On: 2024/7/3
Last Modified: 2024/10/25
Version: 1.0.

Overview:
    Combined script to evaluate various pose estimation models on a specified dataset with default or user-provided settings.

Notes:
    - Modifications should be documented in the "Revision History" section beneath this.
    - Ensure compliance with project coding standards.

Revision History:
    - [2024/7/3] wux024: Initial file creation
    - [2024/10/25] wux024: Added support for spipose models and data path renaming
"""

import argparse
import subprocess
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

def build_output_dir(
    base_dir, 
    optical_field_sizes=None, 
    sub_optical_field_sizes=None, 
    window_size=None, 
    seed=None, 
    hadamard_seed=None, 
    inverse=False, 
    imgsz_hadamard=None,
    aliasing=False
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
    
    if hadamard_seed is not None:
        base_dir += f"-{hadamard_seed}"
    
    if seed is not None:
        base_dir += f"-{seed}"
    
    return base_dir

def construct_val_command(args, model_yaml):
    """Construct the yolo pose val command."""
    datacfg = f"./configs/data/{args.dataset}.yaml"
    model_name = build_output_dir(
        base_dir=model_yaml[:-5],
        optical_field_sizes=args.optical_field_sizes,
        sub_optical_field_sizes=args.sub_optical_field_sizes,
        window_size=args.window_size,
        seed=args.seed,
        hadamard_seed=args.hadamard_seed,
        inverse=args.inverse,
        imgsz_hadamard=args.imgsz_hadamard,
        aliasing=args.aliasing
    )
    model_dir = f"./runs/{args.model_type}/train/{args.dataset}"
    output_dir = f"./runs/{args.model_type}/eval/{args.dataset}"
    model = os.path.join(model_dir, model_name, "weights/best.pt")

    # Construct the yolo pose val command
    cmd = [
        "yolo",
        "pose",
        "val",
        f"data={datacfg}",
        f"model={model}",
        f"imgsz={args.imgsz}",
        f"batch={args.batch}",
        f"project={output_dir}",
        f"name={model_name}",
        f"device={args.device}",
        f"conf={args.conf}",
        f"iou={args.iou}",
        f"max_det={args.max_det}",
        f"half={args.half}",
        f"save_json={args.save_json}",
        f"save_hybrid={args.save_hybrid}",
        f"dnn={args.dnn}",
        f"plots={args.plots}",
        f"rect={args.rect}",
        f"split={args.split}",
    ]
    
    return [arg for arg in cmd if arg]  # Filter out any empty strings

def rename_dataset_directory(original_name, temp_name):
    """Rename the dataset directory."""
    if os.path.exists(original_name):
        os.rename(original_name, temp_name)
    else:
        print(f"Dataset directory {original_name} does not exist.")

def main():
    # Default evaluation settings
    default_settings = {
        "dataset": "mouse",
        "imgsz": 640,
        "batch": 16,
        "conf": 0.001,
        "iou": 0.6,
        "max_det": 300,
        "half": False,
        "device": None,
        "workers": 16,
        "seed": None,
        "models": None,
        "model_type": "animalrtpose",
    }

    parser = argparse.ArgumentParser(description="Evaluate pose estimation models on a specified dataset.")

    # Dataset selection
    parser.add_argument("--dataset", type=str, default=default_settings["dataset"], help="Dataset to evaluate on.")

    # Image size selection
    parser.add_argument("--imgsz", type=int, default=default_settings["imgsz"], help="Image size to use for evaluation.")

    # Batch size selection
    parser.add_argument("--batch", type=int, default=default_settings["batch"], help="Batch size for evaluation.")

    # Confidence threshold selection
    parser.add_argument("--conf", type=float, default=default_settings["conf"], help="Confidence threshold for detections.")

    # IoU threshold selection
    parser.add_argument("--iou", type=float, default=default_settings["iou"], help="IoU threshold for NMS.")

    # Maximum number of detections per image
    parser.add_argument("--max-det", type=int, default=default_settings["max_det"], help="Maximum number of detections per image.")

    # Half precision selection
    parser.add_argument("--half", action="store_true", help="Use half precision for inference.")

    # Device selection
    parser.add_argument("--device", type=str, default=default_settings["device"], help="Device to use for inference.")

    # Workers selection
    parser.add_argument("--workers", type=int, default=default_settings["workers"], help="Number of workers for data loading.")

    # Seed selection
    parser.add_argument("--seed", type=int, default=default_settings["seed"], help="Random seed for evaluation.")

    # Save JSON selection
    parser.add_argument("--save-json", action="store_true", help="Save JSON detections.")

    # Save hybrid selection
    parser.add_argument("--save-hybrid", action="store_true", help="Save hybrid detections.")

    # DNN selection
    parser.add_argument("--dnn", action="store_true", help="Use OpenCV DNN for inference.")

    # Plots selection
    parser.add_argument("--plots", action="store_true", help="Generate plots during evaluation.")

    # Rect selection
    parser.add_argument("--rect", action="store_true", help="Rectangular inference.")

    # Split selection
    parser.add_argument("--split", type=str, default="val", help="Split to evaluate on (train, val, test).")

    # Models selection
    parser.add_argument("--models", type=str, default=default_settings["models"], help="Comma-separated list (n, s, m, l, x) of model codes to evaluate.")

    # Model type selection
    parser.add_argument("--model-type", type=str, default="animalrtpose", help="Model type to evaluate. Supported types are animalrtpose, yolov8, yolo11, spipose.")

    # Optical field sizes selection
    parser.add_argument("--optical-field-sizes", type=int, default=None, help="Optical field size for the entire image.")

    # Sub-optical field sizes selection
    parser.add_argument("--sub-optical-field-sizes", type=int, default=None, help="Optical field size for sub-regions of the image.")

    # Window size selection
    parser.add_argument("--window-size", nargs=2, type=int, default=None, help="Window size for sub-regions of the image.")

    # Inverse order selection
    parser.add_argument("--inverse", action="store_true", help="Order the images by their size before splitting into sub-regions.")

    # Image size for Hadamard transform selection
    parser.add_argument("--imgsz-hadamard", type=int, default=None, help="Image size for the Hadamard transform. If not provided, it will be set to imgsz.")

    # Aliasing selection
    parser.add_argument("--aliasing", action="store_true", help="Use aliasing for the Hadamard transform.")

    parser.add_argument("--hadamard-seed", type=int, default=None, help="Random seed for the Hadamard transform.")

    args = parser.parse_args()

    # Process selected models
    if args.models is None:
        models = parse_models("n,s,m,l,x", model_type=args.model_type)
    else:
        models = parse_models(args.models, model_type=args.model_type)

    # Rename the dataset directory before training
    temp_dataset_dir = f"./datasets/{args.dataset}/images"
    original_dataset_dir = build_output_dir(
            base_dir=temp_dataset_dir,
            optical_field_sizes=args.optical_field_sizes,
            sub_optical_field_sizes=args.sub_optical_field_sizes,
            window_size=args.window_size,
            inverse=args.inverse,
            imgsz_hadamard=args.imgsz_hadamard,
            aliasing=args.aliasing,
            hadamard_seed=args.hadamard_seed
        )
    if original_dataset_dir == temp_dataset_dir:
        original_dataset_dir = f"./datasets/{args.dataset}/images_"
    
    rename_dataset_directory(original_dataset_dir, temp_dataset_dir)

    try:
        # Loop through each model for the given dataset
        for model_yaml in models:
            # Construct the yolo pose val command
            val_cmd = construct_val_command(args, model_yaml)
            # Run the command
            subprocess.run(val_cmd, check=True)
    finally:
        rename_dataset_directory(temp_dataset_dir, original_dataset_dir)

if __name__ == "__main__":
    main()