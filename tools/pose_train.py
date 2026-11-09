#!/usr/bin/env python
"""
File Name: pose_train_combined.py
Author: wux024
Email: wux024@nenu.edu.cn
Created On: 2024/7/3
Last Modified: 2024/10/24
Version: 1.0.

Overview:
    Combined script to train various pose estimation models on a specified dataset with default or user-provided settings.

Notes:
    - Modifications should be documented in the "Revision History" section beneath this.
    - Ensure compliance with project coding standards.

Revision History:
    - [2024/7/3] wux024: Initial file creation
    - [2024/10/24] wux024: Added support for spipose models
"""

import argparse
import os
import subprocess
from datetime import datetime

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

def get_pretrained_model(model_yaml):
    """Get the path to the pretrained model if it exists."""
    pretrained_model = f"{model_yaml[:-5]}.pt"
    pretrained_path = os.path.join("./weights", pretrained_model)
    if os.path.exists(pretrained_path):
        return pretrained_path
    else:
        print(f"Warning: Pretrained model {pretrained_model} not found. Skipping...")
        return None

def construct_train_command(args, model_yaml, pretrained_model):
    """Construct the yolo pose train command."""
    datacfg = f"./configs/data/{args.dataset}.yaml"
    modelcfg = f"./configs/models/{args.dataset}/{model_yaml}"
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
    output_dir = f"./runs/{args.model_type}/train/{args.dataset}"

    cmd = [
        "yolo",
        "pose",
        "train",
        f"data={datacfg}",
        f"model={modelcfg}",
        f"pretrained={pretrained_model}" if pretrained_model else "",
        f"epochs={args.epochs}",
        f"imgsz={args.imgsz}",
        f"batch={args.batch}",
        f"project={output_dir}",
        f"name={model_name}",
        f"device={args.device}",
        f"workers={args.workers}",
        f"seed={args.seed}",
        f"pose={args.pose}",
        f"patience={args.patience}",
        f"cos_lr=True",
        f"resume=True",
    ]

    return [arg for arg in cmd if arg]

def rename_dataset_directory(original_name, temp_name):
    """Rename the dataset directory."""
    if os.path.exists(original_name):
        os.rename(original_name, temp_name)
    else:
        print(f"Dataset directory {original_name} does not exist.")

def main():
    # Get the current date and time for the output directory
    current_date = datetime.now().strftime("%Y%m%d")
    seed_value = int(current_date)

    # Default training settings
    default_settings = {
        "dataset": "ap10k",
        "epochs": 10000,
        "patience": 2000,
        "batch": -1,
        "imgsz": 640,
        "device": None,
        "workers": 16,
        "pose": 40.0,
        "optical_field_sizes": None,
        "sub_optical_field_sizes": None,
        "window_size": None,
        "seed": seed_value,
        "imgsz_hadamard": None,
        "model_type": "animalrtpose",
        "models": None,
    }

    # Define the argument parser
    parser = argparse.ArgumentParser(
        description="Train pose estimation models on a specified dataset with default or user-provided settings."
    )

    # Required argument
    parser.add_argument("--dataset", type=str, default=default_settings["dataset"], help="Name of the dataset.")

    # Optional arguments
    parser.add_argument("--epochs", type=int, default=default_settings["epochs"], help="Number of training epochs.")
    parser.add_argument("--patience", type=int, default=default_settings["patience"], help="Early stopping patience.")
    parser.add_argument("--batch", type=int, default=default_settings["batch"], help="Batch size.")
    parser.add_argument("--imgsz", type=int, default=default_settings["imgsz"], help="Image size.")
    parser.add_argument("--device", type=str, default=default_settings["device"], help="Device to use (e.g., 0, 1, 2, cpu).")
    parser.add_argument("--model-type", type=str, default=default_settings["model_type"], help="Model type.")
    parser.add_argument("--models", type=str, help="Comma-separated list of model codes (n, s, m, l, x).")
    parser.add_argument("--no-pretrained", action="store_true", help="Not use a pretrained model.")
    parser.add_argument("--workers", type=int, default=default_settings["workers"], help="Number of workers.")
    parser.add_argument("--seed", type=int, default=default_settings["seed"], help="Random seed.")
    parser.add_argument("--pose", type=float, default=default_settings["pose"], help="Pose loss weight.")
    parser.add_argument("--optical-field-sizes", type=int, default=default_settings["optical_field_sizes"], help="Optical field size for the entire image.")
    parser.add_argument("--sub-optical-field-sizes", type=int, default=default_settings["sub_optical_field_sizes"], help="Optical field size for sub-regions of the image.")
    parser.add_argument("--window-size", nargs=2, type=int, default=default_settings["window_size"], help="Window size for sub-regions of the image.")
    parser.add_argument("--inverse", action="store_true", help="Order the images by their size before splitting into sub-regions.")
    parser.add_argument("--imgsz-hadamard", type=int, default=None, help="Image size for the Hadamard transform. If not provided, it will be set to imgsz.")
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
            hadamard_seed=args.hadamard_seed,
        )
    if original_dataset_dir == temp_dataset_dir:
        original_dataset_dir = f"./datasets/{args.dataset}/images_"
    
    rename_dataset_directory(original_dataset_dir, temp_dataset_dir)

    try:
        # Loop through each model for the given dataset
        for model_yaml in models:
            pretrained_model = get_pretrained_model(model_yaml) if not args.no_pretrained else None

            # Construct the yolo pose train command
            train_cmd = construct_train_command(args, model_yaml, pretrained_model)

            # Run the command
            print(f"Running command: {' '.join(train_cmd)}")
            subprocess.run(train_cmd, check=True)
    finally:
        rename_dataset_directory(temp_dataset_dir, original_dataset_dir)

if __name__ == "__main__":
    main()