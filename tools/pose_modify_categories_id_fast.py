#!/usr/bin/env python
"""
File Name: pose_modify_categories_id_fast.py
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

def modify_categories_id(json_file):
    """Modify category_id in the given JSON file."""
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    cat_id_set = set([ann["category_id"] for ann in data])
    
    if 0 in cat_id_set:
        for ann in data:
            ann["category_id"] += 1
    
    with open(json_file, "w") as f:
        json.dump(data, f, indent=4)

def process_directory(directory):
    """Process all JSON files in the given directory and its subdirectories."""
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.json'):
                json_file = os.path.join(root, file)
                modify_categories_id(json_file)
                print(f"Processed: {json_file}")

def parse_args():
    parser = argparse.ArgumentParser(description="Quickly modify categories id in COCO json files.")
    parser.add_argument("--model-type", type=str, required=True, help="Type of the model.")
    parser.add_argument("--dataset", type=str, required=True, help="Name of the dataset.")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    model_type = args.model_type
    dataset = args.dataset
    base_dir = os.path.join("runs", model_type, "eval", dataset)
    
    if not os.path.exists(base_dir):
        print(f"Directory does not exist: {base_dir}")
    else:
        process_directory(base_dir)
