#!/usr/bin/env python
"""
File Name: pose_video_demo.py
Author: wux024
Email: wux024@nenu.edu.cn
Created On: 2024/11/19
Last Modified: 2024/11/19
Version: 1.0.

Overview:
    Provide a concise summary of the file's functionality, objectives, or primary logic implemented.

Notes:
    - Modifications should be documented in the "Revision History" section beneath this.
    - Ensure compliance with project coding standards.

Revision History:
    - [2024/11/19] wux024: Initial file creation
"""

import argparse
from ultralytics import YOLO
import yaml
from ultralytics.utils.pose_cfg import SetSkeleton
import cv2
import numpy as np
import os

def plot_results(result, im_black, args):
    """Save the results on different backgrounds."""
    # Update original image size
    result.orig_shape = im_black.shape[:2]

    # Rescale keypoints
    keypoints_rescaled = result.keypoints.xyn.clone()
    keypoints_rescaled[:, :, 0] *= im_black.shape[1]
    keypoints_rescaled[:, :, 1] *= im_black.shape[0]
    keypoints_data_cloned = result.keypoints.data.clone()
    keypoints_data_cloned[:, :, :2] = keypoints_rescaled
    result.keypoints.data = keypoints_data_cloned

    im = result.plot(
        img=im_black,
        conf=args.show_conf,
        line_width=args.line_width,
        kpt_radius=args.kpt_radius,
        kpt_line=args.kpt_line,
        labels=args.show_labels,
        boxes=args.show_boxes,
        )
    
    return im

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=str, default="path/to/video.mp4", help="dataset to use")
    parser.add_argument("--dataset", type=str, default=None, help="path to dataset")
    parser.add_argument("--model", type=str, default="path/to/best.pt", help="model to use")
    parser.add_argument("--conf", type=float, default=0.25, help="object confidence threshold")
    parser.add_argument("--iou", type=float, default=0.7, help="IOU threshold for NMS")
    parser.add_argument("--imgsz", type=int, default=640, help="image size")
    parser.add_argument("--half", action="store_true", help="use FP16 half-precision inference")
    parser.add_argument("--device", default="", help="cuda device, i.e. 0 or 0,1,2,3 or cpu")
    parser.add_argument("--max-det", type=int, default=300, help="maximum detections per image")
    parser.add_argument("--vid-stride", type=int, default=1, help="video stride")
    parser.add_argument("--stream-buffer", action="store_true", help="stream buffer")
    parser.add_argument("--visualize", action="store_true", help="visualize detections")
    parser.add_argument("--augment", action="store_true", help="augmented inference")
    parser.add_argument("--agnostic-nms", action="store_true", help="class-agnostic NMS")
    parser.add_argument("--classes", type=list, default=None, help="filter by class")
    parser.add_argument("--retina-masks", action="store_true", help="use retina mask head")
    parser.add_argument("--embed", action="store_true", help="use embedding head")
    parser.add_argument("--show-labels", action="store_true", help="show detection class labels")
    parser.add_argument("--show-conf", action="store_true", help="show confidence score")
    parser.add_argument("--show-boxes", action="store_true", help="show boxes")
    parser.add_argument("--line-width", type=int, default=None, help="line width for boxes")
    parser.add_argument("--kpt-line", action="store_true", help="draw keypoint lines")
    parser.add_argument("--kpt-radius", type=int, default=5, help="keypoint radius")
    parser.add_argument("--project", default="runs/video_results", help="save results to project/name")
    args = parser.parse_args()

    model = YOLO(args.model)

    if args.dataset is not None:
        data = f"configs/data/{args.dataset}.yaml"
    
    data_cdg = yaml.load(open(f"configs/data/{args.dataset}.yaml"), Loader=yaml.FullLoader)
    
    if "skeleton" in data_cdg.keys():
        SetSkeleton(data_cdg["skeleton"])

    
    results = model(
        args.source,
        conf=args.conf,
        iou=args.iou,
        imgsz=args.imgsz,
        half=args.half,
        device=args.device,
        max_det=args.max_det,
        vid_stride=args.vid_stride,
        stream_buffer=args.stream_buffer,
        visualize=args.visualize,
        augment=args.augment,
        agnostic_nms=args.agnostic_nms,
        classes=args.classes,
        retina_masks=args.retina_masks,
        embed=args.embed,
        stream=True
        )
    
    cap = cv2.VideoCapture(args.source)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_name = os.path.splitext(os.path.basename(args.source))[0]
    out = cv2.VideoWriter(f"{args.project}/{video_name}.mp4", fourcc, fps, (width, height))

    im_previous = np.zeros((height, width, 3), np.uint8)
    for result in results:
        im_black = np.zeros((height, width, 3), np.uint8)
        im_current = plot_results(result, im_black, args)
        if np.all(im_current == 0):
            im_current = im_previous
        out.write(im_current)
        im_previous = im_current
    
    out.release()
    cap.release()