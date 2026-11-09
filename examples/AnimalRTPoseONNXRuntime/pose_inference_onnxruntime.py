# Ultralytics YOLO 🚀, AGPL-3.0 license

import argparse
import os

import cv2
import numpy as np
import onnxruntime as ort
import torch

from ultralytics.utils import ASSETS, yaml_load
from ultralytics.utils.checks import check_requirements, check_yaml

from datetime import datetime


def measure_time(func, *args, **kwargs):
    start_time = datetime.now()
    result = func(*args, **kwargs)
    end_time = datetime.now()
    total_time = (end_time - start_time).total_seconds()
    return result, total_time


class AnimalRTPose:
    """
    Animal pose inference example using ONNXRuntime
    """

    def __init__(self, model_path,
                 source,
                 conf,
                 iou,
                 data,
                 draw_detections_action=True,
                 draw_keypoints_action=True,
                 draw_skeleton_action=True,
                 radius=5,
                 line_width=1):
        """
        Initializes an instance of the AnimalRTPose class.

        Args:
            model: Path to the ONNX model or TensorRT model.
            conf: Confidence threshold for filtering detections.
            iou: IoU (Intersection over Union) threshold for non-maximum suppression.
            data: Path to the input data config.
            draw_detections_action: Action to draw detections.
            draw_keypoints_action: Action to draw keypoints.
            draw_skeleton_action: Action to draw skeleton.
        """
        self.model_path = model_path
        self.model = None
        self.source = source
        self.conf = conf
        self.iou= iou

        self.data = data
        self.draw_detections_action = draw_detections_action
        self.draw_keypoints_action = draw_keypoints_action
        self.draw_skeleton_action = draw_skeleton_action

        self.radius = radius
        self.line_width = line_width


        # Load the class names from the COCO dataset
        self.classes = yaml_load(check_yaml(data))["names"]
        self.keypoints = yaml_load(check_yaml(data))["kpt_shape"][0]
        self.skeleton = yaml_load(check_yaml(data))["skeleton"]

        # Generate a color palette for the classes
        self.classes_color_palette = np.random.uniform(0, 255, size=(len(self.classes), 3))
        self.keypoints_color_palette = np.random.uniform(0, 255, size=(self.keypoints, 3))
        self.skeleton_color_palette = np.random.uniform(0, 255, size=(len(self.skeleton), 3))


    def preprocess(self, input_image):
        """
        Preprocesses the input image before performing inference.

        Returns:
            image_data: Preprocessed image data ready for inference.
        """
        # Read the input image using OpenCV
        img = input_image.copy()

        # Get the height and width of the input image
        img_height, img_width = img.shape[:2]

        scale = min(self.input_width / img_width, self.input_height / img_height)

        ox = self.input_width - scale * img_width
        oy = self.input_height - scale * img_height

        M = np.array([
            [scale, 0, ox],
            [0, scale, oy],
            ], dtype="float32"
        )

        img = cv2.warpAffine(img, M,
                             (self.input_width, self.input_height),
                             flags=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_CONSTANT,
                             borderValue=(114, 114, 114))

        IM = cv2.invertAffineTransform(M)

        # Convert the image color space from BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Normalize the image data by dividing it by 255.0
        img = np.array(img) / 255.0

        # Transpose the image to have the channel dimension as the first dimension
        img = np.transpose(img, (2, 0, 1))  # Channel first

        # Expand the dimensions of the image data to match the expected input shape
        img = np.expand_dims(img, axis=0).astype(np.float32)

        # Return the preprocessed image data
        return img, IM

    def postprocess(self, pred, IM=[]):
        """
        Performs post-processing on the model's output to extract bounding boxes, scores, and class IDs.

        Args:
            pred (numpy.ndarray): The output of the model.
            1, 8400, boxes + classes + keypoints * 3 [cx, cy, w, h, conf1, ..., confn, nk*3]
            for example: for ap10k, it has 50 classes and 17 keypoints, so
            pred's shape is [1,8400, 4+50+17*3].
            IM (list): The list of input images to draw detections on.

        Returns:
            numpy.ndarray: The input image with detections drawn on it.
        """
        # Transpose and squeeze the output to match the expected shape
        outputs = np.transpose(np.squeeze(pred[0]))

        # Get the number of rows in the outputs array
        rows = outputs.shape[0]

        # Lists to store the bounding boxes, scores, and class IDs of the detections
        boxes = []
        keypoints_list = []
        scores = []
        class_ids = []

        # Iterate over each row in the outputs array
        for i in range(rows):
            # Extract the class scores from the current row
            classes_scores = outputs[i][4:4+len(self.classes)]

            # Find the maximum score among the class scores
            max_score = np.amax(classes_scores)

            # If the maximum score is above the confidence threshold
            if max_score >= self.conf:
                # Get the class ID with the highest score
                class_id = np.argmax(classes_scores)

                # Extract the bounding box coordinates from the current row
                x, y, w, h = outputs[i][0], outputs[i][1], outputs[i][2], outputs[i][3]

                # Calculate the scaled coordinates of the bounding box
                left = (x - w * 0.5) * IM[0][0] + IM[0][2]
                top = (y - h * 0.5) * IM[1][1] + IM[1][2]
                width = w * IM[0][0]
                height = h * IM[1][1]

                keypoints = outputs[i][4+len(self.classes):].reshape(-1, 3)
                keypoints[:, 0] = keypoints[:, 0] * IM[0][0] + IM[0][2]
                keypoints[:, 1] = keypoints[:, 1] * IM[1][1] + IM[1][2]

                # Add the class ID, score, and box coordinates to the respective lists
                class_ids.append(class_id)
                scores.append(max_score)
                boxes.append([left, top, width, height])
                keypoints_list.append(keypoints)

        # Apply non-maximum suppression to filter out overlapping bounding boxes
        indices = cv2.dnn.NMSBoxes(boxes, scores, self.conf, self.iou)

        boxes = np.array(boxes)
        keypoints_list = np.array(keypoints_list)
        scores = np.array(scores)
        class_ids = np.array(class_ids)

        if indices is not None:
            indices = indices[0] if len(indices) == 1 else indices
            boxes = boxes[indices]
            keypoints_list = keypoints_list[indices]
            class_ids = class_ids[indices]
            scores = scores[indices]
            results = {
                'boxes': boxes,
                'keypoints_list': keypoints_list,
                'class_ids': class_ids,
                'scores': scores,
            }
        else:
            results = {
                'boxes': None,
                'keypoints_list': None,
                'class_ids': None,
                'scores': None,
            }
        # Return the modified input image
        return results

    def draw_detections(self, img, box, score, class_id):
        """
        Draws bounding boxes and labels on the input image based on the detected objects.

        Args:
            img: The input image to draw detections on.
            box: Detected bounding box.
            score: Corresponding detection score.
            class_id: Class ID for the detected object.

        Returns:
            None
        """
        # Extract the coordinates of the bounding box
        x1, y1, w, h = box

        # Retrieve the color for the class ID
        color = self.classes_color_palette[class_id]

        # Draw the bounding box on the image
        cv2.rectangle(img, (int(x1), int(y1)), (int(x1 + w), int(y1 + h)), color, 2)

        # Create the label text with class name and score
        label = f"{self.classes[class_id]}: {score:.2f}"

        # Calculate the dimensions of the label text
        (label_width, label_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)

        # Calculate the position of the label text
        label_x = int(x1)
        label_y = int(y1 - 10) if y1 - 10 > label_height else int(y1 + 10)

        # Draw a filled rectangle as the background for the label text
        cv2.rectangle(
            img,
            (label_x, label_y - label_height),
            (label_x + label_width, label_y + label_height), color, cv2.FILLED
        )

        # Draw the label text on the image
        cv2.putText(img, label,
                    (label_x, label_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 0, 0),
                    1, cv2.LINE_AA)

    def draw_keypoints(self, img, keypoints, radius=5):
        for i, keypoint in enumerate(keypoints):
            x, y = keypoint[0], keypoint[1]
            color_k = [int(x) for x in self.keypoints_color_palette[i]]

            if x!=0 and y!=0:
                cv2.circle(img,
                           (int(x), int(y)),
                           radius=radius,
                           color=color_k,
                           thickness=-1,
                           lineType=cv2.LINE_AA)


    def draw_skeleton(self, img, keypoints, line_width=1):
        for i, sk in enumerate(self.skeleton):
            pos1 = (int(keypoints[(sk[0]), 0]), int(keypoints[(sk[0]), 1]))
            pos2 = (int(keypoints[(sk[1]), 0]), int(keypoints[(sk[1]), 1]))

            if pos1[0] == 0 or pos1[1] == 0 or pos2[0] == 0 or pos2[1] == 0:
                continue

            cv2.line(img,
                     pos1, pos2,
                     color=self.skeleton_color_palette[i],
                     thickness=line_width,
                     lineType=cv2.LINE_AA
                     )


    def inference(self, frame):
        """
        Performs inference using an ONNX model and returns the output image with drawn detections.

        Returns:
            output_img: The output image with drawn detections.
        """

        results = {}

        # Preprocess the image data
        [img, IM], preprocess_time = measure_time(self.preprocess, frame)

        # Run inference using the preprocessed image data
        pred, inference_time = measure_time(self.model.run,None, {self.model.get_inputs()[0].name: img})

        # Perform post-processing on the outputs to obtain output image.
        results, postprecess_time = measure_time(self.postprocess, pred, IM)

        results['preprocess_time'] = preprocess_time
        results['inference_time'] = inference_time
        results['postprecess_time'] = postprecess_time
        results['fps'] = 1.0 / (inference_time + postprecess_time + preprocess_time + 1e-7)

        return results


    def process_frame(self, frame):

        if not isinstance(frame, np.ndarray):
            frame = cv2.imread(frame)

        results = self.inference(frame)


        boxes = results['boxes']
        keypoints_list = results['keypoints_list']
        scores = results['scores']
        class_ids = results['class_ids']

        if len(boxes) > 0:
            if boxes.ndim == 1:
                if self.draw_detections_action:
                    self.draw_detections(frame, boxes, scores, class_ids)
                if self.draw_keypoints_action:
                    self.draw_keypoints(frame, keypoints_list, radius=self.radius)
                if self.draw_skeleton_action:
                    self.draw_skeleton(frame, keypoints_list, line_width=self.line_width)
            else:
                for i, box in enumerate(boxes):
                    if self.draw_detections_action:
                        self.draw_detections(frame, box, scores[i], class_ids[i])
                    if self.draw_keypoints_action:
                        self.draw_keypoints(frame, keypoints_list[i], radius=self.radius)
                    if self.draw_skeleton_action:
                        self.draw_skeleton(frame, keypoints_list[i], line_width=self.line_width)


        times = {
            'preprocess_time': 'preprocess: {:.3f} ms'.format(results['preprocess_time'] * 1000.0),
            'inference_time': 'inference: {:.3f} ms'.format(results['inference_time'] * 1000.0),
            'postprocess_time': 'postprocess: {:.3f} ms'.format(results['postprecess_time'] * 1000.0),
            'fps': 'FPS: {:.1f} FPS'.format(results['fps'])
        }

        cv2.putText(frame, times['fps'], (5, 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 255, 0), 1, cv2.LINE_AA)
        cv2.putText(frame, times['preprocess_time'], (5, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 255, 0), 1, cv2.LINE_AA)
        cv2.putText(frame, times['inference_time'], (5, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 255, 0), 1, cv2.LINE_AA)
        cv2.putText(frame, times['postprocess_time'], (5, 80),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 255, 0), 1, cv2.LINE_AA)

        return frame, results


    def process_video(self, path):
        cap = cv2.VideoCapture(path)
        while cap.isOpened():
            ret, frame = cap.read()
            if ret:
                frame, _ = self.process_frame(frame)
            cv2.imshow('AnimalRTPose', frame)
            if cv2.waitKey(30) & 0xFF == ord('q'):
                break
        cv2.destroyAllWindows()
        cap.release()

    def process_dir(self, path):
        count = 0
        inference_time = 0.0
        for root, _, files in os.walk(path):
            for file in files:
                file_path = os.path.join(root, file)
                if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                    frame, results = self.process_frame(file_path)
                    count += 1
                    print('Inference time on {} frame: {:.3f} ms'.format(count,
                                                                         results['inference_time'] * 1000.0))
                    inference_time += results['inference_time']
                    cv2.imshow('AnimalRTPose', frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                else:
                    print('Only processing images in path. Skipping {}'.format(file_path))
        cv2.destroyAllWindows()
        print('Average inference time with {} frames: {:.3f} ms'.format(count,
                                                                        inference_time * 1000.0 / count))



    def process_camera(self):
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print('Cannot open camera')

        while True:
            ret, frame = cap.read()
            if not ret:
                print('Not able to read camera')
                break
            frame, time = self.process_frame(frame)

            cv2.imshow('AnimalRTPose', frame)
            if cv2.waitKey(30) & 0xFF == ord('q'):
                break
        cv2.destroyAllWindows()
        cap.release()


    def main(self):

        # Create an inference session using the ONNX model and specify execution providers
        self.model = ort.InferenceSession(self.model_path,
                                       providers=["CUDAExecutionProvider", "CPUExecutionProvider"])

        # Get the model inputs
        model_inputs = self.model.get_inputs()

        # Store the shape of the input for later use
        input_shape = model_inputs[0].shape
        self.input_width = input_shape[2]
        self.input_height = input_shape[3]

        if os.path.isfile(self.source):
            if self.source.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                frame, total_times = self.process_frame(self.source)
                cv2.imshow('AnimalRTPose', frame)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
            elif self.source.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
                self.process_video(self.source)
            else:
                print("Unsupported file type.")
        elif os.path.isdir(self.source):
            self.process_dir(self.source)
        else:
            self.process_camera()


if __name__ == "__main__":
    # Create an argument parser to handle command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="runs/animalrtpose/train/animalpose/animalrtpose-n/weights/best.onnx", help="Input your ONNX model.")
    parser.add_argument("--source", type=str, default="datasets/animalpose/images_/val", help="Path to input image.")
    parser.add_argument("--conf", type=float, default=0.5, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.5, help="NMS IoU threshold")
    parser.add_argument("--data", type=str, default='configs/data/animalpose.yaml', help="Path to input image.")
    parser.add_argument("--draw-detections", action="store_true", help="Draw bounding boxes and labels on input image.")
    parser.add_argument("--draw-keypoints", action="store_true", help="Draw bounding boxes and labels on input image.")
    parser.add_argument("--draw-skeleton", action="store_true", help="Draw bounding boxes and labels on input image.")

    args = parser.parse_args()

    # Check the requirements and select the appropriate backend (CPU or GPU)
    check_requirements("onnxruntime-gpu" if torch.cuda.is_available() else "onnxruntime")


    # Create an instance of the YOLOv8 class with the specified arguments
    animalrtpose = AnimalRTPose(args.model,
                             args.source,
                             args.conf,
                             args.iou,
                             args.data,
                             args.draw_detections,
                             args.draw_keypoints,
                             args.draw_skeleton)

    animalrtpose.main()

