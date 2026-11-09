#!/bin/bash


atc --model=runs/animalrtpose/train/animalpose/animalrtpose-s/weights/best.onnx --frame=5 --output=best --soc_version=Ascend310B4
mv best.om animalrtpose-s.om
atc --model=runs/animalrtpose/train/animalpose/animalrtpose-m/weights/best.onnx --frame=5 --output=best --soc_version=Ascend310B4
mv best.om animalrtpose-m.om
atc --model=runs/animalrtpose/train/animalpose/animalrtpose-l/weights/best.onnx --frame=5 --output=best --soc_version=Ascend310B4
mv best.om animalrtpose-l.om
atc --model=runs/animalrtpose/train/animalpose/animalrtpose-x/weights/best.onnx --frame=5 --output=best --soc_version=Ascend310B4
mv best.om animalrtpose-x.om
