# Animal Pose Estimation with AnimalRTPose

Aanimal pose estimation aims to detect the keypoints of different species. It provides detailed behavioral analysis for neuroscience, medical and ecology applications. Some results are shown below.
![](https://s3.bmp.ovh/imgs/2024/08/19/0e1d3cc45f840729.jpg)



https://github.com/user-attachments/assets/bb694533-0344-453d-b186-46a3c66534f5



## Installation

1. Create a conda virtual environment and activate it.

```
conda create -n animalrtpose python=3.10
conda activate animalrtpose
```

2. Install PyTorch (rqeuired version: 2.1.0) and torchvision following the [official instructions](https://pytorch.org/).

```
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu118
```

3. Clone the ultralytics repository and install the required dependencies.

```
git clone https://github.com/wux024/ultralytics.git
cd ultralytics
git checkout animalrtpose
pip install -v -e .
```
or 
```
pip install git+https://github.com/ultralytics/ultralytics.git@animalrtpose
```

## Training

1. The datasets (e.g. Animal Pose) used AnimalRTPose could be downloaded by contacting the corresponding author and me (<EMAIL>wux024@nenu.edu.cn). Extract the dataset to the `data` folder.
```
ultralytics
├── ultralytics
├── docs
├── tests
├── tools
├── configs
├── weights
├── datasets
    │── animalpose
        ├── images_
        ├── labels
        |── annotations
    |—— other datasets
```

2. Run the following command to train the model:
```
python tools/pose_train.py --dataset animalpose --model-type animalrtpose
```
The pretrained model contact us for download. You should move the downloaded model to the `weights` folder.

3. The training log and checkpoints will be saved in the `runs/animalrtpose/train/apt36k/animalrtpose-n` folder.

4. Test the trained model:
``` 
python tools/pose_val.py --dataset animalpose --models n --save-json
python tools/pose_modify_categories_id.py --dataset animalpose
python tools/pose_coco_eval.py --dataset animalpose --model-type animalrtpose
```

## Animal Pose Estimation Results on Benchmark Datasets

### Animal Pose Dataset

| Model | size(pixels) | mAP50-95 | CPU(ms) | 2080Ti(ms) | 3090Ti(ms) | A800(ms) | Params(M) | GFLOPs |
| :-----: | :------------: | :--------: | :--------:| :-----------: | :--------: | :--------: | :---------: | :------: |
| AnimalRTPose-N| 640 | 72.0| 61.3| 2.1| 1.3 | 0.9 | 2.9 | 8.5 |
| AnimalRTPose-S| 640 | 76.0| 135.1| 3.6| 2.5 | 1.3 | 9.8 | 26.6 |
| AnimalRTPose-M| 640 | 77.2| 291.5| 7.5 | 4.9 | 2.4| 24.2|65.6|
| AnimalRTPose-L| 640 | 77.8| 506.5| 11.4 | 7.5  | 3.7 | 47.9|131.2|
| AnimalRTPose-X| 640 | 80.1| 722.8| 17.1 |  11.9 | 5.6| 74.8|204.6|

- **mAP<sup>val</sup>** values are for single-model single-scale on [Animal Pose](https://sites.google.com/view/animal-pose/) dataset. <br>Reproduce by `python tools/pose_val_.py --dataset animalpose && python tools/pose_modify_categories_id.py --dataset animalpose && python tools/pose_coco_eval.py --dataset animalpose --model yolov8n-pose-cspnext`.
- **Speed** averaged over COCO val images using [NVIDIA RTX 2080 Ti](https://www.nvidia.com/en-us/geforce/graphics-cards/rtx-2080-ti/) , [NVIDIA RTX 3090](https://www.nvidia.com/en-us/geforce/graphics-cards/30-series/rtx-3090/), [NVIDIA A800](https://www.nvidia.cn/content/dam/en-zz/Solutions/Data-Center/a100/pdf/PB-10577-001_v02.pdf) and [12th Gen Intel(R) Core(TM) i5-12400   2.50 GHz](https://www.intel.cn/content/www/cn/zh/products/sku/134586/intel-core-i512400-processor-18m-cache-up-to-4-40-ghz/specifications.html) instance. <br>Reproduce by `python tools/pose_val_.py --dataset animalpose`

### APT36K 

| Model | size(pixels) | mAP50-95 | CPU(ms) | 2080Ti(ms) | 3090Ti(ms) | A800(ms) | Params(M) | GFLOPs |
| :-----: | :------------: | :--------: | :--------:| :-----------: | :--------: | :--------: | :---------: | :------: |
| AnimalRTPose-N| 640 | 68.2| -| 2.0| 1.2 | - | 2.8 | 8.2 |
| AnimalRTPose-S| 640 | 78.1| -| 3.4| 2.3 | - | 9.7 | 26.3 |
| AnimalRTPose-M| 640 | 80.9| -| 7.4 | 4.7 | -| 25.1|66.0|
| AnimalRTPose-L| 640 | 82.3| -| 11.7 | 7.4  | - | 51.8|134.3|
| AnimalRTPose-X| 640 | 85.5| -| 17.6 |  11.9 | -| 80.8|209.5|

- **mAP<sup>val</sup>** values are for single-model single-scale on [APT36K](https://github.com/pandorgan/APT-36K?tab=readme-ov-file#demo) dataset. <br>Reproduce by `python tools/pose_val_.py --dataset animalpose && python tools/pose_modify_categories_id.py --dataset animalpose && python tools/pose_coco_eval.py --dataset animalpose --model yolov8n-pose-cspnext`.
- **Speed** averaged over COCO val images using [NVIDIA RTX 2080 Ti](https://www.nvidia.com/en-us/geforce/graphics-cards/rtx-2080-ti/) , [NVIDIA RTX 3090](https://www.nvidia.com/en-us/geforce/graphics-cards/30-series/rtx-3090/), [NVIDIA A800](https://www.nvidia.cn/content/dam/en-zz/Solutions/Data-Center/a100/pdf/PB-10577-001_v02.pdf) and [12th Gen Intel(R) Core(TM) i5-12400   2.50 GHz](https://www.intel.cn/content/www/cn/zh/products/sku/134586/intel-core-i512400-processor-18m-cache-up-to-4-40-ghz/specifications.html) instance. <br>Reproduce by `python tools/pose_val.py --dataset animalpose`

### ATRW

| Model | size(pixels) | mAP50-95 | CPU(ms) | 2080Ti(ms) | 3090Ti(ms) | A800(ms) | Params(M) | GFLOPs |
| :-----: | :------------: | :--------: | :--------:| :-----------: | :--------: | :--------: | :---------: | :------: |
| AnimalRTPose-N| 640 | 82.8| -| 2.1| - | - | 2.8 | 8.0 |
| AnimalRTPose-S| 640 | 83.9| -| 3.6| - | - | 9.6 | 25.9 |
| AnimalRTPose-M| 640 | 84.1| -| 7.6 | - | -| 24.0|64.9|
| AnimalRTPose-L| 640 | 84.7| -| 11.4 | -  | - | 47.9|131.2|
| AnimalRTPose-X| 640 | 88.4| -| 17.2 |  - | -| 74.8|204.5|

- **mAP<sup>val</sup>** values are for single-model single-scale on [ATRW](https://cvwc2019.github.io/index.html#body-home) dataset. <br>Reproduce by `python tools/pose_val_.py --dataset animalpose && python tools/pose_modify_categories_id.py --dataset animalpose && python tools/pose_coco_eval.py --dataset animalpose --model yolov8n-pose-cspnext`.
- **Speed** averaged over COCO val images using [NVIDIA RTX 2080 Ti](https://www.nvidia.com/en-us/geforce/graphics-cards/rtx-2080-ti/) , [NVIDIA RTX 3090](https://www.nvidia.com/en-us/geforce/graphics-cards/30-series/rtx-3090/), [NVIDIA A800](https://www.nvidia.cn/content/dam/en-zz/Solutions/Data-Center/a100/pdf/PB-10577-001_v02.pdf) and [12th Gen Intel(R) Core(TM) i5-12400   2.50 GHz](https://www.intel.cn/content/www/cn/zh/products/sku/134586/intel-core-i512400-processor-18m-cache-up-to-4-40-ghz/specifications.html) instance. <br>Reproduce by `python tools/pose_val.py --dataset animalpose`


### TopViewMouse-5K

| Model | size(pixels) | mAP50-95 | CPU(ms) | 2080Ti(ms) | 3090Ti(ms) | A800(ms) | Params(M) | GFLOPs |
| :-----: | :------------: | :--------: | :--------:| :-----------: | :--------: | :--------: | :---------: | :------: |
| AnimalRTPose-N| 640 | 74.0| -| 2.9| - | - | 3.1 | 9.3 |
| AnimalRTPose-S| 640 | 82.4| -| 4.4| - | - | 10.1 | 27.6 |
| AnimalRTPose-M| 640 | 82.5| -| 7.9 | - | -| 24.5|66.9|
| AnimalRTPose-L| 640 | 83.2| -| 11.8 | -  | - | 48.3|132.5|
| AnimalRTPose-X| 640 | 83.4| -| 17.3 |  - | -| 74.8|204.7|

- **mAP<sup>val</sup>** values are for single-model single-scale on [TopViewMouse-5K](https://doi.org/10.5281/zenodo.10618947) dataset. <br>Reproduce by `python tools/pose_val_.py --dataset animalpose && python tools/pose_modify_categories_id.py --dataset animalpose && python tools/pose_coco_eval.py --dataset animalpose --model yolov8n-pose-cspnext`.
- **Speed** averaged over COCO val images using [NVIDIA RTX 2080 Ti](https://www.nvidia.com/en-us/geforce/graphics-cards/rtx-2080-ti/) , [NVIDIA RTX 3090](https://www.nvidia.com/en-us/geforce/graphics-cards/30-series/rtx-3090/), [NVIDIA A800](https://www.nvidia.cn/content/dam/en-zz/Solutions/Data-Center/a100/pdf/PB-10577-001_v02.pdf) and [12th Gen Intel(R) Core(TM) i5-12400   2.50 GHz](https://www.intel.cn/content/www/cn/zh/products/sku/134586/intel-core-i512400-processor-18m-cache-up-to-4-40-ghz/specifications.html) instance. <br>Reproduce by `python tools/pose_val.py --dataset animalpose`
