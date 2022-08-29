# KL-Loss-pytorch

A pytorch reimplementation of the paper:

**[Bounding Box Regression with Uncertainty for Accurate Object Detection (CVPR'2019)](https://openaccess.thecvf.com/content_CVPR_2019/papers/He_Bounding_Box_Regression_With_Uncertainty_for_Accurate_Object_Detection_CVPR_2019_paper.pdf)**


## Highlights

- We trained KL-Loss with ResNet50-FPN for 2x schedule following the original paper. The results on COCO dataset are as follows:

|       KL Loss      |       Var Vote     |       soft-NMS     | AP (Paper) | AP (Ours) |
| :----------------: | :----------------: | :----------------: | :--------: | :-------: | 
| :x:                | :x:                | :x:                |    37.9    | **38.4**  |
| :heavy_check_mark: | :x:                | :x:                |    38.5    | **39.2**  |
| :heavy_check_mark: | :heavy_check_mark: | :x:                |    38.8    | **39.8**  |
| :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: |    39.2    | **40.2**  |

- This repository is based on [mmdetection](https://github.com/open-mmlab/mmdetection). Pretrained FasterRCNN model is avaliable at [google drive](https://drive.google.com/file/d/1KZxF8n6SXhoZHX-EyHpP8LID1EtFDrWu/view?usp=sharing).


## Installation

[![Python](https://img.shields.io/badge/python-3.7%20tested-brightgreen)](https://www.python.org/)
[![Pytorch](https://img.shields.io/badge/pytorch-1.10.0%20tested-brightgreen)](https://pytorch.org/)

- Set up environment

```
# env
conda create -n kl python=3.7
conda activate kl

# install pytorch
conda install pytorch==1.10.0 torchvision==0.11.0 -c pytorch -c conda-forge
```

- Install 

```
# clone 
git clone https://github.com/cxliu0/KL-Loss-pytorch.git
cd KL-Loss-pytorch

# install dependecies
pip install -r requirements/build.txt

# install mmcv (will take a while to process)
cd mmcv
MMCV_WITH_OPS=1 pip install -e . 

# install OA-MIL
cd ..
pip install -e .
```

## Data Preparation

- Download [COCO](https://cocodataset.org/#download) datasets. The direcory structure is expected to be as follow:

```
KL-Loss-pytorch
├── data
│    ├── coco
│        ├── train2017
│        ├── val2017
│        ├── annotations
│            ├── instances_train2017.json
│            ├── instances_val2017.json
├── configs
├── mmcv
├── ...
```


## Training

The model is trained on 4 NVIDIA RTX 3090 GPUs with a total batch size of 16 (i.e., 4 images per GPU).

- To train KL-Loss-pytorch on COCO, run

```
sh train_coco.sh
```

Please refer to [faster_rcnn_r50_fpn_coco_kl.py](configs/_base_/models/faster_rcnn_r50_fpn_coco_kl.py) for model configuration


## Inference

- Download pretrained FasterRCNN model from [google drive](https://drive.google.com/file/d/1KZxF8n6SXhoZHX-EyHpP8LID1EtFDrWu/view?usp=sharing), and put it in "./pretrained_model/" directory (you can also use your locally trained model)

- Modify [test.sh](test.sh)
```
/path/to/model_checkpoint ---> ./pretrained_model/kl_model.pth (or your locally trained model path)
```

- Modify [faster_rcnn_r50_fpn_coco_kl.py](configs/_base_/models/faster_rcnn_r50_fpn_coco_kl.py)
```
1. set softnms=False/True according to your need
2. set var_vote=False/True according to your need
```

- Run
```
sh test.sh
```

If ```softnms``` and ```var_vote``` are set to ```True```, the results are as follows:

```
Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.402                                                    
Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=1000 ] = 0.587                                                   
Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=1000 ] = 0.438                                                   
Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=1000 ] = 0.223                                                   
Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=1000 ] = 0.440                                                   
Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=1000 ] = 0.529                                                   
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.560                                                    
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=300 ] = 0.560                                                    
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=1000 ] = 0.560                                                   
Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=1000 ] = 0.345                                                   
Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=1000 ] = 0.604                                                   
Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=1000 ] = 0.717                                                   
{'bbox_mAP': 0.402, 'bbox_mAP_50': 0.587, 'bbox_mAP_75': 0.438, 'bbox_mAP_s': 0.223, 'bbox_mAP_m': 0.44, 'bbox_mAP_l': 0.529, 'bbox_mAP_copypaste': '0.402 0.587 0.438 0.223 0.440 0.529'}                                 
```


## Acknowlegdement

- This repository is based on [mmdetection](https://github.com/open-mmlab/mmdetection)

- The implementation is inspired by [KL-Loss](https://github.com/yihui-he/KL-Loss) and [Stronger-yolo-pytorch](https://yihui-he.github.io/Stronger-yolo-pytorch/)

