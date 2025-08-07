# PoseVMamba
About the repo for paper: Efficient High-Resolution Visual Representation Learning with State Space Model for Human Pose Estimation.

## **News**
2025/08/07 Code is open source.


## **Abstract**

apturing long-range dependencies while preserving high-resolution visual representations is crucial for dense prediction tasks such as human pose estimation. Vision Transformers (ViTs) have advanced global modeling through self-attention but suffer from quadratic computational complexity with respect to token count, limiting their efficiency and scalability to high-resolution inputs, especially on mobile and resource-constrained devices.
State Space Models (SSMs), exemplified by Mamba, offer an efficient alternative by combining global receptive fields with linear computational complexity, enabling scalable and resource-friendly sequence modeling. However, when applied to dense prediction tasks, existing visual SSMs face key limitations: weak spatial inductive bias, long-range forgetting from hidden state decay, and low-resolution outputs that hinder fine-grained localization.
To address these issues, we propose the Dynamic Visual State Space (DVSS) block, which augments visual state space models with multi-scale convolutional operations to enhance local spatial representations and strengthen spatial inductive biases. Through architectural exploration and theoretical analysis, we incorporate deformable operation into the DVSS block, identifying it as an efficient and effective mechanism to enhance semantic aggregation and mitigate long-range forgetting via input-dependent, adaptive spatial sampling.
We embed DVSS into a multi-branch high-resolution architecture to build HRVMamba, a novel model for efficient high-resolution representation learning. Extensive experiments on human pose estimation, image classification, and semantic segmentation show that HRVMamba performs competitively against leading CNN-, ViT-, and SSM-based baselines.

## Getting Started
The steps to create env, train and evaluate HRVMamba of PoseVMamba models：

```
conda create -n PoseVMamba python=3.10.13
pip install torch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1 --index-url https://download.pytorch.org/whl/cu118  
pip install -U openmim  mim install mmengine 
mim install "mmcv==2.1.0"
mim install "mmdet>=3.1.0"
pip install mmpose==1.3.1
pip install "mmsegmentation>=1.0.0"

cd PoseVMamba/pose_estimation/DCNv4_op
python setup.py build install

cd PoseVMamba/pose_estimation
python setup.py build install
```


## Prepare datasets

It is recommended to symlink the dataset root to `$PoseVMamba/data`.
If your folder structure is different, you may need to change the corresponding paths in config files.

**For COCO data**, please download from [COCO download](http://cocodataset.org/#download), 2017 Train/Val is needed for COCO keypoints training and validation. [HRNet-Human-Pose-Estimation](https://github.com/HRNet/HRNet-Human-Pose-Estimation) provides person detection result of COCO val2017 to reproduce our multi-person pose estimation results. Please download from [OneDrive](https://1drv.ms/f/s!AhIXJn_J-blWzzDXoz5BeFl8sWM-)
Download and extract them under `$HF_HRNET/data`, and make them look like this:

```
HF-HRNet
├── configs
├── tools
`── data
    │── coco
        │-- annotations
        │   │-- person_keypoints_train2017.json
        │   |-- person_keypoints_val2017.json
        |-- person_detection_results
        |   |-- COCO_val2017_detections_AP_H_56_person.json
        │-- train2017
        │   │-- 000000000009.jpg
        │   │-- 000000000025.jpg
        │   │-- 000000000030.jpg
        │   │-- ...
        `-- val2017
            │-- 000000000139.jpg
            │-- 000000000285.jpg
            │-- 000000000632.jpg
            │-- ...

```

**For ImageNet data**, ImageNet is an image database organized according to the WordNet hierarchy. Download and extract ImageNet train and val images from http://image-net.org/. Organize the data into the following directory structure:

```
imagenet/
├── train/
│   ├── n01440764/  (Example synset ID)
│   │   ├── image1.JPEG
│   │   ├── image2.JPEG
│   │   └── ...
│   ├── n01443537/  (Another synset ID)
│   │   └── ...
│   └── ...
└── val/
    ├── n01440764/  (Example synset ID)
    │   ├── image1.JPEG
    │   └── ...
    └── ...
```

**For ADE20K data**, Follow these steps to prepare the ADE20K dataset for pure semantic segmentation tasks:

```
cd <path-to-mambavision_seg-root>
mkdir -p data/ade20k && cd data/ade20k
wget http://data.csail.mit.edu/places/ADEchallenge/ADEChallengeData2016.zip
unzip ADEChallengeData2016.zip
```

Organize the data into the following directory structure:
```
datasets/ade20k/ADEChallengeData2016
```

## Model Training and Inference

### **Human Pose Estimation on COCO**

```
cd PoseVMamba/pose_estimation

###Trainig

sh tools/dist_train.sh XXX exp_21 20338 configs/cocofinal/td-hm_hrvmamba_base_8xb32-210e_coco-384x288.py 8
sh tools/dist_train.sh XXX exp_21 20338 configs/cocofinal/td-hm_hrvmamba_small_8xb32-210e_coco-384x288.py 8
sh tools/dist_train.sh XXX exp_21 20338 configs/cocofinal/td-hm_hrvmamba_base_8xb32-210e_coco-256x192.py 8
sh tools/dist_train.sh XXX exp_21 20338 configs/cocofinal/td-hm_hrvmamba_small_8xb32-210e_coco-256x192.py 8
sh tools/dist_train.sh XXX exp_23 20632 configs/cocofinal/td-hm_hrvmamba_tiny_8xb32-210e_coco-256x192.py 8
sh tools/dist_train.sh XXX exp_21 20338 configs/cocofinal/td-hm_hrformer-base_8xb32-210e_coco-384x288.py 8
sh tools/dist_train.sh XXX exp_21 20338 configs/cocofinal/td-hm_hrformer-small_8xb32-210e_coco-384x288.py 8

###Testing

#sh tools/dist_test.sh XXX exp_21 20338 configs/cocofinal/td-hm_hrvmamba_base_8xb32-210e_coco-384x288.py pretrain_model/hrvmamba_base384288.pth 8 
#sh tools/dist_test.sh XXX exp_21 20338 configs/cocofinal/td-hm_hrvmamba_small_8xb32-210e_coco-384x288.py pretrain_model/hrvmamba_small384288.pth 8
#sh tools/dist_test.sh XXX exp_21 20338 configs/cocofinal/td-hm_hrvmamba_base_8xb32-210e_coco-256x192.py pretrain_model/hrvmamba_base256192.pth 8  
#sh tools/dist_test.sh XXX exp_21 20338 configs/cocofinal/td-hm_hrvmamba_small_8xb32-210e_coco-256x192.py pretrain_model/hrvmamba_small256192.pth 8 
#sh tools/dist_test.sh XXX exp_23 20632 configs/cocofinal/td-hm_hrvmamba_tiny_8xb32-210e_coco-256x192.py pretrain_model/hrvmamba_tiny256192.pth 8 
#sh tools/dist_test.sh XXX exp_23 20632 configs/cocofinal/td-hm_hrformer-tiny_8xb32-210e_coco-256x192.py  pretrain_model/hrformer_tiny256192.pth 8
```

### **Classification on ImageNet-1K**

```
cd PoseVMamba/classification

###Trainig

# sh dist_train.sh XXX small 'configs/hrvmamba/hrvmamba_small.yaml' 8 128
# sh dist_train.sh XXX tiny 'configs/hrvmamba/hrvmamba_tiny.yaml' 8 256 
# sh dist_train.sh XXX nano 'configs/hrvmamba/hrvmamba_nano.yaml' 8 256 
# sh dist_train.sh XXX base 'configs/hrvmamba/hrvmamba_base.yaml' 8 128

# sh dist_train.sh XXX small 'configs/hrvmamba/hrformer_small.yaml' 8 128
# sh dist_train.sh XXX tiny 'configs/hrvmamba/hrformer_tiny.yaml' 8 256 
# sh dist_train.sh XXX nano 'configs/hrvmamba/hrformer_nano.yaml' 8 256 
# sh dist_train.sh XXX base 'configs/hrvmamba/hrformer_base.yaml' 8 128

###Testing

# sh dist_test.sh XXX base 'configs/hrvmamba/hrvmamba_base.yaml' 4 128 pretrain_model/hrvmamba_base.pth 
# sh dist_test.sh XXX nano 'configs/hrvmamba/hrvmamba_nano.yaml' 4 256 pretrain_model/hrvmamba_nano.pth 
# sh dist_test.sh XXX small 'configs/hrvmamba/hrvmamba_small.yaml' 4 256 pretrain_model/hrvmamba_small.pth 
# sh dist_test.sh XXX tiny 'configs/hrvmamba/hrvmamba_tiny.yaml' 4 256 pretrain_model/hrvmamba_tiny.pth 

# sh dist_test.sh XXX base 'configs/hrvmamba/hrformer_base.yaml' 4 128 pretrain_model/hrformer_base_best.pth
# sh dist_test.sh XXX nano 'configs/hrvmamba/hrformer_nano.yaml' 4 256 pretrain_model/hrformer_nano_best.pth 
# sh dist_test.sh XXX small 'configs/hrvmamba/hrformer_small.yaml' 4 256 pretrain_model/hrformer_small_best.pth 
# sh dist_test.sh XXX tiny 'configs/hrvmamba/hrformer_tiny.yaml' 4 256 pretrain_model/hrformer_tiny_best.pth  

```


### **Semantic Segmentation on ADE20K**

```
cd PoseVMamba/semantic_segmentation

###Trainig

# sh tools/dist_train.sh XXX exp 'configs/hrvmamba/hrvmamba_base_160k_ade20k-512x512_base.py' 8 27131

###Testing

# sh tools/dist_test.sh XXX exp 'configs/hrvmamba/hrvmamba_base_160k_ade20k-512x512_base.py' 8 17234 hrvmamba_base_160k_ade20k-512x512_base/iter_160000.pth

```

## Main Results

### **Human Pose Estimation on COCO**
|      Model     | Input Size | Param (M) | FLOPs (G) |  AP  |                                             ckpts                                             |
|:--------------:|:----------:|:---------:|:---------:|:----:|:-----------------------------------------------------------------------------------------------:|
|  HRFormer-Tiny |   256x192  |     2     |    1.1    | 68.3 |   [ckpt](https://drive.google.com/file/d/1hS_IuLcLIAdryramUoCOLL-5tNS9gLp7/view?usp=sharing)  |
| HRFormer-Small |   256x192  |     8     |    3.3    | 74.0 |                                               ckpt                                            |
|  HRFormer-Base |   224x192  |     43    |    14.1   | 75.6 |                                               ckpt                                            |
|  HRVMamba-Tiny |   256x192  |     2     |    1.1    | 69.5 |   [ckpt](https://drive.google.com/file/d/1w1wYEUL6dS3KYMeAX3XV7O1Vnz-IATDw/view?usp=sharing)  |
| HRVMamba-Small |   256x192  |     8     |    3.3    | 74.6 |   [ckpt](https://drive.google.com/file/d/1L3CO3D4L1Q1galRaSC82taDDqFqmukKt/view?usp=sharing)  |
|  HRVMamba-Base |   256x192  |     47    |    14.2   | 76.5 |   [ckpt](https://drive.google.com/file/d/14JE7v6jdxtJaEMJCatqbmuhLkfBYVBOX/view?usp=sharing)  |

|      Model     | Input Size | Param (M) | FLOPs (G) |  AP  |                                             ckpts                                             |
|:--------------:|:----------:|:---------:|:---------:|:----:|:-----------------------------------------------------------------------------------------------:|
| HRFormer-Small |   384x288  |     8     |    7.3    | 75.6 |                                               ckpt                                            |
|  HRFormer-Base |   384x288  |     43    |    30.9   | 77.2 |                                               ckpt                                            |
| HRVMamba-Small |   384x288  |     8     |    7.4    | 76.4 |   [ckpt](https://drive.google.com/file/d/1aqkUoQog4hl1dSbkUkwQvlwgistpxtzQ/view?usp=sharing)  |
|  HRVMamba-Base |   384x288  |     47    |    32.0   | 77.7 |   [ckpt](https://drive.google.com/file/d/161d_LhWBBvBOSXxPBGRBNCjYr8PP0w9c/view?usp=sharing)  |

### **Classification on ImageNet-1K**
|      Model     | Input Size | Param (M) | FLOPs (G) | Top-1 Acc |                                              ckpts                                         |
|:--------------:|:----------:|:---------:|:---------:|:---------:|:------------------------------------------------------------------------------------------:|
|  HRFormer-Nano |   256x256  |     12    |    1.9    |    74.3   | [ckpt](https://drive.google.com/file/d/19QivtJS5wqstLHCGRSkAPp29Wo7s9WsK/view?usp=sharing) |
|  HRFormer-Tiny |   256x256  |     14    |    2.8    |    77.8   | [ckpt](https://drive.google.com/file/d/1qm3tCOxHRTRY1CpETxNSQ11MFlrPYOkx/view?usp=sharing) |
| HRFormer-Small |   256x256  |     20    |    6.1    |    80.8   | [ckpt](https://drive.google.com/file/d/1Tk7GFEJsanWsN6dEZNoh8jLqWsKBKUhA/view?usp=sharing) |
|  HRFormer-Base |   224x224  |     57    |    14.5   |    83.3   | [ckpt](https://drive.google.com/file/d/15Mnr4xTXwok9NiCEkIq47WosvQO3NoR5/view?usp=sharing) |
|  HRVMamba-Nano |   256x256  |     12    |    1.9    |    74.8   | [ckpt](https://drive.google.com/file/d/1aGN51e9tgMTdDDWm0wwCmGibAksp2QJ6/view?usp=sharing) |
|  HRVMamba-Tiny |   256x256  |     14    |    2.8    |    78.6   | [ckpt](https://drive.google.com/file/d/1Vn-ZXzvx3Hrv_C2tDQWrOjtA9WxTvWaa/view?usp=sharing) |
| HRVMamba-Small |   256x256  |     20    |    5.8    |    81.3   | [ckpt](https://drive.google.com/file/d/1Gxo0Pxg2D7x3dU0P2AZ2YGEFS-7XMxIX/view?usp=sharing) |
|  HRVMamba-Base |   224x224  |     61    |    15.8   |    84.2   | [ckpt](https://drive.google.com/file/d/1jbXjvCBdjgraeEaJrDjD2wStpxJtCEuy/view?usp=sharing) |

### **Semantic Segmentation on ADE20K**
| Model | mIoU (SS) | mIoU (MS) | Ckpt |
|:---------------|:----:|:--:|
| HRVMamba-Base |   51.4   |   52.2   |   [ckpt](https://drive.google.com/file/d/1mc_a_DpsazSkjwqChttemyMXyN_yrLbg/view?usp=sharing)   |

## **Acknowledgement**:
This project is developed based on the [MMPOSE](https://github.com/open-mmlab/mmpose).
The segmentation pipeline is from [MMCV](https://github.com/open-mmlab/mmcv) and [MMSegmentation](https://github.com/open-mmlab/mmsegmentation). 


