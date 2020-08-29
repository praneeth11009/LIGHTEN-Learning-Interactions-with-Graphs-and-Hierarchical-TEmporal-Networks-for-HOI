# LIGHTEN-Learning-Interactions-with-Graphs-and-hierarchical-TEmporal-Networks-for-HOI

## Introduction

This repository contains code for **LIGHTEN** HOI detection pipeline, proposed in the ACM MM'20 paper: [LIGHTEN: Learning Interactions with Graph and Hierarchical TEmporal Networks for HOI in videos](). 

![Illustration of human-object interaction detection in video (CAD-120) and image (V-COCO) settings](https://github.com/praneeth11009/LIGHTEN-Learning-Interactions-with-Graphs-and-hierarchical-TEmporal-Networks-for-HOI/blob/master/teaser.PNG)

## Installation 

LIGHTEN is implemented in **Pytorch1.4** with **CUDA-10.1** in **python3.8**. Other python packages can be installed using :  
```
pip install -r requirements.txt
```

## Setting up the codebase

### Datasets
- Download RGB frames for CAD120 videos from [CAD120 dataset page](http://pr.cs.cornell.edu/web3/CAD-120/)
- Download COCO image directory (traintest2017) from [COCO website](https://cocodataset.org/#download)
### Pre-trained models
Download the pretrained models from this [site](link)
### Configuration
Set the corresponding paths to data and pre-trained models in config.py file. Hyper-paramters and model configurations can be set from this file.
The directory structure after setting up looks like : 
```
LIGHTEN-Learning-Interactions-with-Graphs-and-hierarchical-TEmporal-Networks-for-HOI/
  CAD120/
    checkpoints/
      checkpoint_GCN_frame_detection.pth
      checkpoint_GCN_segment_detection.pth
    data/
      training_data.p
      testing_data.p
    models/
  V-COCO/
    checkpoints/
    data/
      training_data.p
      testing_data.p
      action_index.json
      Test_Faster_RCNN_R-50-PFN_2x_VCOCO_Keypoints.pkl
    models/
```

## Running the code
### CAD120
- Resnet frame-wise features can be precomputed and stored beforehand as : 
```
cd CAD120/
python compute_RoI_feats.py
```
  - This will create two new files at CAD120/data/, which contain image features from backbone module. Alternately, the precomputed feature files can be downloaded from this [google drive folder](https://drive.google.com/drive/u/1/folders/1D3hlDb6YN0BvayYF_ij7DESErlA8vcxY).
- Training and Testing the LIGHTEN model for CAD120 can be done as follows : 
```
cd CAD120/
python train_CAD120.py
python test_CAD120.py
```
### V-COCO
- Resnet image features can be precomputed and stored using : 
```
cd V-COCO/
python compute_RoI_feats.py
```
  - This will store resnet features in the directory : V-COCO/data/
- Training, Validation and Testing of LIGHTEN on V-COCO (image-setting) can be done as :
```
cd V-COCO/
python train_VCOCO.py
python eval_VCOCO.py
python test_VCOCO.py
```
- Note that eval_VCOCO.py evaluates only action label detection, and uses ground-truth object detections similar to train_VCOCO.py. However, test_VCOCO.py evaluates the model using faster-RCNN-FPN object detections, and computes the final mAP score as per the evaluation script available at [VSRL Repository](https://github.com/s-gupta/v-coco) 

## Citation
