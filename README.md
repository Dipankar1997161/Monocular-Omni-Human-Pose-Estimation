# Monocular-Omni-Human-Pose-Estimation

The following repo perfoms Human Pose Estimation and Object Recognition (In this case person)

The following model can also be switched just for Object Recogtion Purposes.

I used CenterNet for the following task. To know more about CenterNet: https://arxiv.org/abs/1904.07850

## Human Pose Estimation:
Human Pose Estimation (HPE) is a way of identifying and classifying the joints in the human body.

Essentially it is a way to capture a set of coordinates for each joint (arm, head, torso, etc.,) which is known as a key point that can describe a pose of a person. The connection between these points is known as a pair.
The connection formed between the points has to be significant, which means not all points can form a pair. From the outset, the aim of HPE is to form a skeleton-like representation of a human body and then process it further for task-specific applications

## Dataset
In this project, the dataset used are mostly Top-view/ Fisheye Dataset.

Three sets of Datasets are used: 
Synthetic Dataset: THEODORE https://www.tu-chemnitz.de/etit/dst/forschung/comp_vision/datasets/theodore/index.php.en

Real-Dataset: 
FES https://www.tu-chemnitz.de/etit/dst/forschung/comp_vision/datasets/fes/index.php.en

COCO: https://cocodataset.org/#home

## Experiments

Training was carried out on a Synthetic Dataset in order to check its effects in comparision with real dataset.
This was intentionally done since avaiiability of larger amount of dataset is not always feasible except COCO and others which are public dataset.

A total of 10 experiments were carried out until the final results were depicted. 
### Transfer Learning, Training from Sratch, Training with Noise, Mixed Data Training are a few major ones.

The Project used 2 backbone architectures for CenterNet:
    
    DLA - Deep Layer Aggregration
    Hourglass
    
## Results

1. The results were tested in two scenarios and all of them are Omnidirectional/ Fisheye/ Top-view Images.
2. Evaluation was done in 2 aspects: 13 Keypoints and 17 Keypoints

        Scenario 1: one to two people in an image
        Scenario 2: More than 5 people in an image

**The Mean Average Precision values can be found in the "mAP_results" folder** 

**The Keypoint and detection results can be found in the "Image_results" folder**

## Please Note: The Images are too big to display, hence adding a snap ::((
![Scenario 1 detection](https://user-images.githubusercontent.com/85514219/227794635-54710f78-ac3b-406e-9285-5e017780b000.png)
![Senario 2 detection](https://user-images.githubusercontent.com/85514219/227794637-2127f11a-16ca-48a4-bec6-fcc106196eb4.png)

## I hope you like my project. Thank you
