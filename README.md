[指标汇总对比.csv](https://github.com/user-attachments/files/24450105/default.csv)[metrics.txt](https://github.com/user-attachments/files/24450091/metrics.txt)# Intelligent Surveillance and Behavior Analysis System Based on YOLO and DeepSORT

This project implements a real-time intelligent surveillance system for pedestrian detection, multi-object tracking, and behavior analysis based on YOLO and DeepSORT.

## Description
The system integrates YOLO for pedestrian detection and DeepSORT for multi-object tracking, enabling stable identity tracking in crowded, occluded, and low-light environments.  
The detection model is trained on COCO with additional data from CrowdHuman, ExDark, and BDD to improve robustness under night-time and dense-scene conditions.

To reduce performance gaps between day and night scenarios, data balancing and augmentation strategies are applied during training.  
On top of detection and tracking, region-based monitoring and temporal logic are designed to support abnormal behavior analysis and time-window-based alarm triggering, effectively reducing false alarms and enabling practical deployment.

## Features
- Real-time pedestrian detection using YOLO
- Multi-object tracking with DeepSORT
- Robust performance in crowded and low-light scenes
- Region-based monitoring and behavior analysis
- Time-window-based alarm mechanism
- Practical deployment-oriented design

## Demo
![demo](assets/demo_yolo_deepsort.gif)

## Notes
- Dataset and trained weights are not included.
- This project focuses on system integration and engineering practice rather than model architecture innovation.

## Acknowledgement
This project is based on YOLO and DeepSORT.

## output
experiment	mAP@0.5	mAP@0.5:0.95	Precision	Recall
ped_y11s_val_full	0.8503	0.5589	0.8357	0.7742
ped_y11s_val_full_night	0.8446	0.5316	0.8476	0.7441
ped_y11s_val_pretrain	0.584	0.3006	0.6927	0.5586
ped_y11s_val_pretrain_night	0.5963	0.283	0.6937	0.5699
<img width="518" height="91" alt="image" src="https://github.com/user-attachments/assets/926cff2a-619a-4553-8dce-01541196b309" />


## result
![1月6日 (1)](https://github.com/user-attachments/assets/79f1c8ba-18f7-4a0a-b5ac-69ee5349aef2)
![1月6日 (1)(1)](https://github.com/user-attachments/assets/da9c13a5-b643-4b25-82de-87a83a1a7011)
![1月6日 (2)](https://github.com/user-attachments/assets/c4ad4a4e-8745-4c27-adc7-025c8c49ea8e)
