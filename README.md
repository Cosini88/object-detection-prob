# WAYMO Dataset performance of the Probabilistic two-stage detection
Two-stage object detectors that use class-agnostic one-stage detectors as the proposal network.


<p align="center"> <img src='projects/CenterNet2/centernet2_docs/centernet2_teaser.jpg' align="center" height="150px"> </p>

> [**Probabilistic two-stage detection**](http://arxiv.org/abs/2103.07461),            
> Xingyi Zhou, Vladlen Koltun, Philipp Kr&auml;henb&uuml;hl,        
> *arXiv technical report ([arXiv 2103.07461](http://arxiv.org/abs/2103.07461))*   
> 
> ## Summary

- Two-stage CenterNet: First stage estimates object probabilities, second stage conditionally classifies objects.

- The best model achieved 56.4 mAP on COCO test-dev. -> This best model was used for evaluation on Waymo Dataset

- This repo has the relevant parts from the CenterNet2 Repo (https://github.com/xingyizhou/CenterNet2)

- This repo has a modified version of the WaymoCOCO converter (https://github.com/shinya7y/WaymoCOCO)

> ### Workflow for data of the WAYMO Dataset
1) Download the Waymo Validation dataset from https://waymo.com/open/download/ (Access is required)
2) Run convert_waymo_to_coco.py in WaymoCOCO folder with the correct configuration (path of dowloaded dataset and output should be updated)
3) Run train_net.py in /projects/CenterNet2, with the following adjustments: 
   1) In the main function, input_folder should be the folder where the WaymoCOCO converter prepared its outputs (the dataset in COCO format)
   2) The name of the new registered dataset should be adjusted
   3) The same dataset name should be given to the relevant config file (.yaml):
      1) The one corresponding to the best performing one on COCO was CenterNet2_R2-101-DCN-BiFPN_1280_4x.yaml
      2) Hence DATASETS/TEST should be set as per the dataset name registered in point ii.
   
The outputs of train_net.py are:
1) the images of the dataset with all ground truths and detected bounding boxes on, in a subfolder (pred_boxes) of the folder of the input images
2) in the output folder of the relevant configuration (e.g. projects/CenterNet2/output/CenterNet2/CenterNet2_R2-101-DCN-BiFPN_1280_4x/):
   1) in the subfolder of the dataset the file coco_instances_results.json and instances_predictions.pth contains information of all detected boxes
   2) iou_tables: IoU tables for each image in the dataset (of the last run)
   3) objs.pkl: has the ground truths, detected boxes, and results (AP) (of the last run)
   4) log.txt file 

For the Detectron2 installation, refer to README_D2.md