# The official code for Grizzly Bear ReID Project.

Identifying individuals within a species is a cornerstone of ecological and biological research, holding the key to understanding behavior, population dynamics, and conservation needs. However, conventional identification methods such as genetic sampling and physical tags come with significant trade-offs, including invasiveness, high costs, and limited scalability. Despite remarkable advances in computer vision that have transformed the identification of patterned species, reliably distinguishing individuals in unmarked species remains an open challenge. Here, we curate a novel dataset of visually identified individual Alaskan coastal brown bears (Ursus arctos). This 72K image dataset contains high-resolution images of 109 known individuals from multiple seasons and varied conditions (e.g., fur shed, substantial weight gain). Identification of these bears is notoriously difficult, yet our new pose-aware metric-learning-based AI model reveals hidden biometric patterns that enable individual re-identification (ReID) at a level practicable for research and conservation efforts. Our findings indicate that it is possible to reidentify unmarked individuals across years and to detect unknown individuals in a 'real-world' open dataset. We anticipate that this new AI-based approach to non-invasively ReID animals can serve as the basis for new methodologies in ecological and biological inquiry for other unmarked species.

> ⚠️ Data can be found at [https://zenodo.org/records/xxx](https://zenodo.org/records/xxx)


## General informations

* **Authors**: Beth Rosenberg<sup>1,3**</sup>, Mu Zhou<sup>2,3</sup>, Nathan Wolf<sup>1</sup>, Mackenzie W. Mathis<sup>2</sup>, Bradley P. Harris<sup>1</sup>, Alexander Mathis<sup>2,4*</sup> 
* **Affiliation**: <sup>1</sup> Fisheries, Aquatic Science, and Technology Laboratory, Alaska Pacific University, USA, <sup>2</sup> Brain Mind Institute and Neuro-X Institute, School of Life Sciences, École Polytechnique Fédérale de Lausanne (EPFL), Switzerland, <sup>3</sup>  These authors contributed equally, <sup>4</sup> Lead contact, <sup>*</sup> Correspondence: alexander.mathis@epfl.ch, <sup>**</sup> Correspondence: brosenberg@alaskapacific.edu

* **Date of collection**: 2017-2022

## Dataset availability

* **License**: This dataset is released under the non-commercial [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/legalcode) license. 
* **Citation**: Please consider citing the associated publication when using our data. 
* **Repository URL**: [https://github.com/amathislab/BrownBear_ReID](https://github.com/amathislab/BrownBear_ReID)
* **Dataset version**: v1

## Data and files overview
* **Data preparation**: unzip `Public_release.zip` 
* **Repository structure**:

```
Public_release
├── checkpoints
│   ├── preprocessing_ckpts
│   │   ├── detectors
│   │   │   ├── body_detector/
│   │   │   └── face_detector/
│   │   └── pose/
│   │
│   └── reid_ckpts
│       ├── 2017_2022_six_exps
│       │   ├── test_on_2017
│       │   │   └── net_xx.pth
│       │   ├── test_on_2018/
│       │   ├── test_on_2019/
│       │   ├── test_on_2020/
│       │   ├── test_on_2021/
│       │   └── test_on_2022/
│       │
│       └── katmai_exps
│           ├── 1y_model
│           │   └── net_xx.pth
│           ├── 3y_model/
│           └── 6y_model/
│
├── data
│   ├── GT
│   │   ├── coco_bearface_new/
│   │   ├── head_orientation/
│   │   ├── mmpose_faceposeestimation/
│   │   └── superface_data/
│   │
│   └── train_test_reid
│       ├── test_on_2017/
│       │   ├── train_iid.csv
│       │   ├── val_iid.csv
│       │   ├── test_iid.csv
│       │   └── test_ood.csv
│       ├── test_on_2018/
│       ├── test_on_2019/
│       ├── test_on_2020/
│       ├── test_on_2021/
│       └── test_on_2022/
│
└── README.md
```
### Checkpoints Overview

It contains the checkpoints for all processing and re-identification models used in the project.
Including:
(1) Preprocessing models (body detector, head detector, pose estimator), (2) ReID models: six-year ReID experiments, Katmai cross-site experiments.

### Data Overview
It contains the preprocessing annotations (including head detection, face pose estimation) and the ReID dataset (6 experiments with train/val/iid/ood splits).

> We refer the reader to the associated publication for details about data processing and splits description.


# Train your own model on your data
## Data Preparation

### Raw Data -> Camera Trap: Body Images (saved to csv) -> Faster-RCNN: Head Images (saved to csv) 

Note: save your raw images under the $root/raw folder.
The script will run cameratrap and faster-rcnn to process the raw images and save the body and head images to the **$root/body** and **$root/head** folders, respectively, and save the csv files to the **$root/annotation** folder.

```
bash data_preprocessing/run_body_head_detector.sh
```

Structure of the $root folder after running the script:
```
$root/
├── raw/                           # Raw images
│   └── {id}/
│       └── *.JPG
│
├── body/                          # Cropped body images
│   ├── detection_output/          # CameraTrap detection results
│   │   └── output_{id}.json
│   │
│   └── images_uncurated/          # Uncurated body crops from CameraTrap
│       └── {id}/                  # Bear ID
│           ├── single/            # Single bear detected in raw image
│           │   └── *.JPG
│           └── multiple/          # Multiple bears (or animals) detected (needs curation)
│               └── *.JPG
│
├── head/                          # Cropped head images
│   ├── images_uncurated/          # Head crops from Faster-RCNN (uncurated)
│   │   └── {id}/
│   │       ├── single/            # Single bear head detected from body images ($root/body/images_uncurated/{id}/single/)
│   │       │   ├── images/
│   │       │   │   └── *.JPG
│   │       │   └── heads.csv      
│   │       └── multiple/          # Multiple bear heads detected from body images ($root/body/images_uncurated/{id}/multiple/)
│   │           ├── images/        # Cropped bear head images
│   │           │   └── *.JPG
│   │           ├── empty/         # Non-bear head detections (e.g., birds)
│   │           │   └── *.JPG
│   │           └── heads.csv      
│   │
│   └── images_curated/            # **** Manually curated head images by Expert ****
│       └── {id}/
│           ├── single/
│           │   ├── images/
│           │   │   └── *.JPG
│           │   └── heads.csv
│           └── multiple/
│               ├── images/
│               │   └── *.JPG
│               └── heads.csv
│
├── annotation/                    # Annotation metadata CSVs
│   ├── raw.csv                    # Raw image-level metadata
│   ├── body.csv                   # Body detection metadata
│   ├── head.csv                   # Head detection metadata (uncurated)
│   └── head_curated.csv           # Head detection metadata (curated) -> Once you curate the head images, you can rerun the script to get the head_curated.csv
```



## PossSwin ReID

The code is under the [poss_swin_reid](./PoseGuidedReID) folder.

### Installation

```
cd PoseGuidedReID
conda create -n poss_swin python=3.8
conda activate poss_swin
pip install -r requirements.txt
```

### Training & evaluation
```
cd PoseGuidedReID/scripts
year=2017
gpu_id=1
bash run_swin_pose.sh $year $gpu_id
```



## Acknowledgements
We thank Steven A. Rosenberg and members of the Mathis group for feedback on earlier versions of this manuscript. We thank Maxime Vidal for preliminary data processing and analysis; Julia Ditto and Felipe Restrepo for help with figures; Larry Aumiller, Kelly Debure and Teresa Fish for preliminary study design; MB, LB, JH for citizen science images. BR, NW and BPH are grateful for Alaska Education Tax Credit funding donated by the At-Sea Processors Association. We are grateful to EPFL’s School of Life Sciences PTECH fund for providing funding (A.M.).

## Change log (DD.MM.YYYY)
[15.10.2025]: First data & code release ! 


