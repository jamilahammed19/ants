================================================================================
VISDRONE OBJECT DETECTION PROJECT: FASTER R-CNN PIPELINE
This project implements a complete computer vision pipeline for drone-based
object detection using the VisDrone dataset. It utilizes a Faster R-CNN
ResNet50 FPN V2 architecture to detect Vehicles and People.

SYSTEM SETUP & INSTALLATION:
-----------------------------
Hardware Requirement:
Use a T4 GPU (Google Colab) or better. Training on CPU is not feasible.

Installation Commands:
Run these commands at the top of your environment to ensure library compatibility:

!pip install --force-reinstall torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
!pip install albumentations opencv-python pandas tqdm matplotlib Pillow

IMPORTANT: Restart your Python session/runtime after installation to apply
changes.

DATASET PREPARATION:
-------------------------
Unzip your VisDrone dataset into the /content/ directory. Ensure the following
structure is maintained for high-speed local I/O:

/content/visdrone-dataset/VisDrone_Dataset/
├── VisDrone2019-DET-train/
│   ├── images/
│   └── labels/
├── VisDrone2019-DET-val/
│   ├── images/
│   └── labels/
└── VisDrone2019-DET-test-challenge/
└── images/

N.B. code is already implemented to auto unzip to the /content/. Just ensure that the visdrone-dataset.zip will be in /content/drive/MyDrive/ants/visdrone-dataset.zip or in google drive ants/visdrone-dataset.zip

PIPELINE WORKFLOW:
----------------------
Step 1: Metadata Extraction
The script scans your folders to build a primary DataFrame mapping images to
their respective labels and assigning split types (Train=0, Val=1, Test=2).

Step 2: Preprocessing & Augmentation
The VisDroneDataset class performs:

Class Mapping: Vehicles (ID 1), People (ID 2), Background (ID 0).

Normalization: Pixel values scaled to [0, 1].

Albumentations: Letterboxing (640x640), Random Resized Cropping (tiling
alternative), 180-degree rotations, and HSV Jittering.

Step 3: Modular Training

Uses SGD optimizer and ReduceLROnPlateau scheduler.

Automatically saves training history to 'training_history.csv'.

Saves the best model weights to 'best_detection_model.pth'.

Step 4: Inference & Counting

The visualization function loads the trained weights.

It performs object detection on unseen images.

It draws bounding boxes and calculates total counts for each class.

HOW TO RUN THE PROJECT:
---------------------------
LOAD DATA: Place your VisDrone files in /content/ as described in Section 2.

DEFINE CONFIG: Update the Config class if you need to change the
Learning Rate (0.005) or Epochs (10).

INITIALIZE: Run all cells to define classes and helper functions.

TRAIN: Execute the run_training(train_loader, val_loader) function.

PREDICT: Use the visualize_predictions() function with a path to any
image in the test folder to see results.
