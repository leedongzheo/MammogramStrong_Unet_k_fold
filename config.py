from train import get_args
import random
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
from torch.optim import Adam,SGD,AdamW
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from sklearn.model_selection import train_test_split
from torchvision import transforms
from imutils import paths
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import time
import pandas as pd
# import the necessary packages
import os
import cv2
import gc
# base path of the dataset
import shutil
import sys
from torch.optim.lr_scheduler import CosineAnnealingLR, MultiStepLR, ReduceLROnPlateau, CosineAnnealingWarmRestarts, LinearLR, SequentialLR
import segmentation_models_pytorch as smp
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from torch.nn.utils import clip_grad_norm_
from matplotlib.colors import ListedColormap
"""Phần I: Xét các tham số"""
# THAM SỐ HẰNG SỐ
# Đặt seed để đảm bảo tái hiện kết quả
SEED=42
torch.manual_seed(SEED)
# THAM SỐ VỪA LÀ HẰNG SỐ VỪA THAY ĐỔI
INIT_LR = 1e-4
# lr0= INIT_LR
BATCH_SIZE = 8
# WEIGHT_DECAY=1e-6
WEIGHT_DECAY=1e-4
WEIGHT_DECAY1=0
WEIGHT_DECAY2=0.05
# weight_decay = 1e-6  # Regularization term to prevent overfitting
INPUT_IMAGE_WIDTH = 640
INPUT_IMAGE_HEIGHT = 640
NUM_CLASSES = 1
# BETA = (0.99, 0.999)
BETA = (0.9, 0.999)
AMSGRAD=False
WARMUP_EPOCHS = 10
"""Phần II: Xử lý logic"""
args = get_args()
#  Tham số trường hợp:
augment = args.augment
loss = args.loss
optim = args.optimizer

# tham số vừa là hằng số vừa thay đổi:
lr0 = args.lr0 if args.lr0 else INIT_LR
batch_size = args.batchsize if args.batchsize else BATCH_SIZE
weight_decay1 = args.weight_decay if args.weight_decay else WEIGHT_DECAY1
weight_decay2 = args.weight_decay if args.weight_decay else WEIGHT_DECAY2
input_image_width, input_image_height = args.img_size if args.img_size else [INPUT_IMAGE_WIDTH,INPUT_IMAGE_HEIGHT]
numclass = args.numclass if args.numclass else NUM_CLASSES
warmup_epochs = args.warmup if args.warmup else WARMUP_EPOCHS
# THAM SỐ LUÔN THAY ĐỔI THEO nhap.py
NUM_EPOCHS = args.epoch
T_max = NUM_EPOCHS  # T_max là số epoch bạn muốn dùng cho giảm lr
lr_min = 0.0001  # lr_min là learning rate tối thiểu
# CÁC THAM SỐ KHÁC
DEVICE=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
PIN_MEMORY = True if str(DEVICE) == "cuda:0" else False
# DATASET_PATH = args.data
DATASET_ROOT_4FOLDS = args.data if args.data else "INBREAST_4folds_merge_640"
# Mean/Std cho từng Fold (Lưu vào dict để dễ lấy)
FOLD_STATS = {
    0: {'mean': [0.2912, 0.2933, 0.2072], 'std': [0.2892, 0.2644, 0.2167]},
    1: {'mean': [0.2952, 0.2972, 0.2102], 'std': [0.2893, 0.2646, 0.2170]},
    2: {'mean': [0.2870, 0.2891, 0.2045], 'std': [0.2910, 0.2659, 0.2177]},
    3: {'mean': [0.2913, 0.2938, 0.2072], 'std': [0.2882, 0.2635, 0.2158]},
    # 4: {'mean': [0.2879, 0.2908, 0.2047], 'std': [0.2870, 0.2629, 0.2149]},
}
# Metadata file map
FOLD_METADATA = {
    0: "train_metadata_area_INBREAST_4folds_merge_640_fold0.csv",
    1: "train_metadata_area_INBREAST_4folds_merge_640_fold1.csv",
    2: "train_metadata_area_INBREAST_4folds_merge_640_fold2.csv",
    3: "train_metadata_area_INBREAST_4folds_merge_640_fold3.csv",
    # 4: "train_metadata_area_INBREAST_5folds_merge_640_fold4.csv",
}

# TRAIN_PATH = os.path.join(DATASET_PATH, "train")
# VALID_PATH = os.path.join(DATASET_PATH, "valid")
# TEST_PATH = os.path.join(DATASET_PATH, "test")
# # define the path to the images and masks dataset
# IMAGE_TRAIN_PATH = os.path.join(TRAIN_PATH, "images")
# MASK_TRAIN_PATH = os.path.join(TRAIN_PATH, "masks")

# IMAGE_VALID_PATH = os.path.join(VALID_PATH, "images")
# MASK_VALID_PATH = os.path.join(VALID_PATH, "masks")

# IMAGE_TEST_PATH = os.path.join(TEST_PATH, "images")
# MASK_TEST_PATH = os.path.join(TEST_PATH, "masks")
BASE_OUTPUT = args.saveas if  args.saveas else "output"
