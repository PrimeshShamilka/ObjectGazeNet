import torch
from torchvision import transforms, models
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import argparse
import os
from datetime import datetime
import shutil
import numpy as np

from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
import cv2

from utils_logging import setup_logger

from models.shashimal2 import Shashimal2
from models.primesh import Primesh
from models.__init__ import save_checkpoint, resume_checkpoint
#from dataloader.shashimal2_synth import GooDataset
from dataloader.shashimal2 import GooDataset
from dataloader.shashimal2 import GazeDataset
from dataloader import chong_imutils
from training.train_primesh import train, GazeOptimizer, train_with_early_stopping
from training.train_shashimal2 import test

logger = setup_logger(name='first_logger',
                      log_dir ='./logs/',
                      log_file='train_chong_gooreal.log',
                      log_format = '%(asctime)s %(levelname)s %(message)s',
                      verbose=True)

# Dataloaders for GOO-Real
batch_size=4
workers=12
images_dir = '/media/primesh/F4D0EA80D0EA4906/PROJECTS/FYP/Gaze detection/Datasets/gooreal/finalrealdatasetImgsV2'
pickle_path = '/media/primesh/F4D0EA80D0EA4906/PROJECTS/FYP/Gaze detection/Datasets/gooreal/oneshotrealhumansNew.pickle'
test_images_dir = '/media/primesh/F4D0EA80D0EA4906/PROJECTS/FYP/Gaze detection/Datasets/gooreal/finalrealdatasetImgsV2'
test_pickle_path = '/media/primesh/F4D0EA80D0EA4906/PROJECTS/FYP/Gaze detection/Datasets/gooreal/testrealhumansNew.pickle'

print ('Train')
train_set = GooDataset(images_dir, pickle_path, 'train')
train_set, val_set = torch.utils.data.random_split(train_set, [2000, 450])
train_data_loader = DataLoader(dataset=train_set,
                                           batch_size=batch_size,
                                           shuffle=True,
                                           num_workers=16)

val_data_loader = DataLoader(dataset=val_set,
                                           batch_size=batch_size,
                                           shuffle=True,
                                           num_workers=16)

print ('Test')
test_set = GooDataset(test_images_dir, test_pickle_path, 'test')
test_data_loader = DataLoader(test_set, batch_size=batch_size//2,
                            shuffle=False, num_workers=8)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model_ft2 = Primesh()


model_ft2 = model_ft2.to(device)

criterion = nn.NLLLoss().cuda()
# criterion

# Observe that all parameters are being optimized
start_epoch = 0
max_epoch = 5
learning_rate = 1e-4

# Initializes Optimizer
gaze_opt = GazeOptimizer(model_ft2, learning_rate)
optimizer = gaze_opt.getOptimizer(start_epoch)
if False:
    checkpoint_fpath = '/gdrive/MyDrive/shashimal2_gazefollow_chechkpoint1.pt'
    checkpoint = torch.load(checkpoint_fpath)
    model_ft2.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])

from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter('runs/primesh')

# model_ft2 = train(model_ft2,train_data_loader, criterion, optimizer, logger, writer,num_epochs=5)
# model_ft2 = train_with_early_stopping(model_ft2, train_data_loader, val_data_loader, criterion, optimizer, logger, writer,num_epochs=5)

test(model_ft2, test_data_loader,logger, save_output=True)