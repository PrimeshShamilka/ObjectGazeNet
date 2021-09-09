
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
from models.__init__ import save_checkpoint, resume_checkpoint
#from dataloader.shashimal2_synth import GooDataset
from dataloader.shashimal2 import GooDataset
from dataloader import chong_imutils
from training.train_shashimal2 import train, test, GazeOptimizer



logger = setup_logger(name='first_logger',
                      log_dir ='./logs/',
                      log_file='train_chong_gooreal.log',
                      log_format = '%(asctime)s %(levelname)s %(message)s',
                      verbose=True)



batch_size=4
workers=4

images_dir = '/media/primesh/F4D0EA80D0EA4906/PROJECTS/FYP/Gaze detection/Datasets/gooreal/finalrealdatasetImgsV2'
pickle_path = '/media/primesh/F4D0EA80D0EA4906/PROJECTS/FYP/Gaze detection/Datasets/gooreal/oneshotrealhumansNew.pickle'
test_images_dir = '/media/primesh/F4D0EA80D0EA4906/PROJECTS/FYP/Gaze detection/Datasets/gooreal/finalrealdatasetImgsV2'
test_pickle_path = '/media/primesh/F4D0EA80D0EA4906/PROJECTS/FYP/Gaze detection/Datasets/gooreal/testrealhumansNew.pickle'

train_set = GooDataset(images_dir, pickle_path, 'train')
train_data_loader = DataLoader(dataset=train_set,
                                           batch_size=batch_size,
                                           shuffle=True,
                                           num_workers=4)

test_set = GooDataset(test_images_dir, test_pickle_path, 'test')
test_data_loader = DataLoader(test_set, batch_size=batch_size//2,
                            shuffle=False, num_workers=4)



import gc

gc.collect()

torch.cuda.empty_cache()


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model_ft = Shashimal2()


model_ft = model_ft.to(device)

criterion = nn.NLLLoss().cuda()
# criterion =

# Observe that all parameters are being optimized
start_epoch = 0
max_epoch = 5
learning_rate = 1e-4

# Initializes Optimizer
gaze_opt = GazeOptimizer(model_ft, learning_rate)
optimizer = gaze_opt.getOptimizer(start_epoch)

img, face, head_channel, object_channel,gaze_heatmap, image_path, gaze_inside,shifted_targets,gaze_final = next(iter(train_data_loader))

from torch.utils.tensorboard import SummaryWriter

# default `log_dir` is "runs" - we'll be more specific here
writer = SummaryWriter('runs/shashimal2_pretrained')

model_ft = train(model_ft,train_data_loader, criterion, optimizer, logger, writer,num_epochs=4)



