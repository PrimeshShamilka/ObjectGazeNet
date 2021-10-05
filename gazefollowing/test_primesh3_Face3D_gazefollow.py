import torch
from torchvision import transforms, models
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from utils_logging import setup_logger
from dataloader.shashimal6 import GazeDataset
from models.primesh3 import Shashimal6_Face3D, Shashimal6_FaceDepth, Shashimal6_Face3D_Bias
from training.train_primesh3 import train_face3d, GazeOptimizer, test_face3d, train_face_depth, test_face_depth, train_face3d_bias, train_face3d_gazefollow

logger = setup_logger(name='first_logger',
                      log_dir ='./logs/',
                      log_file='train_chong_gooreal.log',
                      log_format = '%(asctime)s %(levelname)s %(message)s',
                      verbose=True)

# Dataloaders for GAZE

batch_size = 4
workers = 12
testbatchsize = 16

images_dir = '/media/primesh/F4D0EA80D0EA4906/PROJECTS/FYP/Gaze detection/Datasets/gazefollow/data_new'
pickle_path = '/media/primesh/F4D0EA80D0EA4906/PROJECTS/FYP/Gaze detection/Datasets/gazefollow/data_new/train_annotations.mat'
test_images_dir = '/media/primesh/F4D0EA80D0EA4906/PROJECTS/FYP/Gaze detection/Datasets/gazefollow/data_new'
test_pickle_path = '/media/primesh/F4D0EA80D0EA4906/PROJECTS/FYP/Gaze detection/Datasets/gazefollow/data_new/test_annotations.mat'

train_set = GazeDataset(images_dir, pickle_path, 'train')
train_data_loader = DataLoader(dataset=train_set,
                               batch_size=batch_size,
                               shuffle=True,
                               num_workers=16)

test_set = GazeDataset(test_images_dir, test_pickle_path, 'train')
test_data_loader = DataLoader(test_set, batch_size=batch_size // 2,
                              shuffle=False, num_workers=8)

# Load model for training
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# model_ft2 = Shashimal6_Face3D()
model_ft2 = Shashimal6_Face3D()
model_ft2 = model_ft2.to(device)
# criterion = nn.NLLLoss().cuda()
# criterion = nn.BCELoss().cuda()
criterion = nn.MSELoss()
# criterion

# Observe that all parameters are being optimized
start_epoch = 0
max_epoch = 5
learning_rate = 1e-4

# Initializes Optimizer
gaze_opt = GazeOptimizer(model_ft2, learning_rate)
optimizer = gaze_opt.getOptimizer(start_epoch)

from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter('runs/face_3d')
train_face3d_gazefollow(model_ft2, train_data_loader, test_data_loader, criterion, optimizer, logger, writer, num_epochs=50, patience=10)
