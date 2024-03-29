import torch
from torchvision import transforms, models
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np

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
from models.primesh3 import DualAttention
from models.primesh2 import Primesh2
from models.__init__ import save_checkpoint, resume_checkpoint
from models.primesh3 import Shashimal6_Face3D, Shashimal6_FaceDepth, Shashimal6_Face3D_Bias
# from dataloader.shashimal2_synth import GooDataset
# from dataloader.primesh import GooDataset
from dataloader.shashimal6 import GooDataset
from dataloader import chong_imutils
# from training.train_primesh import train, GazeOptimizer, train_with_early_stopping, test_gop, test
from training.train_primesh3 import train_face3d, GazeOptimizer, test_face3d, train_face_depth, test_face_depth, train_face3d_bias, test_face3d_prediction
from torch.nn.utils.rnn import pad_sequence

logger = setup_logger(name='first_logger',
                      log_dir ='./logs/',
                      log_file='train_chong_gooreal.log',
                      log_format = '%(asctime)s %(levelname)s %(message)s',
                      verbose=True)

# Dataloaders for GOO-Real
batch_size=4
workers=12
images_dir = '/media/primesh/F4D0EA80D0EA4906/PROJECTS/FYP/Gaze detection/Datasets/gooreal/finalrealdatasetImgsV2'
pickle_path = '/media/primesh/F4D0EA80D0EA4906/PROJECTS/FYP/Gaze detection/Datasets/gooreal/oneshotrealhumansNew2.pickle'
test_images_dir = '/media/primesh/F4D0EA80D0EA4906/PROJECTS/FYP/Gaze detection/Datasets/gooreal/finalrealdatasetImgsV2'
test_pickle_path = '/media/primesh/F4D0EA80D0EA4906/PROJECTS/FYP/Gaze detection/Datasets/gooreal/testrealhumansNew2.pickle'
val_images_dir = '/media/primesh/F4D0EA80D0EA4906/PROJECTS/FYP/Gaze detection/Datasets/gooreal/finalrealdatasetImgsV2'
val_pickle_path = '/media/primesh/F4D0EA80D0EA4906/PROJECTS/FYP/Gaze detection/Datasets/gooreal/valrealhumansNew2.pickle'


def pad_x_collate_function(batch):
    # batch looks like [(x0,y0), (x4,y4), (x2,y2)... ]
    img, face, head, gt_label, centers, gaze_idx, gt_bboxes, gt_labels = zip(*batch)

    # If you want to be a little fancy, you can do the above in one line
    # xs, ys = zip(*samples)
    img = pad_sequence(img, batch_first=True, padding_value=0)
    face = pad_sequence(face, batch_first=True, padding_value=0)
    #eye = pad_sequence(eye, batch_first=True, padding_value=0)
    #gaze = pad_sequence(gaze, batch_first=True, padding_value=0)
    #gtbox = pad_sequence(gtbox, batch_first=True, padding_value=0)
    #gt_bboxes = pad_sequence(gt_bboxes, batch_first=True, padding_value=0)
    return img, face, head, gt_label, centers, gaze_idx, zip(*gt_bboxes), zip(*gt_labels)


print ('Train')
train_set = GooDataset(images_dir, pickle_path, 'train')
train_data_loader = DataLoader(dataset=train_set,
                                           batch_size=batch_size,
                                           shuffle=True,
                                           num_workers=16)
print ('Val')
val_set = GooDataset(val_images_dir, val_pickle_path, 'train')
val_data_loader = DataLoader(dataset=val_set,
                                           batch_size=4,
                                           shuffle=True,
                                           num_workers=16)
print ('Test')
test_set = GooDataset(test_images_dir, test_pickle_path, 'test')
test_data_loader = DataLoader(test_set, batch_size=batch_size//2,
                            shuffle=False, num_workers=8)


print ('Test')
test_pred_set = GooDataset(test_images_dir, test_pickle_path, 'test_prediction')
test_pred_data_loader = DataLoader(test_pred_set, batch_size=1,
                            shuffle=False, num_workers=8, collate_fn=pad_x_collate_function)

# Load model for inference
# print ("Model loading")
# model = DualAttention()
# model.cuda()
#
# img, face, location_channel,object_channel,head_channel ,head,gt_label,heatmap = next(iter(train_data_loader))
# image = img.cuda()
# face = face.cuda()
# object_channel = object_channel.cuda()
# head_point = head.cuda()
#
# model.eval()
# model(image, face, object_channel, head_point)



# Load model for training
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# model_ft2 = Shashimal6_Face3D()
# model_ft2 = Shashimal6_Face3D_Bias()
# model_ft2 = model_ft2.to(device)
# criterion = nn.NLLLoss().cuda()
# criterion = nn.BCELoss().cuda()
# criterion = nn.MSELoss()
# criterion

# Load model for test prediction
model = Shashimal6_Face3D().cuda()
checkpoint_fpath = '/media/primesh/F4D0EA80D0EA4906/PROJECTS/FYP/Gaze detection/saved_weights/shashimal6_Face3d_75_70_30_split.pt'
checkpoint = torch.load(checkpoint_fpath)
model.load_state_dict(checkpoint['state_dict'])

# model2 = Shashimal6_Face3D().cuda()
# checkpoint_fpath = '/media/primesh/F4D0EA80D0EA4906/PROJECTS/FYP/Gaze detection/saved_weights/shashimal6_face_43.pt'
# checkpoint = torch.load(checkpoint_fpath)
# model2.load_state_dict(checkpoint['state_dict'])

# Observe that all parameters are being optimized
start_epoch = 0
max_epoch = 5
learning_rate = 1e-4

# Initializes Optimizer
# gaze_opt = GazeOptimizer(model_ft2, learning_rate)
# optimizer = gaze_opt.getOptimizer(start_epoch)

if False:
    checkpoint_fpath = '/media/primesh/F4D0EA80D0EA4906/PROJECTS/FYP/Gaze detection/saved_weights/primesh_gazefollow_26_gooreal_3.pt'
    checkpoint = torch.load(checkpoint_fpath)
    model_ft2.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])

from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter('runs/face_3d')

# train_face3d(model_ft2, train_data_loader, val_data_loader, criterion, optimizer, logger, writer, num_epochs=50, patience=10)
# test_face3d(model_ft2, test_data_loader, logger, test_depth=False)
# train_face_depth(model_ft2, train_data_loader, val_data_loader, criterion, optimizer, logger, writer, num_epochs=50, patience=10)
# test_face_depth(model_ft2, test_data_loader, logger)
# train_face3d_bias(model_ft2, train_data_loader, val_data_loader, criterion, optimizer, logger, writer, num_epochs=50, patience=10)
test_face3d_prediction(model, test_pred_data_loader, logger)