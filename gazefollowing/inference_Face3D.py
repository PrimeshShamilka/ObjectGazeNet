import sys
import torch
from torchvision import transforms
from torch.autograd import Variable
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

import time
import os
import numpy as np
import json
import cv2
from PIL import Image, ImageOps
import random
from tqdm import tqdm
import operator
import itertools
from scipy.io import  loadmat
import logging
from scipy import signal
from utils import data_transforms
from tqdm import tqdm

from utils import preprocess_image
import argparse
from models.primesh3 import Shashimal6_Face3D, Shashimal6_FaceDepth, Shashimal6_Face3D_Bias
from models.gazenet import GazeNet
from models.__init__ import save_checkpoint, resume_checkpoint
from dataloader.shashimal6 import GooDataset
from torch.utils.data import Dataset, DataLoader
from models.shashimal2 import Shashimal2
# from scipy.misc import imresize
import time
import matplotlib.pyplot as plt

def parse_inputs():

    # Command line arguments
    p = argparse.ArgumentParser()

    p.add_argument('--test_dir', type=str,
                        help='path to test set',
                        default=None)
    p.add_argument('--test_annotation', type=str,
                        help='test annotations (pickle/mat)',
                        default=None)
    p.add_argument('--resume_path', type=str,
                        help='load model file',
                        default=None)

    args = p.parse_args()

    return args


def boxes2centers(normalized_boxes):
    center_x = (normalized_boxes[:, 0] + normalized_boxes[:, 2]) / 2
    center_y = (normalized_boxes[:, 1] + normalized_boxes[:, 3]) / 2
    center_x = np.expand_dims(center_x, axis=1)
    center_y = np.expand_dims(center_y, axis=1)
    normalized_centers = np.hstack((center_x, center_y))

    return normalized_centers

def select_nearest_bbox(gazepoint, gt_bboxes, gt_labels=None):
    '''
    In: Accepts gazepoint (2,) and bboxes (n_boxes, 4), normalized from [0,1]
    Out: Returns the bbox nearest to gazepoint.
    '''

    centers = boxes2centers(gt_bboxes)

    diff = centers - gazepoint
    l2dist = np.sqrt(diff[:, 0] ** 2 + diff[:, 1] ** 2)
    min_idx = l2dist.argmin()

    if gt_labels is not None:
        predicted_class = gt_labels[min_idx]
    else:
        predicted_class = -1

    nearest_box = {
        'box': gt_bboxes[min_idx],
        'label': predicted_class,
        'index': min_idx
    }
    return nearest_box

def draw_results(image_path, eyepoint, gazepoint, idx, gtbox, gaze, bboxes=None):
    # Convert to numpy arrays jic users passed lists
    eyepoint, gazepoint, gazebox = map(np.array, [eyepoint, gazepoint, bboxes])
    eyepoint = eyepoint/224
    gazepoint = gazepoint/224
    gaze = gaze/224

    # Draw gazepoints and gt
    im = cv2.imread(image_path)
    im = cv2.resize(im, (640, 640))
    # im_original = im
    save_dir = './temp/new3/'
    # save_path_original = save_dir + str(idx) + '.png'
    # cv2.imwrite(save_path_original, im_original)
    # image_height, image_width = im.shape[:2]
    image_height, image_width = 640, 640

    x1, y1 = eyepoint
    x2, y2 = gazepoint
    xg, yg = gaze
    x1, y1 = image_width * x1, y1 * image_height
    x2, y2 = image_width * x2, y2 * image_height
    xg, yg = image_width * xg, yg * image_height
    x1, y1, x2, y2, xg, yg = map(int, [x1, y1, x2, y2, xg, yg])
    cv2.circle(im, (x1, y1), 1, [255, 255, 255], -1)
    cv2.circle(im, (x2, y2), 1, [255, 255, 255], -1)
    cv2.circle(im, (xg, yg), 1, [255, 255, 255], -1)
    cv2.line(im, (x1, y1), (x2, y2), [255, 0, 0], 2)
    cv2.line(im, (x1, y1), (xg, yg), [0, 0, 255], 2)

    if bboxes is not None:
        assert (len(bboxes.shape) == 2), 'gazebox must be numpy array of shape (N,4)'
        assert (bboxes.shape[1] == 4), 'gazebox must be numpy array of shape (N,4)'

        # Select nearest bbox given the gazepoint
        bbox_data = select_nearest_bbox(gazepoint, bboxes, gt_labels=None)
        nearest_bbox = bbox_data['box']

        # Scale to image size
        nearest_bbox = nearest_bbox * [image_width, image_height, image_width, image_height]
        nearest_bbox = nearest_bbox.astype(int)
        gtbox = gtbox * [image_width, image_height, image_width, image_height]
        gtbox = gtbox.astype(int)

        # Draw bbox of prediction (orage)
        cv2.rectangle(im, (nearest_bbox[0], nearest_bbox[1]), (nearest_bbox[2], nearest_bbox[3]), (0, 165, 255), 4)
        # Draw gtbbox (black)
        cv2.rectangle(im, (gtbox[0], gtbox[1]), (gtbox[2], gtbox[3]), (0, 0, 0), 2)

    img = im
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    filename = 'inference_%s.png' % str(idx)
    save_path = save_dir + filename
    cv2.imwrite(save_path, im)

    return None



def test_on_image(net, image, face):
    net.eval()
    outputs, id = net(image, face)
    f_point = np.array([outputs[0][0], outputs[0][1]])
    return f_point[0], f_point[1]

def main():
    args = parse_inputs()

    # Load Model
    # net = GazeNet()
    net = Shashimal6_Face3D()
    net.cuda()

    resume_path = "/media/primesh/F4D0EA80D0EA49061/PROJECTS/FYP/Gaze detection/saved_weights/shashimal6_Face3d_75_70_30_split.pt"
    net, optimizer, start_epoch = resume_checkpoint(net, None, resume_path)

    # Prepare dataloaders
    test_images_dir = "/media/primesh/F4D0EA80D0EA49061/PROJECTS/FYP/Gaze detection/Datasets/gooreal/finalrealdatasetImgsV2"
    test_pickle_path = "/media/primesh/F4D0EA80D0EA49061/PROJECTS/FYP/Gaze detection/Datasets/gooreal/testrealhumansNew.pickle"

    # For GOO
    val_set = GooDataset(test_images_dir, test_pickle_path, 'test')

    test_only = True
    if not test_only:
        test_and_save(net, test_data_loader)

    start_time = time.time()
    counter = 0
    # Get a random sample image from the dataset
    for i in range(100):
        idx = np.random.randint(len(val_set))
        print (idx)
        val_item = val_set.__getitem__(idx)
        image_path = val_item[8]
        # image_path = '/media/primesh/F4D0EA80D0EA4906/PROJECTS/FYP/Gaze detection/Datasets/gooreal/general images from internet/3.jpg'
        print(image_path)
#img, face, location_channel,object_channel,head_channel ,head,gt_label, head_box, gtbox
        # import sys
        # sys.exit()
        # img = val_item[0].unsqueeze(0).cuda().to(torch.device('cpu'))
        # head_channel = val_item[2].unsqueeze(0).cuda().to(torch.device('cpu'))
        # face = val_item[1].unsqueeze(0).cuda().to(torch.device('cpu'))
        # bboxes = val_item[9]
        # eyes = val_item[3]
        # image = val_item[0].numpy().transpose(1,2,0)
        # cv2.imwrite('./temp/new2/'+str(idx)+'.png', image)
        img = val_item[0].unsqueeze(0).cuda()
        face = val_item[1].unsqueeze(0).cuda()
        eyes_loc = val_item[5]
        gtbox = val_item[-1]
        gaze = val_item[6]
        # gtbox = np.expand_dims(gtbox, axis=0)

        x, y = test_on_image(net, img, face)
        # draw_results(image_path, eyes_loc, (x,y), idx, gtbox, gt_bboxes[:-1])
        x = x*224 + eyes_loc[0]
        y = y*224 + eyes_loc[1]
        draw_results(image_path, eyes_loc, (x,y), idx, gtbox, gaze, None)


        # original image
        img_new = cv2.imread(image_path)
        img_new = cv2.resize(img_new, (640, 640))
        img_new = cv2.cvtColor(img_new, cv2.COLOR_BGR2RGB)

        # save img
        # obj_channel = object_channel.squeeze()
        # objs = obj_channel.cpu().numpy()
        # im = img.squeeze().cpu().numpy().transpose(1,2,0)
        # fname = './temp/new3/' + str(idx) + '.png'
        # plt.imshow(img_new)
        # plt.savefig(fname)
        # plt.clf()

        # save object channel
        # plt.imshow(img_new)
        # objs = cv2.resize(objs, (640, 640))
        # plt.imshow(objs, alpha=0.5)
        # obj_fname = './temp/new3/' + str(idx) + '_obj_channel.png'
        # plt.savefig(obj_fname)
        # plt.clf()

        # save heatmap
        # plt.imshow(img_new)
        # heatmap_out = cv2.resize(heatmap, (640,640))
        # plt.imshow(heatmap_out, alpha=0.5)
        # # plt.imshow(heatmap_out)
        # heatmap_fname = './temp/new3/' + str(idx) + '_hm.png'
        # plt.savefig(heatmap_fname)

        # break
        # counter+=1
        # print ("DONE")

    print("FPS: ", round(counter/(time.time()-start_time), 2))

if __name__ == "__main__":
    main()