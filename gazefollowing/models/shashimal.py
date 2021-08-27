import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, PackedSequence
import math
from lib.pytorch_convolutional_rnn import convolutional_rnn
import numpy as np

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out



class Shashimal(nn.Module):
    def __init__(self,block = Bottleneck, layers_scene = [3, 4, 6, 3, 2], layers_face = [3, 4, 6, 3, 2]):
        
        self.inplanes_scene = 64
        self.inplanes_face = 64
        super(Shashimal,self).__init__()
        # common
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        # scene pathway
        self.conv1_scene = nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1_scene = nn.BatchNorm2d(64)
        self.layer1_scene = self._make_layer_scene(block, 64, layers_scene[0])
        self.layer2_scene = self._make_layer_scene(block, 128, layers_scene[1], stride=2)
        self.layer3_scene = self._make_layer_scene(block, 256, layers_scene[2], stride=2)
        self.layer4_scene = self._make_layer_scene(block, 512, layers_scene[3], stride=2)
        self.layer5_scene = self._make_layer_scene(block, 256, layers_scene[4], stride=1) # additional to resnet50

        # face pathway
        self.conv1_face = nn.Conv2d(3, 64, kernel_size = 7, stride = 2, padding = 3, bias = False)
        self.bn1_face = nn.BatchNorm2d(64)
        self.layer1_face = self._make_layer_face(block, 64, layers_face[0])
        self.layer2_face = self._make_layer_face(block, 128, layers_face[1], stride=2)
        self.layer3_face = self._make_layer_face(block, 256, layers_face[2], stride=2)
        self.layer4_face = self._make_layer_face(block, 512, layers_face[3], stride=2)
        self.layer5_face = self._make_layer_face(block, 256, layers_face[4], stride=1) # additional to resnet50


        self.compress_conv1 = nn.Conv2d(2048, 1024, kernel_size=1, stride=1, padding=0, bias=False)
        self.compress_bn1 = nn.BatchNorm2d(1024)
        self.compress_conv2 = nn.Conv2d(1024, 512, kernel_size=1, stride=1, padding=0, bias=False)
        self.compress_bn2 = nn.BatchNorm2d(512)
        self.compress_conv3 = nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0, bias=False)
        self.compress_bn3 = nn.BatchNorm2d(256)


        self.attn = nn.Linear(1808, 1*7*7)
        self.fc1 = nn.Linear(256*7*7, 669)
        self.fc2 = nn.Linear(669, 400)
        self.fc3 = nn.Linear(400, 200)
        self.fc4 = nn.Linear(200, 169)

        self.smax = nn.LogSoftmax(dim=1)
        self.nolog_smax = nn.Softmax(dim=1)

        self.fc_0_0 = nn.Linear(169, 25)
        self.fc_0_m1 = nn.Linear(169, 25)
        self.fc_0_1 = nn.Linear(169, 25)
        self.fc_m1_0 = nn.Linear(169, 25)
        self.fc_1_0 = nn.Linear(169, 25)
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        

    def forward(self,images,head,face):
        face = self.conv1_face(face)
        face = self.bn1_face(face)
        face = self.relu(face)
        face = self.maxpool(face)
        face = self.layer1_face(face)
        face = self.layer2_face(face)
        face = self.layer3_face(face)
        face = self.layer4_face(face)
        face_feat = self.layer5_face(face)
        
        # reduce head channel size by max pooling: (N, 1, 224, 224) -> (N, 1, 28, 28)
        head_reduced = self.maxpool(self.maxpool(self.maxpool(head))).view(-1, 784)
        # reduce face feature size by avg pooling: (N, 1024, 7, 7) -> (N, 1024, 1, 1)
        face_feat_reduced = self.avgpool(face_feat).view(-1, 1024)
        # get and reshape attention weights such that it can be multiplied with scene feature map

        attn_weights = self.attn(torch.cat((head_reduced, face_feat_reduced), 1))
        attn_weights = attn_weights.view(-1, 1, 49)
        attn_weights = F.softmax(attn_weights, dim=2) # soft attention weights single-channel
        attn_weights = attn_weights.view(-1, 1, 7, 7)

        im = torch.cat((images, head), dim=1)
        im = self.conv1_scene(im)
        im = self.bn1_scene(im)
        im = self.relu(im)
        im = self.maxpool(im)
        im = self.layer1_scene(im)

        im = self.layer2_scene(im)
        im = self.layer3_scene(im)
        im = self.layer4_scene(im)
        scene_feat = self.layer5_scene(im)
        # attn_weights = torch.ones(attn_weights.shape)/49.0
        attn_applied_scene_feat = torch.mul(attn_weights, scene_feat) # (N, 1, 7, 7) # applying attention weights on scene feat

        scene_face_feat = torch.cat((attn_applied_scene_feat, face_feat), 1)

        encoding = self.compress_conv1(scene_face_feat)
        encoding = self.compress_bn1(encoding)
        encoding = self.relu(encoding)
        encoding = self.compress_conv2(encoding)
        encoding = self.compress_bn2(encoding)
        encoding = self.relu(encoding)
        encoding = self.compress_conv3(encoding)
        encoding = self.compress_bn3(encoding)
        encoding = self.relu(encoding)
        print(encoding.shape)
        encoding = encoding.view(-1, 256 * 7 * 7)

        fc = self.relu(self.fc1(encoding))
        fc =  self.relu(self.fc2(fc))
        fc =  self.relu(self.fc3(fc))
        output =  self.sigmoid(self.fc4(fc))
        print(output.shape)
        
        out_0_0 = self.smax(self.fc_0_0(output))
        out_1_0 = self.smax(self.fc_1_0(output))
        out_m1_0 = self.smax(self.fc_m1_0(output))
        out_0_m1 = self.smax(self.fc_0_m1(output))
        out_0_1 = self.smax(self.fc_0_1(output))
        return [out_0_0, out_1_0, out_m1_0, out_0_m1, out_0_1]

    def predict(self,images,head,face):
        face = self.conv1_face(face)
        face = self.bn1_face(face)
        face = self.relu(face)
        face = self.maxpool(face)
        face = self.layer1_face(face)
        face = self.layer2_face(face)
        face = self.layer3_face(face)
        face = self.layer4_face(face)
        face_feat = self.layer5_face(face)

        # reduce head channel size by max pooling: (N, 1, 224, 224) -> (N, 1, 28, 28)
        head_reduced = self.maxpool(self.maxpool(self.maxpool(head))).view(-1, 784)
        # reduce face feature size by avg pooling: (N, 1024, 7, 7) -> (N, 1024, 1, 1)
        face_feat_reduced = self.avgpool(face_feat).view(-1, 1024)
        # get and reshape attention weights such that it can be multiplied with scene feature map
        attn_weights = self.attn(torch.cat((head_reduced, face_feat_reduced), 1))
        attn_weights = attn_weights.view(-1, 1, 49)
        attn_weights = F.softmax(attn_weights, dim=2) # soft attention weights single-channel
        attn_weights = attn_weights.view(-1, 1, 7, 7)

        im = torch.cat((images, head), dim=1)
        im = self.conv1_scene(im)
        im = self.bn1_scene(im)
        im = self.relu(im)
        im = self.maxpool(im)
        im = self.layer1_scene(im)
        im = self.layer2_scene(im)
        im = self.layer3_scene(im)
        im = self.layer4_scene(im)
        scene_feat = self.layer5_scene(im)
        # attn_weights = torch.ones(attn_weights.shape)/49.0
        attn_applied_scene_feat = torch.mul(attn_weights, scene_feat) # (N, 1, 7, 7) # applying attention weights on scene feat

        scene_face_feat = torch.cat((attn_applied_scene_feat, face_feat), 1)
        
        encoding = self.compress_conv1(scene_face_feat)
        encoding = self.compress_bn1(encoding)
        encoding = self.relu(encoding)
        encoding = self.compress_conv2(encoding)
        encoding = self.compress_bn2(encoding)
        encoding = self.relu(encoding)
        encoding = self.compress_conv3(encoding)
        encoding = self.compress_bn3(encoding)
        encoding = self.relu(encoding)

        encoding = encoding.view(-1, 256 * 7 * 7)

        fc = self.relu(self.fc1(encoding))
        fc =  self.relu(self.fc2(fc))
        fc =  self.relu(self.fc3(fc))
        output =  self.sigmoid(self.fc4(fc))
        
        hm = torch.zeros(output.size(0), 15, 15).cuda()
        count_hm = torch.zeros(output.size(0), 15, 15).cuda()

        f_0_0 = self.nolog_smax(self.fc_0_0(output)).view(-1, 5, 5)
        f_1_0 = self.nolog_smax(self.fc_1_0(output)).view(-1, 5, 5)
        f_m1_0 = self.nolog_smax(self.fc_m1_0(output)).view(-1, 5, 5)
        f_0_m1 = self.nolog_smax(self.fc_0_m1(output)).view(-1, 5, 5)
        f_0_1 = self.nolog_smax(self.fc_0_1(output)).view(-1, 5, 5)

        f_cell = []
        f_cell.extend([f_0_m1, f_0_1, f_m1_0, f_1_0, f_0_0])

        v_x = [0, 1, -1, 0, 0]
        v_y = [0, 0, 0, -1, 1]

        for k in range(5):
            dx, dy = v_x[k], v_y[k]
            f = f_cell[k]
            for x in range(5):
                for y in range(5):

                    i_x = 3*x - dx
                    i_x = max(i_x, 0)
                    if x == 0:
                        i_x = 0

                    i_y = 3*y - dy
                    i_y = max(i_y, 0)
                    if y == 0:
                        i_y = 0

                    f_x = 3*x + 2 - dx
                    f_x = min(14, f_x)
                    if x == 4:
                        f_x = 14

                    f_y = 3*y + 2 - dy
                    f_y = min(14, f_y)
                    if y == 4:
                        f_y = 14

                    a = f[:, x, y].contiguous()
                    a = a.view(output.size(0), 1, 1)

                    hm[:, i_x: f_x+1, i_y: f_y+1] =  hm[:, i_x: f_x+1, i_y: f_y+1] + a
                    count_hm[:, i_x: f_x+1, i_y: f_y+1] = count_hm[:, i_x: f_x+1, i_y: f_y+1] + 1

        hm_base = hm.div(count_hm)

        hm_base = hm_base.unsqueeze(1)

        hm_base = F.interpolate(input = hm_base, size = (227, 227), mode='bicubic', align_corners=False)

        hm_base = hm_base.squeeze(1)


        #modeltester works with this return:
        #return hm_base

        #main.py /training must use this
        return hm_base.view(-1, 227 * 227)


    def raw_hm(self,images,head,face):
        face = self.conv1_face(face)
        face = self.bn1_face(face)
        face = self.relu(face)
        face = self.maxpool(face)
        face = self.layer1_face(face)
        face = self.layer2_face(face)
        face = self.layer3_face(face)
        face = self.layer4_face(face)
        face_feat = self.layer5_face(face)

        # reduce head channel size by max pooling: (N, 1, 224, 224) -> (N, 1, 28, 28)
        head_reduced = self.maxpool(self.maxpool(self.maxpool(head))).view(-1, 784)
        # reduce face feature size by avg pooling: (N, 1024, 7, 7) -> (N, 1024, 1, 1)
        face_feat_reduced = self.avgpool(face_feat).view(-1, 1024)
        # get and reshape attention weights such that it can be multiplied with scene feature map
        attn_weights = self.attn(torch.cat((head_reduced, face_feat_reduced), 1))
        attn_weights = attn_weights.view(-1, 1, 49)
        attn_weights = F.softmax(attn_weights, dim=2) # soft attention weights single-channel
        attn_weights = attn_weights.view(-1, 1, 7, 7)

        im = torch.cat((images, head), dim=1)
        im = self.conv1_scene(im)
        im = self.bn1_scene(im)
        im = self.relu(im)
        im = self.maxpool(im)
        im = self.layer1_scene(im)
        im = self.layer2_scene(im)
        im = self.layer3_scene(im)
        im = self.layer4_scene(im)
        scene_feat = self.layer5_scene(im)
        # attn_weights = torch.ones(attn_weights.shape)/49.0
        attn_applied_scene_feat = torch.mul(attn_weights, scene_feat) # (N, 1, 7, 7) # applying attention weights on scene feat

        scene_face_feat = torch.cat((attn_applied_scene_feat, face_feat), 1)
        
        encoding = self.compress_conv1(scene_face_feat)
        encoding = self.compress_bn1(encoding)
        encoding = self.relu(encoding)
        encoding = self.compress_conv2(encoding)
        encoding = self.compress_bn2(encoding)
        encoding = self.relu(encoding)
        encoding = self.compress_conv3(encoding)
        encoding = self.compress_bn3(encoding)
        encoding = self.relu(encoding)

        encoding = encoding.view(-1, 256 * 7 * 7)

        fc = self.relu(self.fc1(encoding))
        fc =  self.relu(self.fc2(fc))
        fc =  self.relu(self.fc3(fc))
        output =  self.sigmoid(self.fc4(fc))
        
        hm = torch.zeros(output.size(0), 15, 15).cuda()
        count_hm = torch.zeros(output.size(0), 15, 15).cuda()

        f_0_0 = self.nolog_smax(self.fc_0_0(output)).view(-1, 5, 5)
        f_1_0 = self.nolog_smax(self.fc_1_0(output)).view(-1, 5, 5)
        f_m1_0 = self.nolog_smax(self.fc_m1_0(output)).view(-1, 5, 5)
        f_0_m1 = self.nolog_smax(self.fc_0_m1(output)).view(-1, 5, 5)
        f_0_1 = self.nolog_smax(self.fc_0_1(output)).view(-1, 5, 5)

        f_cell = []
        f_cell.extend([f_0_m1, f_0_1, f_m1_0, f_1_0, f_0_0])

        v_x = [0, 1, -1, 0, 0]
        v_y = [0, 0, 0, -1, 1]

        for k in range(5):
            dx, dy = v_x[k], v_y[k]
            f = f_cell[k]
            for x in range(5):
                for y in range(5):

                    i_x = 3*x - dx
                    i_x = max(i_x, 0)
                    if x == 0:
                        i_x = 0

                    i_y = 3*y - dy
                    i_y = max(i_y, 0)
                    if y == 0:
                        i_y = 0

                    f_x = 3*x + 2 - dx
                    f_x = min(14, f_x)
                    if x == 4:
                        f_x = 14

                    f_y = 3*y + 2 - dy
                    f_y = min(14, f_y)
                    if y == 4:
                        f_y = 14

                    a = f[:, x, y].contiguous()
                    a = a.view(output.size(0), 1, 1)

                    hm[:, i_x: f_x+1, i_y: f_y+1] =  hm[:, i_x: f_x+1, i_y: f_y+1] + a
                    count_hm[:, i_x: f_x+1, i_y: f_y+1] = count_hm[:, i_x: f_x+1, i_y: f_y+1] + 1

        hm_base = hm.div(count_hm)
        raw_hm = hm_base
        hm_base = hm_base.unsqueeze(1)

        hm_base = F.interpolate(input = hm_base, size = (227, 227), mode='bicubic', align_corners=False)

        hm_base = hm_base.squeeze(1)


        #modeltester works with this return:
        #return hm_base

        #main.py /training must use this
        return hm_base.view(-1, 227 * 227), raw_hm




    def _make_layer_scene(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes_scene != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes_scene, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes_scene, planes, stride, downsample))
        self.inplanes_scene = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes_scene, planes))

        return nn.Sequential(*layers)

    def _make_layer_face(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes_face != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes_face, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes_face, planes, stride, downsample))
        self.inplanes_face = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes_face, planes))

        return nn.Sequential(*layers)


