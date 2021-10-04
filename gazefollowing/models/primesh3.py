import torch
import torch.nn as nn
import math
import numpy as np
from models.resnet_modified import resnet50
from resnest.torch import resnest50
import matplotlib.pyplot as plt

class Gaze360Static(nn.Module):
    def __init__(self):
        super(Gaze360Static, self).__init__()
        self.img_feature_dim = 256
        self.base_model = resnet50(pretrained=True)
        self.base_model.fc2 = nn.Linear(1000, self.img_feature_dim)
        # self.last_layer = nn.Linear(self.img_feature_dim, 3)

    def forward(self, x_in):
        face = x_in["face"]
        base_out = self.base_model(face)
        base_out = torch.flatten(base_out, start_dim=1)
        # output = self.last_layer(base_out)
        # angular_output = output[:,:2]
        # angular_output[:,0:1] = math.pi*nn.Tanh()(angular_output[:,0:1])
        # angular_output[:,1:2] = (math.pi/2)*nn.Tanh()(angular_output[:,1:2])
        # var = math.pi*nn.Sigmoid()(output[:,2:3])
        # var = var.view(-1,1).expand(var.size(0), 2)
        # return angular_output,var
        return base_out

class PinBallLoss(nn.Module):
    def __init__(self):
        super(PinBallLoss, self).__init__()
        self.q1 = 0.1
        self.q9 = 1-self.q1

    def forward(self, output_o, target_o, var_o):
        q_10 = target_o-(output_o-var_o)
        q_90 = target_o-(output_o+var_o)
        loss_10 = torch.max(self.q1*q_10, (self.q1-1)*q_10)
        loss_90 = torch.max(self.q9*q_90, (self.q9-1)*q_90)
        loss_10 = torch.mean(loss_10)
        loss_90 = torch.mean(loss_90)

        return loss_10+loss_90


class CameraCalibrator(nn.Module):
    def __init__(self):
        super(CameraCalibrator, self).__init__()
        self.img_feature_dim = 256
        self.depth = torch.hub.load("intel-isl/MiDaS", "DPT_Hybrid")
        self.gaze3d_net = Gaze360Static()
        statedict = torch.load("/media/primesh/F4D0EA80D0EA4906/PROJECTS/FYP/Gaze detection/saved_weight/gaze360_static_gaze360_1.pt")
        self.gaze3d_net.cuda()
        self.gaze3d_net.load_state_dict(statedict)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()
        self.fc1 = nn.Linear(self.img_feature_dim, 128)
        self.fc2 = nn.Linear(128, 56)
        self.fc3 = nn.Linear(56, 3)

    def forward(self, image, face):
        self.gaze3d_net.eval()
        self.depth.eval()
        with torch.no_grad():
            base_out = self.gaze3d_net(face) # 3D world coord feature
        with torch.no_grad():
            image_depth = self.depth(image) # monocular depth
            image_depth = torch.nn.functional.interpolate(image_depth.unsqueeze(1),size=image.shape[2:],mode="bicubic",align_corners=False,)
        output = self.relu(self.fc1(base_out))
        output = self.relu(self.fc2(output))
        output = self.sigmoid(self.fc3(output)) # x,y,z in image coords
        return output, image_depth

class Shashimal6_Face3D(nn.Module):
    def __init__(self):
        super(Shashimal6_Face3D, self).__init__()
        self.depth = torch.hub.load("intel-isl/MiDaS", "DPT_Large")
        self.img_feature_dim = 256  # the dimension of the CNN feature to represent each frame
        self.base_model = resnet50(pretrained=True)
        self.base_model.fc2 = nn.Linear(1000, self.img_feature_dim)
        self.last_layer = nn.Linear(self.img_feature_dim, 3)
        self.tanh = nn.Tanh()

    def forward(self, image, face):
        self.depth.eval()
        with torch.no_grad():
            id = self.depth(image)
            id = torch.nn.functional.interpolate(id.unsqueeze(1),size=image.shape[2:],mode="bicubic",align_corners=False,)
        base_out = self.base_model(face)
        base_out = torch.flatten(base_out, start_dim=1)
        output = self.last_layer(base_out)
        return output,id

class Shashimal6_FaceDepth(nn.Module):
    def __init__(self):
        super(Shashimal6_FaceDepth, self).__init__()
        self.model_type = "DPT_Large"
        self.depth = torch.hub.load("intel-isl/MiDaS", self.model_type)
        self.midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
        if self.model_type == "DPT_Large" or self.model_type == "DPT_Hybrid":
            self.transform = self.midas_transforms.dpt_transform
        else:
            self.transform = self.midas_transforms.small_transform
        self.img_feature_dim = 256  # the dimension of the CNN feature to represent each frame
        self.base_model = resnet50(pretrained=True)
        self.base_model.fc2 = nn.Linear(1000, self.img_feature_dim)
        self.last_layer = nn.Linear(self.img_feature_dim, 1)
        # self.fc1 = nn.Linear(self.img_feature_dim, 128)
        # self.fc2 = nn.Linear(128, 56)
        # self.last_layer = nn.Linear(56, 1)
        # self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, image, face):
        self.depth.eval()
        with torch.no_grad():
            # image = self.transform(image)
            id = self.depth(image)
            id = torch.nn.functional.interpolate(id.unsqueeze(1),size=image.shape[2:],mode="bicubic",align_corners=False,)
        base_out = self.base_model(face)
        base_out = torch.flatten(base_out, start_dim=1)
        # base_out = self.relu(self.fc1(base_out))
        # base_out = self.relu(self.fc2(base_out))
        # output = self.sigmoid(self.last_layer(base_out))
        output = self.last_layer(base_out)
        return output,id

'''
Dual Attention Module
'''
class DualAttention(nn.Module):
    def __init__(self):
        super(DualAttention, self).__init__()
        # self.depth = torch.hub.load("intel-isl/MiDaS", "DPT_Hybrid")
        self.face_net = Shashimal6_Face3D()
        statedict = torch.load("/media/primesh/F4D0EA80D0EA4906/PROJECTS/FYP/Gaze detection/saved_weights/shashimal6_face_43.pt")
        self.face_net.cuda()
        self.face_net.load_state_dict(statedict["state_dict"])

    def forward(self, image, face, object_channel, head_point):
        self.face_net.eval()
        with torch.no_grad():
            gaze, depth = self.face_net(image, face)
            depth = torch.mul(depth,object_channel)
            fd_range = torch.zeros(image.shape[0],1).cuda()
            head_depth = torch.zeros(image.shape[0],1).cuda()
            for batch in range(image.shape[0]):
                fd_range[batch,:] = (torch.max(depth[batch]) - torch.min(depth[batch]))/24
                head_depth[batch,:] = depth[batch,:,head_point[batch,0],head_point[batch,1]]
            point_depth = torch.zeros(image.shape[0], 1).cuda()
            for batch in range(image.shape[0]):
                point_depth[batch, :] = head_depth[batch] + gaze[batch, 2] * 224
                # point_depth[batch, :] = head_depth[batch] + gaze[batch, 2] * torch.max(depth)
            fd_0 = torch.zeros(image.shape[0],1,224,224).cuda()
            fd_1 = torch.zeros(image.shape[0],1,224,224).cuda()
            fd_2 = torch.zeros(image.shape[0],1,224,224).cuda()
            for batch in range(image.shape[0]):
                fd_0[batch,:,:,:] = torch.where((point_depth[batch]-fd_range[batch]<=depth[batch,:,:,:]) & (point_depth[batch]+fd_range[batch]>=depth[batch,:,:,:]),depth[batch,:,:,:],torch.tensor(0,dtype=torch.float).cuda())
                fd_1[batch,:,:,:] = torch.where((point_depth[batch]-2*fd_range[batch]<=depth[batch,:,:,:]) & (point_depth[batch]+2*fd_range[batch]>=depth[batch,:,:,:]),depth[batch,:,:,:],torch.tensor(0,dtype=torch.float).cuda())
                fd_2[batch,:,:,:] = torch.where((point_depth[batch]-3*fd_range[batch]<=depth[batch,:,:,:]) & (point_depth[batch]+3*fd_range[batch]>=depth[batch,:,:,:]),depth[batch,:,:,:],torch.tensor(0,dtype=torch.float).cuda())
            xy = gaze[:,:2]
            mask = torch.zeros(image.shape[0],1,224,224).cuda()
            for batch in range(image.shape[0]):
                for i in range(224):
                    for k in range(224):
                        arr = torch.tensor([k,i],dtype=torch.float32).cuda() - head_point[batch,:]
                        mask[batch,:,i,k] = torch.dot(arr,xy[batch,:])/(torch.norm(arr,p=2)*torch.norm(xy[batch,:],p=2))
            mask = torch.arccos(mask)
            mask = torch.maximum(1-(12*mask/np.pi),torch.tensor(0))
            mask = torch.nan_to_num(mask)
            x_0 = torch.mul(fd_0,mask)
            x_1 = torch.mul(fd_1,mask)
            x_2 = torch.mul(fd_2,mask)
            # x = torch.cat([image,x_0,x_1,x_2], dim=1)
        # return x

'''
Heatmap Generation Net
'''
class Primesh3(nn.Module):
    def __init__(self):
        super(Primesh3, self).__init__()
        self.inplanes_scene = 64
        self.inplanes_face = 64
        self.relu = nn.ReLU(inplace=True)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.avgpool = nn.AvgPool2d(7, stride=1)

        # encoding for saliency
        self.compress_conv0 = nn.Conv2d(4096, 2048, kernel_size=1, stride=1, padding=0, bias=False)
        self.compress_bn0 = nn.BatchNorm2d(2048)
        self.compress_conv1 = nn.Conv2d(2048, 1024, kernel_size=1, stride=1, padding=0, bias=False)
        self.compress_bn1 = nn.BatchNorm2d(1024)
        self.compress_conv2 = nn.Conv2d(1024, 512, kernel_size=1, stride=1, padding=0, bias=False)
        self.compress_bn2 = nn.BatchNorm2d(512)
        self.compress_conv3 = nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0, bias=False)
        self.compress_bn3 = nn.BatchNorm2d(256)

        # decoding
        self.deconv1 = nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2)
        self.deconv_bn1 = nn.BatchNorm2d(256)
        self.deconv2 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2)
        self.deconv_bn2 = nn.BatchNorm2d(128)
        self.deconv3 = nn.ConvTranspose2d(128, 1, kernel_size=4, stride=2)
        self.deconv_bn3 = nn.BatchNorm2d(1)
        self.conv4 = nn.Conv2d(1, 1, kernel_size=1, stride=1)

        count = 0
        # Initialize weights
        for m in self.modules():
            count += 1
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        model = resnet50(pretrained=True)
        self.scence_net = nn.Sequential(*(list(model.children())[:-2]))
        self.dual_attention_net = nn.Sequential(*(list(model.children())[:-2]))

    def forward(self, image, dual_attn):
        dual_attn_feat = self.dual_attention_net(dual_attn)
        # dual_attn_feat_reduced = self.avgpool(dual_attn_feat).view(-1, 2048)
        scene_feat = self.scence_net(image)
        scene_dual_feat = torch.cat(scene_feat, dual_attn_feat)
        # encoding
        encoding = self.compress_conv0(scene_dual_feat)
        encoding = self.compress_bn0(encoding)
        encoding = self.relu(encoding)
        encoding = self.compress_conv1(encoding)
        encoding = self.compress_bn1(encoding)
        encoding = self.relu(encoding)
        encoding = self.compress_conv2(encoding)
        encoding = self.compress_bn2(encoding)
        encoding = self.relu(encoding)
        # decoding
        x = self.deconv1(encoding)
        x = self.deconv_bn1(x)
        x = self.relu(x)
        x = self.deconv2(x)
        x = self.deconv_bn2(x)
        x = self.relu(x)
        x = self.deconv3(x)
        x = self.deconv_bn3(x)
        x = self.relu(x)
        heatmap = self.conv4(x)
        return heatmap
