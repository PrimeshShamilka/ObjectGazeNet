import time
import torch
import torch.optim as optim
import numpy as np
from early_stopping_pytorch.pytorchtools import EarlyStopping
from tqdm import tqdm
import torch.nn as nn
import warnings

# warnings.filterwarnings('error')

class GazeOptimizer():
    def __init__(self, net, initial_lr, weight_decay=1e-6):

        self.INIT_LR = initial_lr
        self.WEIGHT_DECAY = weight_decay
        self.optimizer = optim.Adam(net.parameters(), lr=initial_lr, weight_decay=weight_decay)

    def getOptimizer(self, epoch, decay_epoch=15):

        if epoch < decay_epoch:
            lr = self.INIT_LR
        else:
            lr = self.INIT_LR / 10

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
            param_group['weight_decay'] = self.WEIGHT_DECAY

        return self.optimizer

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


def get_bb_binary(box):
    xmin, ymin, xmax, ymax = box
    b = np.zeros((224, 224), dtype='float32')
    for j in range(ymin, ymax):
        for k in range(xmin, xmax):
            b[j][k] = 1
    return b

def train_face3d(model,train_data_loader,validation_data_loader, criterion, optimizer, logger, writer ,num_epochs=5,patience=10):
    since = time.time()
    n_total_steps = len(train_data_loader)
    n_total_steps_val = len(validation_data_loader)
    early_stopping = EarlyStopping(patience=patience, verbose=True)
    for epoch in range(num_epochs):

        model.train()  # Set model to training mode

        running_loss = []
        validation_loss = []
        for i, (img, face, location_channel,object_channel,head_channel ,head,gt_label, head_box, gtbox) in tqdm(enumerate(train_data_loader), total=len(train_data_loader)) :
            image =  img.cuda()
            face = face.cuda()
            gt_label = gt_label
            head = head
            optimizer.zero_grad()
            gaze,depth = model(image,face)
            depth =  depth.cpu()
            max_depth = torch.max(depth)
            depth = depth / max_depth
            head_box = head_box.cpu().detach().numpy()*224
            head_box = head_box.astype(int)
            gtbox = gtbox.cpu().detach().numpy()*224
            gtbox = gtbox.astype(int)
            label = np.zeros((image.shape[0],3))
            for i in range(image.shape[0]):
                gt = (gt_label[i] - head[i])/224
                label[i,0] = gt[0]
                label[i,1] = gt[1]
                hbox_binary = torch.from_numpy(get_bb_binary(head_box[i]))
                gtbox_binary = torch.from_numpy(get_bb_binary(gtbox[i]))
                hbox_depth = torch.mul(depth[i], hbox_binary)
                gtbox_depth = torch.mul(depth[i], gtbox_binary)
                head_depth = torch.sum(hbox_depth) / torch.sum(hbox_binary==1)
                gt_depth = torch.sum(gtbox_depth) / torch.sum(gtbox_binary==1)
                label[i, 2] = (gt_depth - head_depth)
                # label[i,2] = (depth[i,:,gt_label[i,0],gt_label[i,1]] - depth[i,:,head[i,0],head[i,1]])
            label = torch.tensor(label, dtype=torch.float)
            gaze = gaze.cpu()
            loss = criterion(gaze, label)
            loss.backward()
            optimizer.step()
            running_loss.append(loss.item())

            if i % 10 == 9:
                logger.info('%s'%(str(np.mean(running_loss))))
                # writer.add_scalar('training_loss',np.mean(running_loss),epoch*n_total_steps+i)
                running_loss = []


         # Validation
        model.eval()
        for i, (img, face, location_channel,object_channel,head_channel ,head,gt_label, head_box, gtbox) in tqdm(enumerate(validation_data_loader), total=len(validation_data_loader)) :
            image = img.cuda()
            face = face.cuda()
            gt_label = gt_label
            head = head
            optimizer.zero_grad()
            gaze, depth = model(image, face)
            depth = depth.cpu()
            max_depth = torch.max(depth)
            depth = depth / max_depth
            head_box = head_box.cpu().detach().numpy() * 224
            head_box = head_box.astype(int)
            gtbox = gtbox.cpu().detach().numpy() * 224
            gtbox = gtbox.astype(int)
            label = np.zeros((image.shape[0], 3))
            for i in range(image.shape[0]):
                gt = (gt_label[i] - head[i]) / 224
                label[i, 0] = gt[0]
                label[i, 1] = gt[1]
                hbox_binary = torch.from_numpy(get_bb_binary(head_box[i]))
                gtbox_binary = torch.from_numpy(get_bb_binary(gtbox[i]))
                hbox_depth = torch.mul(depth[i], hbox_binary)
                gtbox_depth = torch.mul(depth[i], gtbox_binary)
                head_depth = torch.sum(hbox_depth) / torch.sum(hbox_binary == 1)
                gt_depth = torch.sum(gtbox_depth) / torch.sum(gtbox_binary == 1)
                label[i, 2] = (gt_depth - head_depth)
                # label[i,2] = (depth[i,:,gt_label[i,0],gt_label[i,1]] - depth[i,:,head[i,0],head[i,1]])
            label = torch.tensor(label, dtype=torch.float)
            gaze = gaze.cpu()
            loss = criterion(gaze, label)
            validation_loss.append(loss.item())
        val_loss = np.mean(validation_loss)

        logger.info('%s'%(str(val_loss)))
        writer.add_scalar('validation_loss',val_loss,epoch)
        validation_loss = []

        early_stopping(val_loss, model, optimizer, epoch, logger)
        if early_stopping.early_stop:
            print("Early stopping")
            break

    return model


def test_face3d(model, test_data_loader, logger, test_depth=True, save_output=False):
    model.eval()
    angle_error = []
    with torch.no_grad():
        for img, face, location_channel,object_channel,head_channel ,head,gt_label,heatmap, head_box, gtbox in test_data_loader:
            image =  img.cuda()
            face = face.cuda()
            gaze,depth = model(image,face)
            max_depth = torch.max(depth)
            depth = depth / max_depth
            depth =  depth.cpu()
            gaze =  gaze.cpu().data.numpy()
            head_box = head_box.cpu().detach().numpy() * 224
            head_box = head_box.astype(int)
            gtbox = gtbox.cpu().detach().numpy() * 224
            gtbox = gtbox.astype(int)
            label = np.zeros((image.shape[0],3))
            for i in range(image.shape[0]):
                gt = (gt_label[i] - head[i])/224
                label[i,0] = gt[0]
                label[i,1] = gt[1]
                hbox_binary = torch.from_numpy(get_bb_binary(head_box[i]))
                gtbox_binary = torch.from_numpy(get_bb_binary(gtbox[i]))
                hbox_depth = torch.mul(depth[i], hbox_binary)
                gtbox_depth = torch.mul(depth[i], gtbox_binary)
                head_depth = torch.sum(hbox_depth) / torch.sum(hbox_binary == 1)
                gt_depth = torch.sum(gtbox_depth) / torch.sum(gtbox_binary == 1)
                label[i, 2] = (gt_depth - head_depth)
                # label[i,2] = (depth[i,:,gt_label[i,0],gt_label[i,1]] - depth[i,:,head[i,0],head[i,1]])
            for i in range(img.shape[0]):
                if test_depth == True:
                    ae = np.dot(gaze[i,:],label[i,:])/np.sqrt(np.dot(label[i,:],label[i,:])*np.dot(gaze[i,:],gaze[i,:]))
                else:
                    ae = np.dot(gaze[i,:2],label[i,:2])/np.sqrt(np.dot(label[i,:2],label[i,:2])*np.dot(gaze[i,:2], gaze[i,:2]))
                ae = np.arccos(np.maximum(np.minimum(ae,1.0),-1.0)) * 180 / np.pi
                angle_error.append(ae)
        angle_error = np.mean(np.array(angle_error),axis=0)
    print(angle_error)


def train_face3d_gazefollow(model,train_data_loader,validation_data_loader, criterion, optimizer, logger, writer ,num_epochs=5,patience=10):
    since = time.time()
    n_total_steps = len(train_data_loader)
    n_total_steps_val = len(validation_data_loader)
    early_stopping = EarlyStopping(patience=patience, verbose=True)
    loss_amp = 10

    for epoch in range(num_epochs):

        model.train()  # Set model to training mode

        running_loss = []
        validation_loss = []

        for i, (img, face, head_point, gt_point) in tqdm(enumerate(train_data_loader), total=len(train_data_loader)) :
            image =  img.cuda()
            face = face.cuda()
            gt_label = gt_point
            head = head_point
            optimizer.zero_grad()
            gaze,depth = model(image,face)
            depth =  depth.cpu()
            max_depth = torch.max(depth)
            depth = depth / max_depth
            label = np.zeros((image.shape[0],3))
            for i in range(image.shape[0]):
                gt = (gt_label[i] - head[i])
                label[i,0] = gt[0]
                label[i,1] = gt[1]
                gt_x = int(gt_label[i, 0]*224)
                gt_y = int(gt_label[i, 1]*224)
                head_x = int(head[i, 0]*224)
                head_y = int(head[i, 1]*224)
                label[i,2] = (depth[i,:,gt_x,gt_y] - depth[i,:,head_x, head_y])
            label = torch.tensor(label, dtype=torch.float)
            gaze = gaze.cpu()
            loss = criterion(gaze, label)*loss_amp
            loss.backward()
            optimizer.step()
            running_loss.append(loss.item())

            if i % 10 == 9:
                logger.info('%s'%(str(np.mean(running_loss))))
                writer.add_scalar('training_loss',np.mean(running_loss),epoch*n_total_steps+i)
                running_loss = []


         # Validation
        model.eval()
        for i, (img, face, head_point, gt_point) in tqdm(enumerate(validation_data_loader), total=len(validation_data_loader)) :
            image =  img.cuda()
            face = face.cuda()
            gt_label = gt_point
            head = head_point
            optimizer.zero_grad()
            gaze,depth = model(image,face)
            depth =  depth.cpu()
            max_depth = torch.max(depth)
            depth = depth / max_depth
            label = np.zeros((image.shape[0],3))
            for i in range(image.shape[0]):
                gt = (gt_label[i] - head[i])
                label[i,0] = gt[0]
                label[i,1] = gt[1]
                gt_x = int(gt_label[i, 0]*224)
                gt_y = int(gt_label[i, 1]*224)
                head_x = int(head[i, 0]*224)
                head_y = int(head[i, 1]*224)
                label[i,2] = (depth[i,:,gt_x,gt_y] - depth[i,:,head_x, head_y])
            label = torch.tensor(label, dtype=torch.float)
            gaze = gaze.cpu()
            loss = criterion(gaze, label)*loss_amp
            validation_loss.append(loss.item())
        val_loss = np.mean(validation_loss)

        logger.info('%s'%(str(val_loss)))
        writer.add_scalar('validation_loss',val_loss,epoch)
        validation_loss = []

        early_stopping(val_loss, model, optimizer, epoch, logger)
        if early_stopping.early_stop:
            print("Early stopping")
            break

    return model


def train_face_depth(model,train_data_loader,validation_data_loader, criterion, optimizer, logger, writer ,num_epochs=5,patience=10):
    since = time.time()
    n_total_steps = len(train_data_loader)
    n_total_steps_val = len(validation_data_loader)
    early_stopping = EarlyStopping(patience=patience, verbose=True)
    for epoch in range(num_epochs):

        model.train()  # Set model to training mode

        running_loss = []
        validation_loss = []
        for i, (img, face, location_channel,object_channel,head_channel ,head,gt_label,gaze_heatmap, head_box, gtbox) in tqdm(enumerate(train_data_loader), total=len(train_data_loader)) :
            image =  img.cuda()
            face = face.cuda()
            gt_label = gt_label
            head = head
            optimizer.zero_grad()
            gaze,depth = model(image,face)
            depth = depth.cpu()
            max_depth = torch.max(depth)
            depth = depth / max_depth
            head_box = head_box.cpu().detach().numpy()*224
            head_box = head_box.astype(int)
            gtbox = gtbox.cpu().detach().numpy()*224
            gtbox = gtbox.astype(int)
            label = np.zeros((image.shape[0],1))
            for i in range(image.shape[0]):
                # gt = (gt_label[i] - head[i])/224
                hbox_binary = torch.from_numpy(get_bb_binary(head_box[i]))
                gtbox_binary = torch.from_numpy(get_bb_binary(gtbox[i]))
                hbox_depth = torch.mul(depth[i], hbox_binary)
                gtbox_depth = torch.mul(depth[i], gtbox_binary)
                head_depth = torch.sum(hbox_depth) / torch.sum(hbox_binary==1)
                gt_depth = torch.sum(gtbox_depth) / torch.sum(gtbox_binary==1)
                label[i, 0] = (gt_depth - head_depth)
            label = torch.tensor(label, dtype=torch.float)
            gaze = gaze.cpu()
            loss = criterion(gaze, label)
            loss.backward()
            optimizer.step()
            running_loss.append(loss.item())

            if i % 10 == 9:
                logger.info('%s'%(str(np.mean(running_loss))))
                writer.add_scalar('training_loss',np.mean(running_loss),epoch*n_total_steps+i)
                running_loss = []


         # Validation
        model.eval()
        for i, (img, face, location_channel,object_channel,head_channel ,head,gt_label,gaze_heatmap, head_box, gtbox) in tqdm(enumerate(validation_data_loader), total=len(validation_data_loader)) :
            image =  img.cuda()
            face = face.cuda()
            gt_label = gt_label
            head = head
            optimizer.zero_grad()
            gaze,depth = model(image,face)
            depth = depth.cpu()
            max_depth = torch.max(depth)
            depth = depth / max_depth
            head_box = head_box.cpu().detach().numpy()*224
            head_box = head_box.astype(int)
            gtbox = gtbox.cpu().detach().numpy()*224
            gtbox = gtbox.astype(int)
            label = np.zeros((image.shape[0],1))
            for i in range(image.shape[0]):
                # gt = (gt_label[i] - head[i])/224
                hbox_binary = torch.from_numpy(get_bb_binary(head_box[i]))
                gtbox_binary = torch.from_numpy(get_bb_binary(gtbox[i]))
                hbox_depth = torch.mul(depth[i], hbox_binary)
                gtbox_depth = torch.mul(depth[i], gtbox_binary)
                head_depth = torch.sum(hbox_depth) / torch.sum(hbox_binary==1)
                gt_depth = torch.sum(gtbox_depth) / torch.sum(gtbox_binary==1)
                label[i, 0] = (gt_depth - head_depth)
            label = torch.tensor(label, dtype=torch.float)
            gaze = gaze.cpu()
            loss = criterion(gaze, label)
            validation_loss.append(loss.item())
        val_loss = np.mean(validation_loss)

        logger.info('%s'%(str(val_loss)))
        writer.add_scalar('validation_loss',val_loss,epoch)
        validation_loss = []

        early_stopping(val_loss, model, optimizer, epoch, logger)
        if early_stopping.early_stop:
            print("Early stopping")
            break

    return model


def test_face_depth(model, test_data_loader, logger, save_output=False):
    model.eval()
    angle_error = []
    with torch.no_grad():
        for img, face, location_channel,object_channel,head_channel ,head,gt_label,heatmap, head_box, gtbox in test_data_loader:
            image =  img.cuda()
            face = face.cuda()
            gaze,depth = model(image,face)
            max_depth = torch.max(depth)
            depth = depth / max_depth
            depth =  depth.cpu()
            gaze =  gaze.cpu().data.numpy()
            head_box = head_box.cpu().detach().numpy() * 224
            head_box = head_box.astype(int)
            gtbox = gtbox.cpu().detach().numpy() * 224
            gtbox = gtbox.astype(int)
            label = np.zeros((image.shape[0],3))
            for i in range(image.shape[0]):
                hbox_binary = torch.from_numpy(get_bb_binary(head_box[i]))
                gtbox_binary = torch.from_numpy(get_bb_binary(gtbox[i]))
                hbox_depth = torch.mul(depth[i], hbox_binary)
                gtbox_depth = torch.mul(depth[i], gtbox_binary)
                head_depth = torch.sum(hbox_depth) / torch.sum(hbox_binary == 1)
                gt_depth = torch.sum(gtbox_depth) / torch.sum(gtbox_binary == 1)
                label[i, 0] = (gt_depth - head_depth)
            for i in range(img.shape[0]):
                ae = np.dot(gaze[i,0],label[i,0])/np.sqrt(np.dot(label[i,0],label[i,0])*np.dot(gaze[i,0], gaze[i,0]))
                ae = np.arccos(np.maximum(np.minimum(ae,1.0),-1.0)) * 180 / np.pi
                angle_error.append(ae)
        angle_error = np.mean(np.array(angle_error),axis=0)
    print(angle_error)



def train_face3d_bias(model,train_data_loader,validation_data_loader, criterion, optimizer, logger, writer ,num_epochs=5,patience=10):
    since = time.time()
    n_total_steps = len(train_data_loader)
    n_total_steps_val = len(validation_data_loader)
    early_stopping = EarlyStopping(patience=patience, verbose=True)
    pinball_loss = PinBallLoss()

    for epoch in range(num_epochs):

        model.train()  # Set model to training mode

        running_loss = []
        validation_loss = []
        for i, (img, face, location_channel,object_channel,head_channel ,head,gt_label,gaze_heatmap, head_box, gtbox) in tqdm(enumerate(train_data_loader), total=len(train_data_loader)) :
            image =  img.cuda()
            face = face.cuda()
            gt_label = gt_label
            head = head
            optimizer.zero_grad()
            gaze, bias, depth = model(image,face)
            depth =  depth.cpu()
            max_depth = torch.max(depth)
            depth = depth / max_depth
            head_box = head_box.cpu().detach().numpy()*224
            head_box = head_box.astype(int)
            gtbox = gtbox.cpu().detach().numpy()*224
            gtbox = gtbox.astype(int)
            label = np.zeros((image.shape[0],3))
            for i in range(image.shape[0]):
                gt = (gt_label[i] - head[i])/224
                label[i,0] = gt[0]
                label[i,1] = gt[1]
                hbox_binary = torch.from_numpy(get_bb_binary(head_box[i]))
                gtbox_binary = torch.from_numpy(get_bb_binary(gtbox[i]))
                hbox_depth = torch.mul(depth[i], hbox_binary)
                gtbox_depth = torch.mul(depth[i], gtbox_binary)
                head_depth = torch.sum(hbox_depth) / torch.sum(hbox_binary==1)
                gt_depth = torch.sum(gtbox_depth) / torch.sum(gtbox_binary==1)
                label[i, 2] = (gt_depth - head_depth)
            label = torch.tensor(label, dtype=torch.float)
            gaze = gaze.cpu()
            bias = bias.cpu()
            loss = pinball_loss(gaze, label, bias)
            loss.backward()
            optimizer.step()
            running_loss.append(loss.item())

            if i % 10 == 9:
                logger.info('%s'%(str(np.mean(running_loss))))
                writer.add_scalar('training_loss',np.mean(running_loss),epoch*n_total_steps+i)
                running_loss = []


         # Validation
        model.eval()
        for i, (img, face, location_channel,object_channel,head_channel ,head,gt_label,gaze_heatmap, head_box, gtbox) in tqdm(enumerate(validation_data_loader), total=len(validation_data_loader)) :
            image = img.cuda()
            face = face.cuda()
            gt_label = gt_label
            head = head
            optimizer.zero_grad()
            gaze, bias, depth = model(image, face)
            depth = depth.cpu()
            max_depth = torch.max(depth)
            depth = depth / max_depth
            head_box = head_box.cpu().detach().numpy() * 224
            head_box = head_box.astype(int)
            gtbox = gtbox.cpu().detach().numpy() * 224
            gtbox = gtbox.astype(int)
            label = np.zeros((image.shape[0], 3))
            for i in range(image.shape[0]):
                gt = (gt_label[i] - head[i]) / 224
                label[i, 0] = gt[0]
                label[i, 1] = gt[1]
                hbox_binary = torch.from_numpy(get_bb_binary(head_box[i]))
                gtbox_binary = torch.from_numpy(get_bb_binary(gtbox[i]))
                hbox_depth = torch.mul(depth[i], hbox_binary)
                gtbox_depth = torch.mul(depth[i], gtbox_binary)
                head_depth = torch.sum(hbox_depth) / torch.sum(hbox_binary == 1)
                gt_depth = torch.sum(gtbox_depth) / torch.sum(gtbox_binary == 1)
                label[i, 2] = (gt_depth - head_depth)
            label = torch.tensor(label, dtype=torch.float)
            gaze = gaze.cpu()
            bias = bias.cpu()
            loss = pinball_loss(gaze, label, bias)
            validation_loss.append(loss.item())
        val_loss = np.mean(validation_loss)

        logger.info('%s'%(str(val_loss)))
        writer.add_scalar('validation_loss',val_loss,epoch)
        validation_loss = []

        early_stopping(val_loss, model, optimizer, epoch, logger)
        if early_stopping.early_stop:
            print("Early stopping")
            break

    return model


def test_face3d_bias(model, test_data_loader, logger, test_depth=True, save_output=False):
    model.eval()
    angle_error = []
    with torch.no_grad():
        for img, face, location_channel,object_channel,head_channel ,head,gt_label,heatmap, head_box, gtbox in test_data_loader:
            image =  img.cuda()
            face = face.cuda()
            gaze, bias, depth = model(image,face)
            max_depth = torch.max(depth)
            depth = depth / max_depth
            depth =  depth.cpu()
            gaze =  gaze.cpu().data.numpy()
            head_box = head_box.cpu().detach().numpy() * 224
            head_box = head_box.astype(int)
            gtbox = gtbox.cpu().detach().numpy() * 224
            gtbox = gtbox.astype(int)
            label = np.zeros((image.shape[0],3))
            for i in range(image.shape[0]):
                gt = (gt_label[i] - head[i])/224
                label[i,0] = gt[0]
                label[i,1] = gt[1]
                hbox_binary = torch.from_numpy(get_bb_binary(head_box[i]))
                gtbox_binary = torch.from_numpy(get_bb_binary(gtbox[i]))
                hbox_depth = torch.mul(depth[i], hbox_binary)
                gtbox_depth = torch.mul(depth[i], gtbox_binary)
                head_depth = torch.sum(hbox_depth) / torch.sum(hbox_binary == 1)
                gt_depth = torch.sum(gtbox_depth) / torch.sum(gtbox_binary == 1)
                label[i, 2] = (gt_depth - head_depth)
            for i in range(img.shape[0]):
                if test_depth == True:
                    ae = np.dot(gaze[i,:],label[i,:])/np.sqrt(np.dot(label[i,:],label[i,:])*np.dot(gaze[i,:],gaze[i,:]))
                else:
                    ae = np.dot(gaze[i,:2],label[i,:2])/np.sqrt(np.dot(label[i,:2],label[i,:2])*np.dot(gaze[i,:2], gaze[i,:2]))
                ae = np.arccos(np.maximum(np.minimum(ae,1.0),-1.0)) * 180 / np.pi
                angle_error.append(ae)
        angle_error = np.mean(np.array(angle_error),axis=0)
    print(angle_error)

def t(p, q, r):
    x = p-q
    return np.dot(r-q, x)/np.dot(x, x)

def d(p, q, r):
    return np.linalg.norm(t(p, q, r)*(p-q)+q-r)

def calc_ang_err(output, target, eyes):
    pred_point = output
    eye_point = eyes
    gt_point = target
    pred_dir = pred_point - eye_point
    gt_dir = gt_point - eye_point
    norm_pred = (pred_dir[0] ** 2 + pred_dir[1] ** 2) ** 0.5
    norm_gt = (gt_dir[0] ** 2 + gt_dir[1] ** 2) ** 0.5
    cos_sim = (pred_dir[0] * gt_dir[0] + pred_dir[1] * gt_dir[1]) / \
              (norm_gt * norm_pred + 1e-6)
    cos_sim = np.maximum(np.minimum(cos_sim, 1.0), -1.0)
    ang_error = np.arccos(cos_sim) * 180 / np.pi
    return ang_error

def bb_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0]) # left
    yA = max(boxA[1], boxB[1]) # top
    xB = min(boxA[2], boxB[2]) # right
    yB = min(boxA[3], boxB[3]) # down
    if xB < xA or yB < yA:
        return 0.0
    interArea = (xB - xA) * (yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    iou = round(interArea / float(boxAArea + boxBArea - interArea), 2)
    assert iou >= 0.0
    assert iou <= 1.0
    return iou

def test_face3d_prediction(model, test_data_loader, logger, save_output=False):
    model.eval()
    angle_error = []
    l2=[]
    all_iou = []
    class_iou = []
    with torch.no_grad():
        for img, face, head, gt_label, centers, gaze_idx, gt_bboxes, gt_labels in test_data_loader:
            image = img.cuda()
            face = face.cuda()
            gt_bboxes = np.array(list(gt_bboxes))
            gt_labels = np.array(list(gt_labels))
            box_avg_width = np.mean(gt_bboxes[:, 0, 2] - gt_bboxes[:, 0, 0])
            box_avg_height = np.mean(gt_bboxes[:, 0, 3] - gt_bboxes[:, 0, 1])
            gaze, depth = model(image, face)
            # make normalized 3D points
            depth = depth.squeeze().detach().cpu().numpy()
            depth = depth / np.max(depth)
            for i in range (image.shape[0]):
                points = []
                for cen in centers[i]:
                    point = [cen[0]/224, cen[1]/224, depth[cen[1], cen[0]]]
                    points.append(point)
                points = np.array(points)
                head_point = points[-1,:]
                gt_point = points[gaze_idx, :]
                xyz = gaze.detach().cpu().numpy()[i]
                xyz = xyz + head_point
                pred_box = np.array([(xyz[0]-box_avg_width), (xyz[1]-box_avg_height), (xyz[0]+box_avg_width), (xyz[1]+box_avg_height)])
                pred_box = pred_box * [640, 480, 640, 480]
                pred_box = pred_box.astype(int)
                # bbox IOU
                max_id = -1
                max_iou = 0
                for k, b in enumerate(gt_bboxes):
                    b = b[0] * [640, 480, 640, 480]
                    b = b.astype(int)
                    iou = bb_iou(b, pred_box)
                    if iou > max_iou:
                        max_iou = iou
                        max_id = k
                # nearest box by box_iou
                if max_id == -1:
                    all_iou.append(0)
                elif (gaze_idx[i] == max_id):
                    all_iou.append(1)
                else:
                    all_iou.append(0)
                # by class
                if max_id == -1:
                    class_iou.append(0)
                elif (gt_labels[gaze_idx[i]] == gt_labels[max_id]):
                    class_iou.append(1)
                else:
                    class_iou.append(0)
        class_auc = (sum(class_iou) / len(class_iou)) * 100
        iou_auc = (sum(all_iou) / len(all_iou)) * 100
        print (iou_auc, class_auc)

                # Angular error
                # xyz_copy = np.repeat(np.expand_dims(xyz, axis=0), points.shape[0], axis=0)
                # dist = np.sum(np.sqrt((points - xyz_copy)**2), axis=1)
                # distance = np.apply_along_axis(lambda x: d(x, head_point, xyz), 1, points[:-1, :])
                # pred_idx = np.argmin(dist)
                # gt_idx = gaze_idx
                # pred_point = points[pred_idx]
                # # ang_error = calc_ang_err(xyz[:2], gt_point[:2], head_point[:2])
                # # angle_error.append(ang_error)
                # label = (gt_point - head_point)[i]
                # # pred = pred_point - head_point
                # pred = xyz
                # ae = np.dot(pred[:2],label[:2])/(np.sqrt(np.dot(label[:2],label[:2])*np.dot(pred[:2], pred[:2])) + np.finfo(np.float32).eps)
                # ae = np.arccos(np.maximum(np.minimum(ae,1.0),-1.0)) * 180 / np.pi
                # angle_error.append(ae)

                # L2 dist
                # euclid_dist = np.sqrt(np.power((gt_point[i, 0] - xyz[0]), 2) + np.power((gt_point[i, 1] - xyz[1]), 2))
                # l2.append(euclid_dist)
        # angle_error = np.mean(np.array(angle_error), axis=0)
        # l2_dist = np.mean(np.array(l2), axis=0)
    # print(l2_dist)


