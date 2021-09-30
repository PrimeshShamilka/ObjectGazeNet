import time
import torch
import torch.optim as optim
import numpy as np
from early_stopping_pytorch.pytorchtools import EarlyStopping
from tqdm import tqdm

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
        for i, (img, face, location_channel,object_channel,head_channel ,head,gt_label,gaze_heatmap, head_box, gtbox) in tqdm(enumerate(train_data_loader), total=len(train_data_loader)) :
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
            label =  torch.tensor(label, dtype=torch.float).cuda()
            loss = criterion(label,gaze)
            loss.backward()
            optimizer.step()
            running_loss.append(loss.item())

            if i % 10 == 9:
                logger.info('%s'%(str(np.mean(running_loss))))
                # writer.add_scalar('training_loss',np.mean(running_loss),epoch*n_total_steps+i)
                running_loss = []


         # Validation
        model.eval()
        for i, (img, face, location_channel,object_channel,head_channel ,head,gt_label,gaze_heatmap, head_box, gtbox) in tqdm(enumerate(train_data_loader), total=len(train_data_loader)) :
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
            label =  torch.tensor(label, dtype=torch.float).cuda()
            loss = criterion(label,gaze)
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