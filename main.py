import numpy as np

import torch
import torch.nn as nn
from torch import Tensor

from torch.utils.data import DataLoader
from torch.autograd import Variable

from util.helpers import helpers
from util.EEGDataLoader import EEGDataLoader
from model.EEGClassifier import EEGClassifier

gpus = [1]

# hyperparameters
batch_size = 100
n_epochs = 100
lr = 0.0002
b1 = 0.5
b2 = 0.999
alpha = 0.0002


Tensor = torch.cuda.FloatTensor
LongTensor = torch.cuda.LongTensor
criterion_l1 = torch.nn.L1Loss().cuda()
criterion_l2 = torch.nn.MSELoss().cuda()
criterion_cls = torch.nn.CrossEntropyLoss().cuda()

# load pilot data 
_dir = 'data/'
_data = EEGDataLoader(_dir)

# data format trainData: 360x3x1000 (trials x eeg channels x time) note: time is sequence, split by milisecond
data, label, test_data, test_label = _data.trainData, _data.trainLabel, _data.testData, _data.testLabel

data = torch.from_numpy(data)
label = torch.from_numpy(label)
dataset = torch.utils.data.TensorDataset(data, label)
dataloader = torch.utils.data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)

test_data = torch.from_numpy(test_data)
test_label = torch.from_numpy(test_label)
test_data = Variable(test_data.type(Tensor))
test_label = Variable(test_label.type(LongTensor))


model = EEGClassifier(num_channels=len(data[0][0]))
model = nn.DataParallel(model, device_ids=[i for i in range(len(gpus))])
model = model.cuda()

# Optimizers
optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(b1, b2))

accuracies = []

for e in range(n_epochs):
    # train
    model.train()
    for i, (data, label) in enumerate(dataloader):

        data = Variable(data.cuda().type(Tensor))
        label = Variable(label.cuda().type(LongTensor))

        # optional: data augmentation
        aug_data, aug_label = helpers().augment(_data.trainData, _data.trainLabel, batch_size)
        aug_data = torch.from_numpy(aug_data).cuda().float()
        aug_label = torch.from_numpy(aug_label).cuda().long()
        data = torch.cat((data, aug_data))
        label = torch.cat((label, aug_label))
        ###
        
        _, outputs = model(data)

        loss = criterion_cls(outputs, label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    train_pred = torch.max(outputs, 1)[1]
    train_acc = float((train_pred == label).cpu().numpy().astype(int).sum()) / float(label.size(0))
    
    # predict
    model.eval()
    _, probs = model(test_data)
    
    loss_test = criterion_cls(probs, test_label)
    y_pred = torch.max(probs, 1)[1] # get indices of max prob
    test_acc = float((y_pred == test_label).cpu().numpy().astype(int).sum()) / float(test_label.size(0))
    print('Epoch:', e,
          '  Train loss: %.6f' % loss.detach().cpu().numpy(),
          '  Test loss: %.6f' % loss_test.detach().cpu().numpy(),
          '  Train accuracy %.6f' % train_acc,
          '  Test accuracy is %.6f' % test_acc)
    accuracies.append(test_acc)
        
print('Average accuracy:', np.mean(accuracies))