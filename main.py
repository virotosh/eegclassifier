import time
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch import Tensor

from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch.autograd as autograd


from util.helpers import helpers
from util.EEGDataLoader import EEGDataLoader
from model.EEGTransformer import EEGTransformer


batch_size = 100
n_epochs = 100#2000
lr = 0.0002
b1 = 0.5
b2 = 0.999
alpha = 0.0002

gpus = [1]
from torch.backends import cudnn
cudnn.benchmark = False
cudnn.deterministic = True

Tensor = torch.cuda.FloatTensor
LongTensor = torch.cuda.LongTensor
criterion_l1 = torch.nn.L1Loss().cuda()
criterion_l2 = torch.nn.MSELoss().cuda()
criterion_cls = torch.nn.CrossEntropyLoss().cuda()

model = EEGTransformer()
model = nn.DataParallel(model, device_ids=[i for i in range(len(gpus))])
model = model.cuda()


# load pilot data 
_dir = 'data/'
params = {
            'subjectID' : 1
}
_data = EEGDataLoader(_dir, params)
_data.load_data()

# data format 1000x3x120 (time x eeg channels x trials) note: time is sequence, split by milisecond
data, label, test_data, test_label = _data.trainData, _data.trainLabel, _data.testData, _data.testLabel

data = torch.from_numpy(data)
label = torch.from_numpy(label)
dataset = torch.utils.data.TensorDataset(data, label)
dataloader = torch.utils.data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)

test_data = torch.from_numpy(test_data)
test_label = torch.from_numpy(test_label)
test_dataset = torch.utils.data.TensorDataset(test_data, test_label)
test_dataloader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)

# Optimizers
optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(b1, b2))

test_data = Variable(test_data.type(Tensor))
test_label = Variable(test_label.type(LongTensor))

Accuracy = []

# Train the model
for e in range(n_epochs):
    in_epoch = time.time()
    model.train()
    for i, (data, label) in enumerate(dataloader):

        data = Variable(data.cuda().type(Tensor))
        label = Variable(label.cuda().type(LongTensor))

        # optional data augmentation
        aug_data, aug_label = helpers().augment(_data.trainData, _data.trainLabel, batch_size)
        aug_data = torch.from_numpy(aug_data).cuda().float()
        aug_label = torch.from_numpy(aug_label).cuda().long()
        data = torch.cat((data, aug_data))
        label = torch.cat((label, aug_label))
        ###
        
        tok, outputs = model(data)

        loss = criterion_cls(outputs, label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    out_epoch = time.time()

    if (e + 1) % 1 == 0:
        model.eval()
        TOK, CLS = model(test_data)
        
        loss_test = criterion_cls(CLS, test_label)
        y_pred = torch.max(CLS, 1)[1] # get indices of max prob
        acc = float((y_pred == test_label).cpu().numpy().astype(int).sum()) / float(test_label.size(0))
        train_pred = torch.max(outputs, 1)[1]
        train_acc = float((train_pred == label).cpu().numpy().astype(int).sum()) / float(label.size(0))
        print('Epoch:', e,
              '  Train loss: %.6f' % loss.detach().cpu().numpy(),
              '  Test loss: %.6f' % loss_test.detach().cpu().numpy(),
              '  Train accuracy %.6f' % train_acc,
              '  Test accuracy is %.6f' % acc)
        Accuracy.append(acc)
print('The average accuracy is:', np.mean(Accuracy))