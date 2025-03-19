from util.EEGDataLoader import EEGDataLoader
from model.EEGTransformer import EEGTransformer

_dir = 'data/'
params = {
            'subjectID' : 1
}
_data = EEGDataLoader(_dir, params)
_data.load_data()


batch_size = 100
n_epochs = 2000
img_height = 22
img_width = 600
channels = 1
c_dim = 4
lr = 0.0002
b1 = 0.5
b2 = 0.999
alpha = 0.0002
dimension = (190, 50)
start_epoch = 0

import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch import Tensor

from torch.utils.data import DataLoader
from torch.autograd import Variable
#from torchsummary import summary
import torch.autograd as autograd

gpus = [1]
from torch.backends import cudnn
cudnn.benchmark = False
cudnn.deterministic = True



img, label, test_data, test_label = _data.trainData, _data.trainLabel, _data.testData, _data.testLabel

img = torch.from_numpy(img)
label = torch.from_numpy(label - 1)


dataset = torch.utils.data.TensorDataset(img, label)
dataloader = torch.utils.data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)

test_data = torch.from_numpy(test_data)
test_label = torch.from_numpy(test_label - 1)
test_dataset = torch.utils.data.TensorDataset(test_data, test_label)
test_dataloader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)

for i in range(c_dim):
    centers[i] = torch.randn(dimension)
    centers[i] = centers[i].cuda()

# Optimizers
optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(b1, b2))

test_data = Variable(test_data.type(Tensor))
test_label = Variable(test_label.type(LongTensor))

bestAcc = 0
averAcc = 0
num = 0
Y_true = 0
Y_pred = 0

# Train the cnn model
total_step = len(dataloader)
curr_lr = lr

for e in range(n_epochs):
    in_epoch = time.time()
    model.train()
    for i, (img, label) in enumerate(dataloader):

        img = Variable(img.cuda().type(Tensor))
        label = Variable(label.cuda().type(LongTensor))

        #aug_data, aug_label = interaug(allData, allLabel)
        #img = torch.cat((img, aug_data))
        #label = torch.cat((label, aug_label))

        tok, outputs = model(img)

        loss = criterion_cls(outputs, label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    out_epoch = time.time()

    if (e + 1) % 1 == 0:
        model.eval()
        Tok, Cls = model(test_data)


        loss_test = criterion_cls(Cls, test_label)
        y_pred = torch.max(Cls, 1)[1]
        acc = float((y_pred == test_label).cpu().numpy().astype(int).sum()) / float(test_label.size(0))
        train_pred = torch.max(outputs, 1)[1]
        train_acc = float((train_pred == label).cpu().numpy().astype(int).sum()) / float(label.size(0))
        print('Epoch:', e,
              '  Train loss: %.6f' % loss.detach().cpu().numpy(),
              '  Test loss: %.6f' % loss_test.detach().cpu().numpy(),
              '  Train accuracy %.6f' % train_acc,
              '  Test accuracy is %.6f' % acc)
        num = num + 1
        averAcc = averAcc + acc
        if acc > bestAcc:
            bestAcc = acc
            Y_true = test_label
            Y_pred = y_pred

averAcc = averAcc / num
print('The average accuracy is:', averAcc)
print('The best accuracy is:', bestAcc)