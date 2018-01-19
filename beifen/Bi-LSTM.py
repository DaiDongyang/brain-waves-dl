import torch
from torch import nn
from torch.autograd import Variable
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import load_data
import numpy as np
#import matplotlib.pyplot as plt

def convert_oneHot_to_num(data):
    b = np.zeros(data.shape[0])
    for i in range(0, data.shape[0]):
        for j in range(0, data.shape[1]):
            if (data[i][j] == 1):
                break;
        b[i] = j
    return b

# torch.manual_seed(1)    # reproducible

# Hyper Parameters
EPOCH = 5               # train the training data n times, to save time, we just train 1 epoch
BATCH_SIZE = 200
TIME_STEP = 30          # rnn time step / image height
INPUT_SIZE = 50         # rnn input size / image width
LR = 0.01               # learning rate
#DOWNLOAD_MNIST = True   # set to True if haven't download the data


tvt = load_data.TrainValiTest()
tvt.load()
train_samples, train_ls = tvt.train_samples_ls()
vali_samples, vali_ls = tvt.vali_samples_ls()
test_samples, test_ls = tvt.test_samples_ls()
train_ls = convert_oneHot_to_num(train_ls)
vali_ls = convert_oneHot_to_num(vali_ls)
test_ls = convert_oneHot_to_num(test_ls)
train_samples = np.reshape(train_samples,(train_samples.shape[0],30,50))
vali_samples = np.reshape(vali_samples,(vali_samples.shape[0],30,50))
test_samples = np.reshape(test_samples,(test_samples.shape[0],30,50))


train_samples_torch = torch.from_numpy(train_samples).type(torch.FloatTensor)
train_ls_torch = torch.from_numpy(train_ls).type(torch.IntTensor)

# Data Loader for easy mini-batch return in training
brain_dataset = torch.utils.data.TensorDataset(data_tensor=train_samples_torch, target_tensor=train_ls_torch)
train_loader = torch.utils.data.DataLoader(dataset=brain_dataset, batch_size=BATCH_SIZE, shuffle=True)

test_samples_torch = torch.from_numpy(test_samples).type(torch.FloatTensor)
test_x = Variable(test_samples_torch)
test_y = test_ls

class RNN(nn.Module):
    def __init__(self):
        super(RNN, self).__init__()

        self.rnn = nn.LSTM(         # if use nn.RNN(), it hardly learns
            input_size=INPUT_SIZE,
            hidden_size=64,         # rnn hidden unit
            num_layers=1,           # number of rnn layer
            batch_first=True,       # input & output will has batch size as 1s dimension. e.g. (batch, time_step, input_size)
            bidirectional=True
        )

        self.out = nn.Linear(64*2, 3)

    def forward(self, x):
        # x shape (batch, time_step, input_size)
        # r_out shape (batch, time_step, output_size)
        # h_n shape (n_layers, batch, hidden_size)
        # h_c shape (n_layers, batch, hidden_size)
        r_out, (h_n, h_c) = self.rnn(x, None)   # None represents zero initial hidden state

        # choose r_out at the last time step
        out = self.out(r_out[:, -1, :])
        return out


rnn = RNN()
print(rnn)

optimizer = torch.optim.Adam(rnn.parameters(), lr=LR)   # optimize all cnn parameters
loss_func = nn.CrossEntropyLoss()                       # the target label is not one-hotted

# training and testing
for epoch in range(EPOCH):
    for step, (x, y) in enumerate(train_loader):        # gives batch data
        b_x = Variable(x.view(-1, 30, 50))              # reshape x to (batch, time_step, input_size)
        b_y = Variable(y)                               # batch y

        output = rnn(b_x)                               # rnn output
        loss = loss_func(output, b_y)                   # cross entropy loss
        optimizer.zero_grad()                           # clear gradients for this training step
        loss.backward()                                 # backpropagation, compute gradients
        optimizer.step()                                # apply gradients

        if step % 50 == 0:
            test_output = rnn(test_x)                   # (samples, time_step, input_size)
            pred_y = torch.max(test_output, 1)[1].data.numpy().squeeze()
            accuracy = sum(pred_y == test_y) / float(test_y.size)
            print('Epoch: ', epoch, '| train loss: %.4f' % loss.data[0], '| test accuracy: %.2f' % accuracy)
