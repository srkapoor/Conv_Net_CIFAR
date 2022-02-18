import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import sampler

import torchvision.datasets as dset
import torchvision.transforms as T

import matplotlib.pyplot as plt
import numpy as np
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import math


NUM_TRAIN = 49000
transform = T.Compose([
                T.ToTensor(),
                T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
            ])
cifar10_train = dset.CIFAR10('./cifar_data', train=True, download=True,
                             transform=transform)
loader_train = DataLoader(cifar10_train, batch_size=64,
                          sampler=sampler.SubsetRandomSampler(range(NUM_TRAIN)))
cifar10_val = dset.CIFAR10('./cifar_data', train=True, download=True,
                           transform=transform)
loader_val = DataLoader(cifar10_val, batch_size=64,
                        sampler=sampler.SubsetRandomSampler(range(NUM_TRAIN, 50000)))
cifar10_test = dset.CIFAR10('./cifar_data', train=False, download=True,
                            transform=transform)
loader_test = DataLoader(cifar10_test, batch_size=64)

'''
Device specification
'''
USE_GPU = False
dtype = torch.float32
if USE_GPU and torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
# Constant to control how frequently we print train loss
print_every = 100
print('using device:', device)
best_acc = 0

'''
Training :
Train a model on CIFAR-10 using the PyTorch Module API.

Inputs:
- model: A PyTorch Module giving the model to train.
- optimizer: An Optimizer object we will use to train the model
- epochs: (Optional) A Python integer giving the number of epochs to train for

Returns: Nothing, but prints model accuracies during training.
'''
def train(model, optimizer, epochs=1):
    loss_function = nn.CrossEntropyLoss()
    model = model.to(device=device)  # move the model parameters to CPU/GPU
    for e in range(epochs):
        for t, (x, y) in enumerate(loader_train):
            optimizer.zero_grad()
            outputs = model(x)
            loss = loss_function(outputs, y)
            loss.backward()
            optimizer.step()
            if t % print_every == 0:
                print('Epoch %d, Iteration %d, loss = %.4f' % (e, t, loss.item()))
                test(loader_val, model)
                print()
'''
Testing :
Test a model on CIFAR-10 using the PyTorch Module API.

Inputs:
- loader:
- model: A PyTorch Module giving the model to test.

Returns: Nothing, but prints model accuracies during training.
'''
def test(loader, model):
    global best_acc
    if loader.dataset.train:
        print('Checking accuracy on validation set')
    else:
        print('Checking accuracy on test set')
    num_correct = 0
    num_samples = 0
    model.eval()  # set model to evaluation mode
    with torch.no_grad():
        for x, y in loader:
            outputs = model(x)
            outputs,y = outputs.to(device), y.to(device)
            #predicted based on the model
            _, predicted = torch.max(outputs.data, 1)
            #updating number of samples
            num_samples += y.size(0)
            num_correct += (predicted == y).sum().item()
        acc = float(num_correct) / num_samples
        if loader.dataset.train and acc > best_acc:
            best_acc = acc
        print('Got %d / %d correct (%.2f)' % (num_correct, num_samples, 100 * acc))


class myNet(nn.Module):
    def __init__(self):
        super(myNet, self).__init__()
        # Set up your own convnets.
        #creating three convolutional layers and three fully connected layers with max pool and batch-normalization
        self.conv1 = nn.Conv2d(3, 32, 4)
        self.conv1_bn = nn.BatchNorm2d(32)
        self.dropout1 = nn.Dropout2d(0.25)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 70, 5)
        self.conv2_bn = nn.BatchNorm2d(70)
        #creating a downsampling layer
        self.conv3 = nn.Conv2d(70, 64, 5)
        self.conv3_bn = nn.BatchNorm2d(64)
        self.dropout2 = nn.Dropout2d(0.25)
        self.fc1 = nn.Linear(576, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # forward
        x = self.pool(F.relu(self.dropout1(self.conv1_bn(self.conv1(x)))))
        x = self.dropout2(self.conv2_bn(self.conv2(x)))
        x = self.pool(F.relu(self.conv3_bn(self.conv3(x))))
        x = x.view(-1, 576)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        out = x
        return out

model = myNet()
#setting our optimizier
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
##########################################################################

# You should get at least 70% accuracy
train(model, optimizer, epochs=10)
# load saved model to best_model for final testing
PATH = './cifar_net.pth'
torch.save(model.state_dict(), PATH)
best_model = myNet()
best_model.load_state_dict(torch.load(PATH))
test(loader_test, best_model)
