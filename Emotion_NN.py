#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 23 09:42:40 2020

@author: arjun
"""
from torch import nn
from torch import no_grad
from torch import optim
from torch import max
from torch.utils.data import Dataset, DataLoader, random_split, ConcatDataset
import scipy.io as scio

class SpeechDataset(Dataset):
    
    def __init__(self, mat_file_path):
        self.dataset = scio.loadmat(mat_file_path)
        self.features = self.dataset["features"]
        self.labels = self.dataset["labels"]
        self.info = self.dataset["info"]
    
    def __len__(self):
        
        return len(self.features[:,0])

    def __getitem__(self, index):
        frame = self.features[index,:]
        label = self.labels[0,index]
        info = self.info[index]
        return frame, label, info
    
zeros_dataset = SpeechDataset(mat_file_path='/home/arjun/.config/spyder-py3/zero_features.mat')
train_size = int(0.7*len(zeros_dataset.features[:,0]))
test_size = len(zeros_dataset.features[:,0]) - train_size
train_set_0, test_set_0 = random_split(zeros_dataset, [train_size, test_size])
ones_dataset = SpeechDataset(mat_file_path='/home/arjun/.config/spyder-py3/one_features.mat')
train_size = int(0.7*len(ones_dataset.features[:,0]))
test_size = len(ones_dataset.features[:,0]) - train_size
train_set_1, test_set_1 = random_split(ones_dataset, [train_size, test_size])
twos_dataset = SpeechDataset(mat_file_path='/home/arjun/.config/spyder-py3/two_features.mat')
train_size = int(0.7*len(twos_dataset.features[:,0]))
test_size = len(twos_dataset.features[:,0]) - train_size
train_set_2, test_set_2 = random_split(twos_dataset, [train_size, test_size])
threes_dataset = SpeechDataset(mat_file_path='/home/arjun/.config/spyder-py3/three_features.mat')
train_size = int(0.7*len(threes_dataset.features[:,0]))
test_size = len(threes_dataset.features[:,0]) - train_size
train_set_3, test_set_3 = random_split(threes_dataset, [train_size, test_size])
fours_dataset = SpeechDataset(mat_file_path='/home/arjun/.config/spyder-py3/four_features.mat')
train_size = int(0.7*len(fours_dataset.features[:,0]))
test_size = len(fours_dataset.features[:,0]) - train_size
train_set_4, test_set_4 = random_split(fours_dataset, [train_size, test_size])

trainloader = DataLoader(ConcatDataset([train_set_0,train_set_1,train_set_2,train_set_3,train_set_4]), batch_size=32, shuffle=True, drop_last=True)
testloader = DataLoader(ConcatDataset([test_set_0,test_set_1,test_set_2,test_set_3,test_set_4]), batch_size=1, shuffle=True, drop_last=False)


class Network(nn.Module):
    def __init__(self):
        super().__init__()
        self.fir = nn.Linear(144,80)
        self.dropout = nn.Dropout(p=0.01)
        self.bn = nn.BatchNorm1d(80)
        self.output = nn.Linear(80,5)
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, x):
        x = self.fir(x)
        x = self.dropout(x)
        x = self.bn(x)
        x = self.sigmoid(x)
        x = self.output(x)
        x = self.softmax(x)
        
        return x
    
model = Network()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.003)
model.train()
epochs = 5
for e in range(epochs):
    running_loss = 0
    for i, data in enumerate(trainloader):
        frames, labels, info = data
        optimizer.zero_grad()
        outputs = model(frames.float())
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    else:
         print(f"Training loss: {running_loss/len(trainloader)}")
        
model.eval()
correct = 0
total = 0
a = dict()

with no_grad():
    for frame, lab, info in testloader:
        #frame, lab = data
        outputs = model(frame.float())
        _, predicted = max(outputs.data, 1)
        print("info: " + str(info) + " label: " + str(lab) + " pred: " + str(outputs.data))
        if(a.get(info)!=None):
            a[info].append(predicted.item())
        else:
            a[info] = []
            a[info].append(predicted.item())
        total += lab.size(0)
        correct += (predicted == lab).sum().item()

print('Accuracy of the network on the test images: %d %%' % (
    100 * correct / total))
