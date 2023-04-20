# -*- coding: utf-8 -*-
"""
Created on Mon Apr 18 22:16:49 2022
@author: QI YU
@email: yq123456leo@outlook.com
"""

import math
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

def getpad(in_size, kernel_size, stride, label = 'same'):
    if label == 'same':
        out_size = math.ceil(in_size / stride)
        padding = math.ceil(((out_size - 1) * stride + kernel_size - in_size) / 2)
    elif label == 'valid':
        out_size = math.ceil((in_size - kernel_size + 1) / stride)
        padding = math.ceil(((out_size - 1) * stride + kernel_size - in_size) / 2)
    
    return padding, out_size

def crosscat(tensors, step = 1, dim = 2):
    tensor_x, tensor_y = tensors
    tx_subs = torch.split(tensor_x, step, dim = dim)
    ty_subs = torch.split(tensor_y, step, dim = dim)
    tx_n, ty_n = len(tx_subs), len(ty_subs)
    
    tensor_res = tx_subs[0]
    i = 0
    while i < ty_n and i + 1 < tx_n:
        tensor_res = torch.cat((tensor_res, ty_subs[i]), dim = dim)
        tensor_res = torch.cat((tensor_res, tx_subs[i + 1]), dim = dim)
        i += 1
    while i < ty_n:
        tensor_res = torch.cat((tensor_res, ty_subs[i]), dim = dim)
        i += 1
    while i + 1 < tx_n:
        tensor_res = torch.cat((tensor_res, tx_subs[i]), dim = dim)
        i += 1
    
    return tensor_res

class OrgConvLSTM(nn.Module):
    def __init__(self, num_classes, input_shape):
        super(OrgConvLSTM, self).__init__()
        
        batch_size, in_chs, in_size = input_shape
        
        padding, out_size = getpad(in_size, 8, 2, 'same')
        out_size //= 2 
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels = in_chs, out_channels = 32,
                      kernel_size = 8, stride = 2,
                      padding = padding),
            nn.ReLU(),
            nn.BatchNorm1d(num_features = 32),
            nn.MaxPool1d(kernel_size = 2, stride = 2)
            )
        
        padding, out_size = getpad(out_size, 4, 2, 'same')
        out_size //= 2
        self.conv2 = nn.Sequential(
            nn.Conv1d(in_channels = 32, out_channels = 32,
                      kernel_size = 4, stride = 2,
                      padding = padding),
            nn.ReLU(),
            nn.BatchNorm1d(num_features = 32),
            nn.MaxPool1d(kernel_size = 2, stride = 2)
            )
        
        in_size = out_size
        padding, out_size = getpad(out_size, 4, 1, 'same')
        self.conv3 = nn.Sequential(
            nn.Conv1d(in_channels = 32, out_channels = 32,
                      kernel_size = 4, stride = 1,
                      padding = padding),
            nn.ReLU(),
            nn.BatchNorm1d(num_features = 32)
            )
        
        out_size = (in_size - 4 + 2 * padding) // 1 + 1
        
        self.LSTM1 = nn.LSTM(input_size = out_size, hidden_size = 128)
        self.lstm_drop1 = nn.Dropout(0.5)
        self.LSTM2 = nn.LSTM(input_size = 128, hidden_size = 64)
        self.lstm_drop2 = nn.Dropout(0.2)
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features = 32 * 64, out_features = 256),
            nn.ReLU(),
            nn.Linear(in_features = 256, out_features = 128),
            nn.ReLU(),
            nn.Linear(in_features = 128, out_features = num_classes)
            )
    
    def forward(self, x):
        conv1_out = self.conv1(x)
        conv2_out = self.conv2(conv1_out)
        conv3_out = self.conv3(conv2_out)
        lstm1_out, (hn1, cn1) = self.LSTM1(conv3_out)
        lstm_drop1_out = self.lstm_drop1(lstm1_out)
        lstm2_out, (hn2, cn2) = self.LSTM2(lstm_drop1_out)
        lstm_drop2_out = self.lstm_drop2(lstm2_out)
        out = self.classifier(lstm_drop2_out)
        return out
    

class MyConvLSTM(nn.Module):
    def __init__(self, num_classes, input_shape):
        super(MyConvLSTM, self).__init__()
        
        batch_size, in_chs, in_size = input_shape
        
        padding, out_size = getpad(in_size, 8, 2, 'same')
        out_size //= 2 
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels = in_chs, out_channels = 32,
                      kernel_size = 8, stride = 2,
                      padding = padding),
            nn.ReLU(),
            nn.BatchNorm1d(num_features = 32),
            nn.MaxPool1d(kernel_size = 2, stride = 2)
            )
        
        padding, out_size = getpad(out_size, 4, 2, 'same')
        out_size //= 2
        self.conv2 = nn.Sequential(
            nn.Conv1d(in_channels = 32, out_channels = 32,
                      kernel_size = 4, stride = 2,
                      padding = padding),
            nn.ReLU(),
            nn.BatchNorm1d(num_features = 32),
            nn.MaxPool1d(kernel_size = 2, stride = 2)
            )
        
        in_size = out_size
        padding, out_size = getpad(out_size, 4, 1, 'same')
        self.conv3 = nn.Sequential(
            nn.Conv1d(in_channels = 32, out_channels = 32,
                      kernel_size = 4, stride = 1,
                      padding = padding),
            nn.ReLU(),
            nn.BatchNorm1d(num_features = 32)
            )
        
        out_size = (in_size - 4 + 2 * padding) // 1 + 1
        self.LSTM = nn.LSTM(input_size = out_size, hidden_size = 40)
        
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Flatten(),
            nn.Linear(in_features = 32 * 40, out_features = 128),
            nn.ReLU(),
            nn.Linear(in_features = 128, out_features = num_classes)
            )
    
    def forward(self, x):
        conv1_out = self.conv1(x)
        conv2_out = self.conv2(conv1_out)
        conv3_out = self.conv3(conv2_out)
        lstm_out, (hn, cn) = self.LSTM(conv3_out)
        out = self.classifier(lstm_out)
        return out

class BinConvLSTM(nn.Module):
    def __init__(self, num_classes, input_shape):
        super(BinConvLSTM, self).__init__()
        
        batch_size, in_chs, in_size = input_shape
        
        padding, out_size = getpad(in_size // 2, 8, 2, 'same')
        out_size //= 2 
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels = in_chs, out_channels = 32,
                      kernel_size = 8, stride = 2,
                      padding = padding),
            nn.ReLU(),
            nn.BatchNorm1d(num_features = 32),
            nn.MaxPool1d(kernel_size = 2, stride = 2)
            )
        
        padding, out_size = getpad(out_size, 4, 2, 'same')
        out_size //= 2
        self.conv2 = nn.Sequential(
            nn.Conv1d(in_channels = 32, out_channels = 32,
                      kernel_size = 4, stride = 2,
                      padding = padding),
            nn.ReLU(),
            nn.BatchNorm1d(num_features = 32),
            nn.MaxPool1d(kernel_size = 2, stride = 2)
            )
        
        in_size = out_size * 2
        padding, out_size = getpad(out_size * 2, 4, 1, 'same')
        self.conv3 = nn.Sequential(
            nn.Conv1d(in_channels = 32, out_channels = 32,
                      kernel_size = 4, stride = 1,
                      padding = padding),
            nn.ReLU(),
            nn.BatchNorm1d(num_features = 32)
            )
        
        out_size = (in_size - 4 + 2 * padding) // 1 + 1
        self.LSTM = nn.LSTM(input_size = out_size, hidden_size = 40)
        
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Flatten(),
            nn.Linear(in_features = 32 * 40, out_features = 128),
            nn.ReLU(),
            nn.Linear(in_features = 128, out_features = num_classes)
            )
        
    
    def forward(self, x):
        num_feat = x.shape[2]
        x1, x2 = torch.split(x, num_feat // 2, dim = 2)
        conv1_out1, conv1_out2 = self.conv1(x1), self.conv1(x2)
        conv2_out1, conv2_out2 = self.conv2(conv1_out1), self.conv2(conv1_out2)
        conv2_out = crosscat((conv2_out1, conv2_out2), step = 1, dim = 2)
        conv3_out = self.conv3(conv2_out)
        lstm_out, (hn, cn) = self.LSTM(conv3_out)
        out = self.classifier(lstm_out)
        return out
        
    
class DDIdataset(Dataset):
    def __init__(self, X, y):
        self.X = np.reshape(X, (X.shape[0], 1, X.shape[1]))
        self.y = y
        self.num_samples = self.X.shape[0]
    
    def __getitem__(self, index):
        inputs = self.X[index]
        label = self.y[index]
        return inputs, label
    
    def __len__(self):
        return self.num_samples

if __name__ == "__main__":
    inputs = torch.rand(128, 1, 400)
    conv_lstm = OrgConvLSTM(num_classes = 2, input_shape = inputs.shape)
    outputs = conv_lstm.forward(inputs)
        