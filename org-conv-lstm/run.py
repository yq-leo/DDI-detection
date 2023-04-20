# -*- coding: utf-8 -*-
"""
Created on Thu Jun  2 01:20:47 2022

@author: surface
"""

import numpy as np
import sys
import pandas as pd
import itertools
import math
import time
import os

from sklearn import svm, linear_model, neighbors
from sklearn import tree, ensemble
from sklearn import metrics
from sklearn.naive_bayes import GaussianNB

import networkx as nx
import random
import numbers

from sklearn.model_selection import StratifiedKFold

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if device.type == 'cuda':
    cudnn.benchmark = True
    

def generatePairs(ddi_df, embedding_df, emb_prop = 1):
    print('Generating drug pairs ... ', end = '')
    drugs = set(ddi_df.Drug1.unique())
    drugs = drugs.union(ddi_df.Drug2.unique())
    emb_drugs = embedding_df.Drug.unique()
    np.random.shuffle(emb_drugs)
    drugs = drugs.intersection(emb_drugs[:int(emb_prop * emb_drugs.shape[0])])

    ddiKnown = set([tuple(x) for x in ddi_df[['Drug1','Drug2']].values])

    pairs = list()
    classes = list()

    for dr1, dr2 in itertools.combinations(sorted(drugs), 2):
        if dr1 == dr2: continue

        if (dr1, dr2) in ddiKnown or (dr2, dr1) in ddiKnown: 
            cls = 1  
        else:
            cls = 0

        pairs.append((dr1, dr2))
        classes.append(cls)

    pairs = np.array(pairs)        
    classes = np.array(classes)

    print('done')
    print('{:,d} valid drugs'.format(len(drugs)))
    print('{:,d} drug pairs, positive: {:,d}, negative: {:,d}'.format(len(pairs), sum(classes), len(classes) - sum(classes)))
    
    return pairs, classes


def balance_data(pairs, classes, n_proportion):
    classes = np.array(classes)
    pairs = np.array(pairs)
    
    indices_true = np.where(classes == 1)[0]
    indices_false = np.where(classes == 0)[0]

    np.random.shuffle(indices_false)
    indices = indices_false[:int(n_proportion * indices_true.shape[0])]
    print("pos(+): {:,d}, neg(-): {:,d}".format(len(indices_true), len(indices)))
    pairs = np.concatenate((pairs[indices_true], pairs[indices]), axis = 0)
    classes = np.concatenate((classes[indices_true], classes[indices]), axis = 0)
    
    return pairs, classes


def getpad(in_size, kernel_size, stride, label = 'same'):
    if label == 'same':
        out_size = math.ceil(in_size / stride)
        padding = math.ceil(((out_size - 1) * stride + kernel_size - in_size) / 2)
    elif label == 'valid':
        out_size = math.ceil((in_size - kernel_size + 1) / stride)
        padding = math.ceil(((out_size - 1) * stride + kernel_size - in_size) / 2)
    
    return padding, out_size


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


def exportRes(scores_df, filename):
    dirname = 'result'
    if not os.path.isdir(dirname):
        os.mkdir(dirname)

    columns = list(scores_df.columns)
    scores_df.to_csv(os.path.join(dirname, filename), columns = columns)


def run(feature_filename, dst_filename):
    ddi_filename = 'data/input/ddi_drugbank_v5.1.9_all.txt'
    ddi_df = pd.read_csv(ddi_filename, sep = '\t')
    ddi_df.head()

    # Load RDF2Vec SG embedding
    feature_filename = 'vectors/DB/pbg_drug_embeddings.txt'
    embedding_df = pd.read_csv(feature_filename, delimiter = '\t')
    embedding_df.Drug = embedding_df.Drug.str[-8:-1]

    emb_proportion = 0.25
    pairs, classes = generatePairs(ddi_df, embedding_df, emb_proportion)
    
    n_seed = 100
    n_fold = 10
    n_run = 1         
    n_proportion = 1

    random.seed(n_seed)
    np.random.seed(n_seed)

    pairs_sub, classes_sub = balance_data(pairs, classes, n_proportion)
            
    # Generate K-fold
    skf = StratifiedKFold(n_splits = n_fold, shuffle = True, random_state = n_seed)
    cv = skf.split(pairs_sub, classes_sub)
       
    cv_list = [(train, test, k) for k, (train, test) in enumerate(cv)]

    cv_item = cv_list[1]
    train, test = cv_item[0], cv_item[1]

    # get train & test pairs and labels according to K-fold division indexs
    train_df = pd.DataFrame(list(zip(pairs_sub[train, 0], pairs_sub[train, 1], classes_sub[train])), columns=['Drug1', 'Drug2', 'Class'])
    test_df = pd.DataFrame(list(zip(pairs_sub[test, 0], pairs_sub[test, 1], classes_sub[test])), columns=['Drug1', 'Drug2', 'Class'])

    # concatenate embeddings of Drug1 & Drug2
    emb_train_df = train_df.merge(embedding_df, left_on = 'Drug1', right_on = 'Drug').merge(embedding_df, left_on = 'Drug2', right_on = 'Drug')
    emb_test_df = test_df.merge(embedding_df, left_on = 'Drug1', right_on = 'Drug').merge(embedding_df, left_on = 'Drug2', right_on = 'Drug')
    
    # extract feature columns indexs
    features_cols = emb_train_df.columns.difference(['Drug1', 'Drug2' ,'Class', 'Drug_x', 'Drug_y'])
    train_X = emb_train_df[features_cols].values
    train_y = emb_train_df['Class'].values.ravel()
    test_X = emb_test_df[features_cols].values
    test_y = emb_test_df['Class'].values.ravel()
    
    batch_size = 128
    num_classes = 2

    train_data = DDIdataset(train_X, train_y)
    train_loader = DataLoader(train_data, batch_size = batch_size, shuffle = True, num_workers = 2)
    test_data = DDIdataset(test_X, test_y)
    test_loader = DataLoader(test_data, batch_size = batch_size, num_workers = 2)

    input_shape = batch_size, 1, train_X.shape[1]
    lr = 0.003

    model = OrgConvLSTM(num_classes, input_shape).to(device)
    if device.type == 'cuda':
        model = torch.nn.DataParallel(model)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.RMSprop(model.parameters(), lr = lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = 3, gamma = 0.975)
    
    epochs = 100

    results = pd.DataFrame()
    for epoch in range(epochs):
        train_loss = 0
        start_time = time.time()
        model.train()
        for index, data in enumerate(train_loader):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs.float())
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            if index % 100 == 99:
                print('Epoch %d, index %d, loss: %.4f' % (epoch + 1, index + 1, train_loss / 100))
                train_loss = 0
  
        end_time = time.time()

        model.eval()
        with torch.no_grad():
            #y_scores_1, y_scores_2 = list(), list()
            y_pred_all, y_prob_all = list(), list()
            
            for data in test_loader:
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)
      
                outputs = model(inputs.float())
                y_prob = F.softmax(outputs, dim = 1).cpu().detach().numpy().tolist()
                y_prob_all.extend(y_prob)

                conf, y_pred = torch.max(outputs.detach(), 1)
                y_pred_all.extend(y_pred.cpu().detach().numpy().tolist())
    
            y_true = test_loader.dataset.y
            y_pred_all = np.array(y_pred_all)

            print('-----Result-----')
            acc = metrics.accuracy_score(y_true, y_pred_all)
            print('Accuracy: %.4f' % acc)

            precision = metrics.precision_score(y_true, y_pred_all)
            recall = metrics.recall_score(y_true, y_pred_all)
            print('Precision: %.4f' % precision)
            print('Recall: %.4f' % recall)

    
            y_probs = np.array(y_prob_all)[:,1]
            roc_auc = metrics.roc_auc_score(y_true, y_probs)
            print('ROC AUC: %.4f' % roc_auc)

            f1 = metrics.f1_score(y_true, y_pred_all)
            print('F1 score: %.4f' % f1)

            mcc = metrics.matthews_corrcoef(y_true, y_pred_all)
            print('MCC: %.4f' % mcc)

            precisions, recalls, thresholds = metrics.precision_recall_curve(y_true, y_probs)
            aupr = metrics.auc(recalls, precisions)
            print('AUPR: %.4f' % aupr)

            scores = dict()
            scores['precision'] = precision
            scores['recall'] = recall
            scores['accuracy'] = acc
            scores['roc_auc'] = roc_auc
            scores['f1_score'] = f1
            scores['mcc'] = mcc
            scores['aupr'] = aupr
            scores_df = pd.DataFrame([list(scores.values())], columns = list(scores.keys()))
            results = pd.concat([results, scores_df], ignore_index = True)

        scheduler.step()
    
    exportRes(results, dst_filename)


if __name__ == "__main__":
    embedding_files = ['Entity2Vec_cbow_200_5_5_2_500_d5_randwalks.txt',
                       'Entity2Vec_sg_200_5_5_15_2_500_d5_randwalks.txt',
                       'rotate_drug_embeddings.txt',
                       'simple_h_drug_embeddings.txt']
    dst_files = ['rdf2vec_cbow_org-conv-lstm_result.csv',
                 'rdf2vec_sg_org-conv-lstm_result.csv',
                 'rotate_org-conv-lstm_result.txt',
                 'simple_h_org-conv-lstm_result.txt']
    
    for i in range(len(embedding_files)):
        feature_filename = embedding_files[i]
        dst_filename = dst_files[i]
        print('----------Using %s----------' % feature_filename)
        run(feature_filename, dst_filename)
        print('------------Done------------')
        
    
    
    