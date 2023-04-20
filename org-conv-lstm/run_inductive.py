# -*- coding: utf-8 -*-
"""
Created on Thu Jun  2 11:46:53 2022

@author: surface
"""

import csv
import numpy as np
import sys
import pandas as pd
import itertools
from itertools import combinations
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
import torch.backends.cudnn as cudnn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if device.type == 'cuda':
    cudnn.benchmark = True


class IndCVGenor:
    def __init__(self, ddi_file, K = 5, n_proportion = 0.25):
        print('-----Building Inductive Cross-Validation Generator-----')
        self.K = K
        print(f'Cross: {K}')
        
        self.ddi_df = pd.read_csv(ddi_file, sep = '\t')
        drugs = set(self.ddi_df.Drug1.unique()).union(set(self.ddi_df.Drug2.unique()))
        print('Number of Total Drugs: {:,d}'.format(len(drugs)))
        
        v_drugs = np.array(list(drugs))
        np.random.shuffle(v_drugs)
        v_drugs = set(v_drugs[:int(len(v_drugs) * n_proportion)].tolist())
        print('Number of Valid Drugs: {:,d}'.format(len(v_drugs)))
        
        self.id2drug = self.generate_drug_dict(drugs)
        self.cv_input_data = self.generate_cv(v_drugs)
        self.balance_cv()
    
    def generate_drug_dict(self, drugs):
        id2drug = dict()
        for did, drug in enumerate(drugs):
            id2drug[did] = drug
        return id2drug
    
    def generate_cv(self, v_drugs):
        # Generate drug2id & known ddis
        drug2id = dict()
        for did in self.id2drug:
            drug2id[self.id2drug[did]] = did
        ddi_known = set()
        for pair in self.ddi_df[['Drug1','Drug2']].values:
            drug1, drug2 = tuple(pair)
            did1, did2 = drug2id[drug1], drug2id[drug2]
            ddi_known.add((did1, did2))
        
        # Generate cross validation split
        dids = np.array([drug2id[drug] for drug in v_drugs])
        np.random.shuffle(dids)
        
        drug_cv_list = list()
        split_list = np.array_split(dids, self.K)
        for idx, cv_split in enumerate(split_list):
            ind_drugs = cv_split
            train_drugs = np.setdiff1d(dids, ind_drugs)
            drug_cv_list.append([idx, train_drugs, ind_drugs])
        
        # Generate K-fold
        cv_input_data = list()
        for cv in drug_cv_list:
            print(f'Generateing cross {cv[0] + 1} data ...', end = ' ')
            train_drugs, ind_drugs = cv[1], cv[2]
            train_pairs, semi_ind_pairs, ind_pairs = list(), list(), list()
            for did1, did2 in combinations(sorted(dids), 2):
                if did1 == did2:
                    continue
                if (did1, did2) in ddi_known or (did2, did1) in ddi_known:
                    label = 1
                else:
                    label = 0
                
                if did1 in train_drugs and did2 in train_drugs:
                    train_pairs.append((did1, did2, label))
                elif did1 in ind_drugs and did2 in ind_drugs:
                    ind_pairs.append((did1, did2, label))
                else:
                    semi_ind_pairs.append((did1, did2, label))
            
            train_pairs = np.array(train_pairs)
            semi_ind_pairs = np.array(semi_ind_pairs)
            ind_pairs = np.array(ind_pairs)
            print('done, nt: {:,d}(+) {:,d}(-), nsi: {:,d}(+) {:,d}(-), ni: {:,d}(+) {:,d}(-)'.format(train_pairs.sum(axis = 0)[-1], len(train_pairs) - train_pairs.sum(axis = 0)[-1],
                                                                                                      semi_ind_pairs.sum(axis = 0)[-1], len(semi_ind_pairs) - semi_ind_pairs.sum(axis = 0)[-1],
                                                                                                      ind_pairs.sum(axis = 0)[-1], len(ind_pairs) - ind_pairs.sum(axis = 0)[-1]))
            cv_input_data.append([cv[0], train_pairs, semi_ind_pairs, ind_pairs])
        
        return cv_input_data
    
    def balance_cv(self, n_proportion = 1):
        for idx, cv in enumerate(self.cv_input_data):
            idx, pairs_list = cv[0], cv[1:]
            
            print('Balancing cross 1 data ...', end = ' ')
            balanced_pairs_list = list()
            for pairs in pairs_list:
                indices_true = np.where(pairs[:, 2] == 1)[0]
                indices_false = np.where(pairs[:, 2] == 0)[0]
                
                np.random.shuffle(indices_false)
                indices_false = indices_false[:int(n_proportion * indices_true.shape[0])]
                balanced_pairs = np.concatenate((pairs[indices_true], pairs[indices_false]), axis = 0)
                balanced_pairs_list.append(balanced_pairs)
            print('done, nt: {:,d}(+) {:,d}(-), '\
                  'nsi: {:,d}(+) {:,d}(-), '\
                  'ni: {:,d}(+) {:,d}(-)'.format(balanced_pairs_list[0].sum(axis = 0)[-1], len(balanced_pairs_list[0]) - balanced_pairs_list[0].sum(axis = 0)[-1],
                                                 balanced_pairs_list[1].sum(axis = 0)[-1], len(balanced_pairs_list[1]) - balanced_pairs_list[1].sum(axis = 0)[-1],
                                                 balanced_pairs_list[2].sum(axis = 0)[-1], len(balanced_pairs_list[2]) - balanced_pairs_list[2].sum(axis = 0)[-1]))
            
            self.cv_input_data[idx][1:] = balanced_pairs_list
    
    def generate_df(self, idx):
        pairs_list = self.cv_input_data[idx][1:]
        
        df_list = list()
        for pairs in pairs_list:
            rename_pairs = list()
            for pair in pairs:
                rename_pair = [self.id2drug[pair[0]], self.id2drug[pair[1]], pair[2]]
                rename_pairs.append(rename_pair)
            df = pd.DataFrame(rename_pairs, columns = ['Drug1', 'Drug2', 'Class'])
            df_list.append(df)
            
        return df_list


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
    dirname = 'result_inductive'
    if not os.path.isdir(dirname):
        os.mkdir(dirname)

    columns = list(scores_df.columns)
    scores_df.to_csv(os.path.join(dirname, filename), columns = columns)


def run(feature_filename, dst_filename, test_mode):
    # Load DDI
    ddi_filename = 'data/input/ddi_full.txt'
    ddi_df = pd.read_csv(ddi_filename, sep = '\t')
    # Load Embeddings
    #feature_filename = 'vectors/DB/rotate_drug_embeddings.txt'
    embedding_df = pd.read_csv(feature_filename, delimiter = '\t')
    embedding_df.Drug = embedding_df.Drug.str[-8:-1]

    neg_proportion = 1
    emb_proportion = 0.25

    cv_genor = IndCVGenor(ddi_filename)
    train_df, semi_ind_df, ind_df = cv_genor.generate_df(idx = 0)
    
    if test_mode == 'semi-inductive':
        test_df = semi_ind_df
    else:
        test_df = ind_df
    
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
                 'rotate_org-conv-lstm_result.csv',
                 'simple_h_org-conv-lstm_result.csv']
    
    for i in range(len(embedding_files)):
        feature_filename = embedding_files[i]
        dst_filename = dst_files[i]
        print('----------Using %s----------' % feature_filename)
        info = dst_filename.split('_')
        prefix = ''
        for i in range(len(info) - 1):
            prefix += info[i] + '_'
        dst_file1 = prefix + 'semi-ind_' + info[-1]
        dst_file2 = prefix + 'ind_' + info[-1]
        
        run(feature_filename, dst_file1, 'semi-inductive')
        run(feature_filename, dst_file2, 'inductive')
        print('------------Done------------')
    
    