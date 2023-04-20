# -*- coding: utf-8 -*-
"""
Created on Wed Mar 23 18:41:20 2022
@author: QI YU
@email: yq123456leo@outlook.com
"""

import csv
import numpy as np
import sys
import pandas as pd
import itertools
import math
import time

from sklearn import svm, linear_model, neighbors
from sklearn import tree, ensemble
from sklearn import metrics
from sklearn.naive_bayes import GaussianNB

import networkx as nx
import random
import numbers

from sklearn.model_selection import StratifiedKFold

from src import ml

# Load DDI
ddi_df = pd.read_csv('data/input/ddi_v4.txt', sep = '\t')
ddi_df.head()

# Load RDF2Vec SG embedding
featureFileName = 'vectors/DB/Entity2Vec_sg_200_5_5_15_2_500_d5_randwalks.txt'
embedding_df = pd.read_csv(featureFileName, delimiter = '\t')
#embedding_df.Entity = embedding_df.Entity.str[-8:-1]
embedding_df.Drug = embedding_df.Drug.str[-8:-1]
#embedding_df.rename(columns = {'Entity':'Drug'}, inplace = True)

pairs, classes = ml.generatePairs(ddi_df, embedding_df)

# Naive Bayes & Logistic Regression & Random Forest
nb_model = GaussianNB()
lr_model = linear_model.LogisticRegression()
rf_model = ensemble.RandomForestClassifier(n_estimators = 200, n_jobs = -1)
clfs = [('Naive Bayes', nb_model), ('Logistic Regression', lr_model), ('Random Forest', rf_model)]

# Generate K-fold cross-validation set
n_seed = 100
n_fold = 10
n_run = 10          
n_proportion = 1    
all_scores_df = ml.kfoldCV(pairs, classes, embedding_df, clfs, n_run, n_fold, n_proportion, n_seed)



