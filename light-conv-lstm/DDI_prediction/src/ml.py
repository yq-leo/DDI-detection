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

from sklearn.model_selection import StratifiedKFold, cross_validate

import networkx as nx
import random
import numbers


def multimetric_score(estimator, X_test, y_test, scorers):
    """Return a dict of score for multimetric scoring"""
    scores = {}
    for name, scorer in scorers.items():
        if y_test is None:
            score = scorer(estimator, X_test)
        else:
            score = scorer(estimator, X_test, y_test)

        if hasattr(score, 'item'):
            try:
                # e.g. unwrap memmapped scalars
                score = score.item()
            except ValueError:
                # non-scalar?
                pass
        scores[name] = score

        if not isinstance(score, numbers.Number):
            raise ValueError("scoring must return a number, got %s (%s) "
                             "instead. (scorer=%s)"
                             % (str(score), type(score), name))
    return scores

def generatePairs(ddi_df, embedding_df):
    
    drugs = set(ddi_df.Drug1.unique())
    drugs = drugs.union(ddi_df.Drug2.unique())
    drugs = drugs.intersection(embedding_df.Drug.unique())

    ddiKnown = set([tuple(x) for x in ddi_df[['Drug1','Drug2']].values])

    pairs = list()
    classes = list()

    for dr1, dr2 in itertools.combinations(sorted(drugs), 2):
        if dr1 == dr2: continue

        if (dr1, dr2) in ddiKnown or (dr2, dr1) in ddiKnown: 
            cls=1  
        else:
            cls=0

        pairs.append((dr1, dr2))
        classes.append(cls)

    pairs = np.array(pairs)        
    classes = np.array(classes)
    
    return pairs, classes

def balance_data(pairs, classes, n_proportion):
    classes = np.array(classes)
    pairs = np.array(pairs)
    
    indices_true = np.where(classes == 1)[0]
    indices_false = np.where(classes == 0)[0]

    np.random.shuffle(indices_false)
    indices = indices_false[:(n_proportion*indices_true.shape[0])]
    print ("+/-:", len(indices_true), len(indices), len(indices_false))
    pairs = np.concatenate((pairs[indices_true], pairs[indices]), axis = 0)
    classes = np.concatenate((classes[indices_true], classes[indices]), axis = 0)
    
    return pairs, classes

def get_scores(clf, X_new, y_new):
    scoring = ['precision', 'recall', 'accuracy', 'roc_auc', 'f1', 'average_precision']
    scorers = metrics._scorer._check_multimetric_scoring(clf, scoring = scoring)

    scores = multimetric_score(clf, X_new, y_new, scorers)
    return scores

def get_scores_new(clf, X_new, y_new):
    scoring = ['precision', 'recall', 'accuracy', 'roc_auc', 'f1', 'average_precision']
    scores = metrics.check_scoring(clf, scoring = 'precision')
    print(scores)

def crossvalid(train_df, test_df, clfs, run_index, fold_index): 
    features_cols = train_df.columns.difference(['Drug1','Drug2' ,'Class', 'Drug_x', 'Drug_y'])     # extract feature columns indexs
    X = train_df[features_cols].values                                                              # extract features => matrix
    y = train_df['Class'].values.ravel()                                                            # extract labels => vector

    X_new = test_df[features_cols].values
    y_new = test_df['Class'].values.ravel()

    results = pd.DataFrame()
    for name, clf in clfs:
        clf.fit(X, y)
        scores = get_scores(clf, X_new, y_new)
        
        # test traditional method
        #y_pred = clf.predict(X_new)
        
        scores['method'] = name
        scores['fold'] = fold_index
        scores['run'] = run_index
        scores_df = pd.DataFrame([list(scores.values())], columns = list(scores.keys()))
        results = pd.concat([results, scores_df], ignore_index = True)
        #results = results.append(scores, ignore_index=True)

    return results

def cv_run(run_index, pairs, classes, embedding_df, train, test, fold_index, clfs):
    print('run %d: train samples: %d, test samples: %d' % (run_index, len(train), len(test)))
    
    # get train & test pairs and labels according to K-fold division indexs
    train_df = pd.DataFrame(list(zip(pairs[train, 0], pairs[train, 1], classes[train])), columns=['Drug1', 'Drug2', 'Class'])
    test_df = pd.DataFrame(list(zip(pairs[test, 0], pairs[test, 1], classes[test])), columns=['Drug1', 'Drug2', 'Class'])
    
    # concatenate embeddings of Drug1 & Drug2
    emb_train_df = train_df.merge(embedding_df, left_on = 'Drug1', right_on = 'Drug').merge(embedding_df, left_on = 'Drug2', right_on = 'Drug')
    emb_test_df = test_df.merge(embedding_df, left_on = 'Drug1', right_on = 'Drug').merge(embedding_df, left_on = 'Drug2', right_on = 'Drug')
    
    all_scores = crossvalid(emb_train_df, emb_test_df, clfs, run_index, fold_index)

    return all_scores

def cv_run_all(run_index, pairs, classes, cv, embedding_df, clfs):
    all_scores = pd.DataFrame()
    for cv_item in cv:
        run_scores = cv_run(run_index, pairs, classes, embedding_df, cv_item[0], cv_item[1], cv_item[2], clfs)
        all_scores = pd.concat([all_scores, run_scores], ignore_index = True)
    
    return all_scores

def kfoldCV(pairs_all, classes_all, embedding_df, clfs, n_run, n_fold, n_proportion,  n_seed):
    scores_df = pd.DataFrame()
    #bc_embedding_df = sc.broadcast(embedding_df)
    for r in range(n_run): 
        n_seed += r
        random.seed(n_seed)
        np.random.seed(n_seed)
        n_proportion = 1
        pairs, classes= balance_data(pairs_all, classes_all, n_proportion)
        
        # Generate K-fold
        skf = StratifiedKFold(n_splits = n_fold, shuffle = True, random_state = n_seed)
        cv = skf.split(pairs, classes)
       
        print ('run', r)
        cv_list = [(train, test, k) for k, (train, test) in enumerate(cv)]
        scores = cv_run_all(r, pairs, classes, cv_list, embedding_df, clfs)
        scores_df = pd.concat([scores_df, scores], ignore_index = True)
        
    return scores_df


