# -*- coding: utf-8 -*-
"""
Created on Wed May  4 22:43:12 2022
@author: QI YU
@email: yq123456leo@outlook.com
"""

import os
import sys
import numpy as np
from sklearn.model_selection import StratifiedKFold

import prepare_data as prep
from train import train
from ind_cv import IndCVGenor
import utils

sys.path.append(os.getcwd()) # Add the env path

def k_fold_cv(all_data, k_fold):
    ddi_pairs, labels = all_data
    
    # Generate K-fold
    skf_model = StratifiedKFold(n_splits = k_fold, shuffle = True)
    cv_model = skf_model.split(ddi_pairs, labels)
    
    cv_list = list()
    for k, (train, test) in enumerate(cv_model):
        valid, test = np.split(test, [test.shape[0] // 2])
        cv_list.append((k, train, valid, test))
    
    return cv_list

def inductive_cv(ddi_file, k_fold):
    id2entity = utils.pickle_load(os.path.join('temp_data', 'id2entity.pkl'))
    entity2id = {id2entity[eid]:eid for eid in id2entity}
    namespace = 'http://bio2rdf.org/drugbank:'
    
    cv_genor = IndCVGenor(ddi_file, k_fold)
    cv_list = list()
    for idx in range(k_fold):
        train_df, semi_ind_df, ind_df = cv_genor.generate_df(idx = idx)
        ddi_dfs = (train_df, semi_ind_df, ind_df)
        ddis_list = list()
        for ddi_df in ddi_dfs:
            ddi_mat = list()
            ddis = ddi_df.values
            np.random.shuffle(ddis)
            for ddi in ddis:
                new_ddi = [entity2id['<' + namespace + ddi[0] + '>'], entity2id['<' + namespace + ddi[1] + '>'], ddi[2]]
                ddi_mat.append(new_ddi)
            ddis_list.append(np.array(ddi_mat))
        train, valid = np.split(ddis_list[0], [(ddis_list[0].shape[0] // 9 + 1) * 8])
        cv_list.append((idx, train, valid, ddis_list[1], ddis_list[2]))
        
    return cv_list

def run_tradition(aggregator_type = 'sum', k_fold = 5, dataset = 'drugbank', neighbor_size = 4):
    #prep.read_kg('kg')
    #prep.generateAdjMat()
    
    ddi_name = 'ddi_full.txt'
    ddi_pairs, labels = prep.generate_pairs(ddi_name, proportion = 0.25)
    ddi_pairs, labels = prep.balance_pairs((ddi_pairs, labels), proportion = 1)
    
    cv_list = k_fold_cv((ddi_pairs, labels), k_fold)
    ddis = np.concatenate((ddi_pairs, labels.reshape((-1, 1))), axis = -1)
    
    temp = {'dataset': dataset, 'aggregator_type': aggregator_type, 'avg_auc': 0.0, 'avg_acc': 0.0,
            'avg_f1': 0.0, 'avg_aupr': 0.0, 'avg_precision': 0.0, 'avg_recall': 0.0, 'avg_mcc': 0.0}
    for cv in cv_list:
        n_fold, train_index, valid_index, test_index = cv
        train_data, valid_data, test_data = ddis[train_index], ddis[valid_index], ddis[test_index]
        train_log = train(k_fold = n_fold, 
                          dataset = dataset,
                          train_data = train_data,
                          valid_data = valid_data,
                          test_data = test_data,
                          neighbor_sample_size = neighbor_size,
                          embed_dim = 32,
                          n_depth = 2,
                          l2_weight = 1e-7,
                          lr = 2e-2,
                          optim_type = 'adam',
                          batch_size = 2048,
                          aggre_type = aggregator_type,
                          n_epoch = 50,
                          callbacks_to_add = ['modelcheckpoint', 'earlystopping']) 
        
        temp['avg_auc'] = temp['avg_auc'] + train_log['test_auc']
        temp['avg_acc'] = temp['avg_acc'] + train_log['test_acc']
        temp['avg_f1'] = temp['avg_f1'] + train_log['test_f1']
        temp['avg_aupr'] = temp['avg_aupr'] + train_log['test_aupr']
        temp['avg_precision'] = temp['avg_precision'] + train_log['test_precision']
        temp['avg_recall'] = temp['avg_recall'] + train_log['test_recall']
        temp['avg_mcc'] = temp['avg_mcc'] + train_log['test_mcc']
    
    for key in temp:
        if key=='aggregator_type' or key=='dataset':
            continue
        temp[key] = temp[key] / k_fold
    utils.write_log(os.path.join('log', f'{dataset}_result.txt'), temp, 'a')
    print(f'Logging Info - {k_fold} fold result: avg_auc: {temp["avg_auc"]}, avg_acc: {temp["avg_acc"]}, avg_f1: {temp["avg_f1"]}, '\
          f'avg_aupr: {temp["avg_aupr"]}, avg_precision: {temp["avg_precision"]}, avg_recall: {temp["avg_recall"]}, avg_mcc: {temp["avg_mcc"]}')

def run_inductive(aggregator_type = 'sum', k_fold = 5, dataset = 'drugbank', neighbor_size = 4):
    ddi_name = 'ddi_full.txt'
    
    cv_list = inductive_cv(os.path.join('ddi', ddi_name), k_fold)
    
    temp = {'dataset': dataset, 'aggregator_type': aggregator_type, 'avg_auc': 0.0, 'avg_acc': 0.0,
            'avg_f1': 0.0, 'avg_aupr': 0.0, 'avg_precision': 0.0, 'avg_recall': 0.0, 'avg_mcc': 0.0}
    for cv in cv_list:
        n_fold, train_data, valid_data, semi_ind_data, ind_data = cv
        test_data = semi_ind_data
        train_log = train(k_fold = n_fold, 
                          dataset = dataset,
                          train_data = train_data,
                          valid_data = valid_data,
                          test_data = test_data,
                          neighbor_sample_size = neighbor_size,
                          embed_dim = 32,
                          n_depth = 2,
                          l2_weight = 1e-7,
                          lr = 2e-2,
                          optim_type = 'adam',
                          batch_size = 2048,
                          aggre_type = aggregator_type,
                          n_epoch = 50,
                          callbacks_to_add = ['modelcheckpoint', 'earlystopping']) 
        
        temp['avg_auc'] = temp['avg_auc'] + train_log['test_auc']
        temp['avg_acc'] = temp['avg_acc'] + train_log['test_acc']
        temp['avg_f1'] = temp['avg_f1'] + train_log['test_f1']
        temp['avg_aupr'] = temp['avg_aupr'] + train_log['test_aupr']
        temp['avg_precision'] = temp['avg_precision'] + train_log['test_precision']
        temp['avg_recall'] = temp['avg_recall'] + train_log['test_recall']
        temp['avg_mcc'] = temp['avg_mcc'] + train_log['test_mcc']
    
    for key in temp:
        if key=='aggregator_type' or key=='dataset':
            continue
        temp[key] = temp[key] / k_fold
    utils.write_log(os.path.join('log', f'{dataset}_result.txt'), temp, 'a')
    print(f'Logging Info - {k_fold} fold result: avg_auc: {temp["avg_auc"]}, avg_acc: {temp["avg_acc"]}, avg_f1: {temp["avg_f1"]}, '\
          f'avg_aupr: {temp["avg_aupr"]}, avg_precision: {temp["avg_precision"]}, avg_recall: {temp["avg_recall"]}, avg_mcc: {temp["avg_mcc"]}')

if __name__ == "__main__":
    run_tradition()

        