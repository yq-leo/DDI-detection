# -*- coding: utf-8 -*-
"""
Created on Wed May 11 21:39:59 2022
@author: QI YU
@email: yq123456leo@outlook.com
"""

import numpy as np
import os
import time
import gc
from keras import optimizers
from keras import backend as K

from config import ModelConfig
import utils
from models import KGCN

def get_optimizer(op_type, learning_rate):
    if op_type == 'sgd':
        return optimizers.SGD(learning_rate)
    elif op_type == 'rmsprop':
        return optimizers.RMSprop(learning_rate)
    elif op_type == 'adagrad':
        return optimizers.Adagrad(learning_rate)
    elif op_type == 'adadelta':
        return optimizers.Adadelta(learning_rate)
    elif op_type == 'adam':
        return optimizers.adam_v2.Adam(learning_rate, clipnorm=5)
    else:
        raise ValueError('Optimizer Not Understood: {}'.format(op_type))


def train(train_data, valid_data, test_data, k_fold, dataset, neighbor_sample_size,
          embed_dim, n_depth, l2_weight, lr, optim_type, batch_size, aggre_type,
          n_epoch, callbacks_to_add = None, overwrite=True):
    # Configurations
    config = ModelConfig()
    config.neighbor_sample_size = neighbor_sample_size
    config.embed_dim = embed_dim
    config.n_depth = n_depth
    config.l2_weight = l2_weight
    config.dataset = dataset
    config.K_Fold = k_fold
    config.lr = lr
    config.optimizer = get_optimizer(optim_type, lr)
    config.batch_size = batch_size
    config.aggregator_type = aggre_type
    config.n_epoch = n_epoch
    config.callbacks_to_add = callbacks_to_add
    
    temp_data_dir = 'temp_data'
    config.drug_vocab_size = len(utils.pickle_load(os.path.join(temp_data_dir, 'id2entity.pkl')))
    config.entity_vocab_size = len(utils.pickle_load(os.path.join(temp_data_dir, 'id2entity.pkl')))
    config.relation_vocab_size = len(utils.pickle_load(os.path.join(temp_data_dir, 'id2relation.pkl')))
    
    config.adj_entity = utils.npy_load(os.path.join(temp_data_dir, 'adj_entity.npy'))
    config.adj_relation = utils.npy_load(os.path.join(temp_data_dir, 'adj_relation.npy'))
    
    config.exp_name = f'kgcn_{dataset}_neigh_{neighbor_sample_size}_embed_{embed_dim}_depth_' \
                      f'{n_depth}_agg_{aggre_type}_optimizer_{optim_type}_lr_{lr}_' \
                      f'batch_size_{batch_size}_epoch_{n_epoch}'
    callback_str = '_' + '_'.join(config.callbacks_to_add)
    callback_str = callback_str.replace('_modelcheckpoint', '').replace('_earlystopping', '')
    config.exp_name += callback_str
    
    # Training log
    train_log = {'exp_name': config.exp_name, 'batch_size': batch_size, 'optimizer': optim_type,
                 'epoch': n_epoch, 'learning_rate': lr}
    print('Logging Info - Experiment: %s' % config.exp_name)
    model_save_path = os.path.join(config.checkpoint_dir, '{}.hdf5'.format(config.exp_name))
    
    # Graph Neural Network
    model = KGCN(config)
    
    #train_data = np.array(train_data)
    #valid_data = np.array(valid_data)
    #test_data = np.array(test_data)
    if not os.path.exists(model_save_path) or overwrite:
        start_time = time.time()
        model.fit(x_train = [train_data[:, :1], train_data[:, 1:2]], y_train = train_data[:, 2:3],
                  x_valid = [valid_data[:, :1], valid_data[:, 1:2]], y_valid = valid_data[:, 2:3])
        elapsed_time = time.time() - start_time
        print('Logging Info - Training time: %s' % time.strftime("%H:%M:%S",
                                                                 time.gmtime(elapsed_time)))
        train_log['train_time'] = time.strftime("%H:%M:%S", time.gmtime(elapsed_time))

    # Validation
    print('Logging Info - Evaluate over valid data:')
    model.load_best_model()
    auc, acc, f1, aupr, precision, recall, mcc = model.score(x = [valid_data[:, :1], valid_data[:, 1:2]], 
                                                             y = valid_data[:, 2:3])

    print(f'Logging Info - valid_auc: {auc}, valid_acc: {acc}, valid_f1: {f1}, valid_aupr: {aupr}, '\
          f'valid_precision: {precision}, valid_recall: {recall}, valid_mcc: {mcc}')
    train_log['valid_auc'] = auc
    train_log['valid_acc'] = acc
    train_log['valid_f1'] = f1
    train_log['valid_aupr'] = aupr
    train_log['valid_precision'] = precision
    train_log['valid_recall'] = recall
    train_log['valid_mcc'] = mcc
    
    train_log['k_fold'] = k_fold
    train_log['dataset'] = dataset
    train_log['aggregate_type'] = config.aggregator_type
    if 'swa' in config.callbacks_to_add:
        model.load_swa_model()
        print('Logging Info - Evaluate over valid data based on swa model:')
        auc, acc, f1, aupr, precision, recall, mcc = model.score(x = [valid_data[:, :1], valid_data[:, 1:2]], 
                                                                 y = valid_data[:, 2:3])
        train_log['swa_valid_auc'] = auc
        train_log['swa_valid_acc'] = acc
        train_log['swa_valid_f1'] = f1
        train_log['swa_valid_aupr'] = aupr
        train_log['swa_valid_precision'] = precision
        train_log['swa_valid_recall'] = recall
        train_log['swa_valid_mcc'] = mcc
        
        print(f'Logging Info - swa_valid_auc: {auc}, swa_valid_acc: {acc}, swa_valid_f1: {f1}, swa_valid_aupr: {aupr}, ' \
              f'swa_valid_precision: {precision}, swa_valid_recall: {recall}, swa_valid_mcc: {mcc}')
        
    # Testing
    print('Logging Info - Evaluate over test data:')
    model.load_best_model()
    auc, acc, f1, aupr, precision, recall, mcc = model.score(x = [test_data[:, :1], test_data[:, 1:2]], 
                                                             y = test_data[:, 2:3])
    train_log['test_auc'] = auc
    train_log['test_acc'] = acc
    train_log['test_f1'] = f1
    train_log['test_aupr'] = aupr
    train_log['test_precision'] = precision
    train_log['test_recall'] = recall
    train_log['test_mcc'] = mcc
    
    print(f'Logging Info - test_auc: {auc}, test_acc: {acc}, test_f1: {f1}, test_aupr: {aupr}, '\
          f'test_precision: {precision}, test_recall: {recall}, test_mcc: {mcc}')
    if 'swa' in config.callbacks_to_add:
        model.load_swa_model()
        print('Logging Info - Evaluate over test data based on swa model:')
        auc, acc, f1, aupr, precision, recall, mcc = model.score(x = [test_data[:, :1], test_data[:, 1:2]], 
                                                                 y = test_data[:, 2:3])
        train_log['swa_test_auc'] = auc
        train_log['swa_test_acc'] = acc
        train_log['swa_test_f1'] = f1
        train_log['swa_test_aupr'] = aupr
        train_log['swa_test_precision'] = precision
        train_log['swa_test_recall'] = recall
        train_log['swa_test_mcc'] = mcc
        
        print(f'Logging Info - swa_test_auc: {auc}, swa_test_acc: {acc}, swa_test_f1: {f1}, swa_test_aupr: {aupr}, '\
              f'swa_test_precision: {precision}, swa_test_recall: {recall}, swa_test_mcc: {mcc}')
            
    train_log['timestamp'] = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
    if not os.path.exists('log'):
        os.mkdir('log')
    utils.write_log(os.path.join('log', 'gnn_performance.log'), log = train_log, mode = 'a')
    del model
    
    gc.collect()
    K.clear_session()
    return train_log
    
    