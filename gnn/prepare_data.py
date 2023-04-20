# -*- coding: utf-8 -*-
"""
Created on Tue May  4 10:17:31 2022
@author: QI YU
@email: yq123456leo@outlook.com
"""

import os
import pandas as pd
import numpy as np
from collections import defaultdict
from itertools import combinations

import utils

def read_kg(dirname = 'kg'):
    if not os.path.exists(dirname):
        print('KG does not exist!')
        return
    print('-----Reading KG in dir: %s-----' % dirname)
    
    # Load entities in KG to id2entity dictionary
    print('Loading entities ...', end = ' ')
    id2entity = dict()
    with open(os.path.join(dirname, 'entity2id.txt')) as f:
        n_entity = -1
        for line in f:
            if n_entity == -1:
                n_entity = int(line.strip())
                continue
            entity, eid = line.strip().split(' ')
            id2entity[int(eid)] = entity
    print('done.')
    
    # Load relations in KG to id2relation dictionary
    print('Loading relations ...', end = ' ')
    id2relation = dict()
    with open(os.path.join(dirname, 'relation2id.txt')) as f:
        n_relation = -1
        for line in f:
            if n_relation == -1:
                n_relation = int(line.strip())
                continue
            relation, rid = line.strip().split(' ')
            id2relation[int(rid)] = relation
    print('done.')
    
    # Load triples in KG to adjacency list
    print('Loading KG ...', end = ' ')
    kg = defaultdict(list)
    with open(os.path.join(dirname, 'train2id.txt')) as f:
        n_triple = -1
        for line in f:
            if n_triple == -1:
                n_triple = int(line.strip())
                continue
            hid, tid, rid = line.strip().split(' ')
            # Adjacency list for KG
            kg[int(hid)].append((int(tid), int(rid)))
            kg[int(tid)].append((int(hid), int(rid)))
    print('done.')
    print()
    
    # Statistics
    print('number of entities: {:,d}'.format(len(id2entity)))
    print('number of relation: {:,d}'.format(len(id2relation)))
    print('----------------------------')
    
    temp_dir = 'temp_data'
    if not os.path.exists(temp_dir):
        os.mkdir(temp_dir)
   
    utils.pickle_dump(os.path.join(temp_dir, 'id2entity.pkl'), id2entity)
    utils.pickle_dump(os.path.join(temp_dir, 'id2relation.pkl'), id2relation)
    utils.pickle_dump(os.path.join(temp_dir, 'kg.pkl'), kg)
    print()

def check_temp_data(temp_dir = 'temp_data'):
    if not os.path.exists(temp_dir):
        print('KG not loaded, please run `read_kg` to load kg first.')
        return False
    return True

def read_ddi(filename, dirname = 'ddi'):
    if not os.path.exists(dirname):
        print('DDI matrix does not exist!')
        return None
    ddi_df = pd.read_csv(os.path.join(dirname, filename), sep = '\t')
    print('File loaded: %s' % os.path.join(dirname, filename))
    return ddi_df
 
# Generate valid ddi_pairs for GNN input
def generate_pairs(ddi_name, proportion = 1):
    # Load known DDI pairs
    all_ddi_df = read_ddi(ddi_name)
    ddi_known = set([tuple(x) for x in all_ddi_df[['Drug1','Drug2']].values])
    drugs = set(all_ddi_df.Drug1.unique()).union(set(all_ddi_df.Drug2.unique()))
    
    temp_dir = 'temp_data'
    if not check_temp_data(temp_dir):
        return None
    id2entity = utils.pickle_load(os.path.join(temp_dir, 'id2entity.pkl'))
    
    # Collect drug entities in KG
    namespace = 'http://bio2rdf.org/drugbank:'
    urls = id2entity.values()
    drug_entities = set([url[-8:-1] for url in urls if namespace in url])
    drug_entities= drug_entities.intersection(drugs)
    # Randomly choose proportion of drug entites
    drug_entities = list(drug_entities)
    np.random.shuffle(drug_entities)
    drugs = drugs.intersection(drug_entities[:int(proportion * len(drug_entities))])
    
    # entity2id
    entity2id = {id2entity[eid]:eid for eid in id2entity}
    
    # Generate DDI pairs for training & testing
    print('Generating drug pairs ...', end = ' ')
    ddi_pairs, labels = list(), list()
    for drug1, drug2 in combinations(sorted(drugs), 2):
        if drug1 == drug2:
            continue
        if (drug1, drug2) in ddi_known or (drug2, drug1) in ddi_known:
            label = 1
        else:
            label = 0
        drug1_id, drug2_id = entity2id['<' + namespace + drug1 + '>'], entity2id['<' + namespace + drug2 + '>']
        ddi_pairs.append((drug1_id, drug2_id))
        labels.append(label)
    del entity2id
    print('done.')
    print('{:,d} valid drugs'.format(len(drugs)))
    print('{:,d} drug pairs, positive: {:,d}, negative: {:,d}'.format(len(ddi_pairs), sum(labels), len(labels) - sum(labels)))
    print()
    
    return np.array(ddi_pairs), np.array(labels)

# Balance valid ddi_pairs for GNN input
def balance_pairs(data, proportion = 1):
    print('Balancing positive & negative drug pairs ...', end = ' ')
    ddi_pairs, labels = data
    
    indices_true = np.where(labels == 1)[0]
    indices_false = np.where(labels == 0)[0]

    np.random.shuffle(indices_false)
    indices_false = indices_false[:int(proportion * indices_true.shape[0])]
    
    res_ddi_pairs = np.concatenate((ddi_pairs[indices_true], ddi_pairs[indices_false]), axis = 0)
    res_labels = np.concatenate((labels[indices_true], labels[indices_false]), axis = 0)
    
    res_ddis = np.concatenate((res_ddi_pairs, res_labels.reshape((-1, 1))), axis = -1)
    np.random.shuffle(res_ddis)
    res_ddi_pairs, res_labels = res_ddis[:, :2], res_ddis[:, -1]
    
    print('done.')
    print("pos(+): {:,d}, neg(-): {:,d}".format(len(indices_true), len(indices_false)))
    print()
    
    return res_ddi_pairs, res_labels

def gen_ddi_input(ddi_file, embed_prop = 1, neg_prop = 1):
    temp_dir = 'temp_data'
    if not check_temp_data(temp_dir):
        return
    
    ddi_pairs, labels = generate_pairs(ddi_file, proportion = embed_prop)
    ddi_pairs, labels = balance_pairs((ddi_pairs, labels), proportion = neg_prop)
    
    ddi_input_dir = os.path.join(temp_dir, 'ddi_input')
    if not os.path.exists(ddi_input_dir):
        os.mkdir(ddi_input_dir)
    utils.npy_dump(os.path.join(ddi_input_dir, 'ddi_pairs.npy'), ddi_pairs)
    utils.npy_dump(os.path.join(ddi_input_dir, 'labels.npy'), labels)
    
    ddis = np.concatenate((ddi_pairs, labels.reshape((-1, 1))), axis = -1)
    ddis_df = pd.DataFrame(ddis, columns = ['Drug1', 'Drug2', 'Label'])
    ddis_df.to_csv(os.path.join(ddi_input_dir, 'approved_example.txt'), sep = '\t', header = False, index = False)
    print('File saved: %s' % os.path.join(ddi_input_dir, 'approved_example.txt'))
    
    print()

def generateAdjMat(neighbor_size = 4):
    temp_dir = 'temp_data'
    if not check_temp_data(temp_dir):
        return
    
    id2entity = utils.pickle_load(os.path.join(temp_dir, 'id2entity.pkl'))
    n_entity = len(id2entity)
    adj_entity = np.zeros((n_entity, neighbor_size), dtype = np.int64)
    adj_relation = np.zeros((n_entity, neighbor_size), dtype = np.int64)
    
    kg = utils.pickle_load(os.path.join(temp_dir, 'kg.pkl'))
    print('Generating Adjacency Matrix (neighbor size = %d) ...' % neighbor_size, end = ' ')
    for eid in id2entity:
        all_neighbors = kg[eid]
        n_neighbor = len(all_neighbors)
        sample_indices = np.random.choice(n_neighbor, neighbor_size,
                                          replace = False if n_neighbor >= neighbor_size else True)
        adj_entity[eid] = np.array([all_neighbors[i][0] for i in sample_indices])
        adj_relation[eid] = np.array([all_neighbors[i][1] for i in sample_indices])
    print('done.')
    
    utils.npy_dump(os.path.join(temp_dir, 'adj_entity.npy'), adj_entity)
    utils.npy_dump(os.path.join(temp_dir, 'adj_relation.npy'), adj_relation)

if __name__  == "__main__":
    #read_kg('kg')
    #generateAdjMat()
    ddi_file = 'ddi_full.txt'
    gen_ddi_input(ddi_file, embed_prop = 0.25, neg_prop = 1)
    pass
            