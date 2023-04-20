# -*- coding: utf-8 -*-
"""
Created on Fri May 13 19:35:31 2022
@author: QI YU
@email: yq123456leo@outlook.com
"""

import os
import pandas as pd
import numpy as np
from itertools import combinations

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
        

if __name__ == "__main__":
    ddi_dir = ''
    ddi_file = 'ddi_drugbank_v5.1.9_all.txt'
    ddi_path = os.path.join(ddi_dir, ddi_file)
    
    cv_genor = IndCVGenor(ddi_path)
    train_df, semi_ind_df, ind_df = cv_genor.generate_df(idx = 0)
            
        