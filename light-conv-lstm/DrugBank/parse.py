# -*- coding: utf-8 -*-
"""
Created on Thu Mar 31 15:48:44 2022
@author: QI YU
@email: yq123456leo@outlook.com
"""

import os
import csv
import gzip
import collections
import re
import io
import json
import xml.etree.ElementTree as ET
from collections import OrderedDict, defaultdict

import requests
import pandas as pd
import numpy as np

from sklearn.utils import shuffle

class OrderedSet():
  def __init__(self, ls = list()):
    self.ord_dict = OrderedDict()
    for item in ls:
      self.ord_dict[item] = None
  def add(self, item):
    self.ord_dict[item] = None
  def union(self, ord_set):
    self.ord_dict.update(ord_set.ord_dict)
  def intersection(self, ord_set):
    int_dict = OrderedDict()
    for key in self.ord_dict:
      if key in ord_set.ord_dict:
        int_dict[key] = self.ord_dict[key]
    self.ord_dict = int_dict
  def __repr__(self):
    head = 'orderedset'
    elems = ''
    for key in self.ord_dict:
      elems += str(key) + ', '
    context = head + '({' + elems[:-2] + '})'
    return context
  def __iter__(self):
    self.iter = iter(self.ord_dict)
    return self
  def __next__(self):
    return next(self.iter)

def ParseDBddi():
    full_xml_path = os.path.join('data', 'full_drugbank.xml')

    tags_stack, info_stack = list(), list()
    parallel_attrs = defaultdict(list)
    is_start, flag = True, True
    ddi_set, ddi_num = OrderedSet(), 0

    num = 0
    ns = "{http://www.drugbank.ca}"
    last_db_id = 'DB00001'
    for event, elem in ET.iterparse(full_xml_path, events=('start', 'end')):
        tag = elem.tag.rsplit('}', 1)[-1].strip()

        if event == 'start':
            if tag != 'drugbank':
                tags_stack.append(tag)
                if is_start:
                    parallel_attrs = defaultdict(list)
                info_stack.append(parallel_attrs.copy())
            is_start = True

        elif event == 'end':
            if tag == 'drugbank':
                break
            
            if is_start:
                info_stack[-1][tag].append(elem.text)
            else:
                info_stack[-1][tag].append(parallel_attrs) 
        
            del tags_stack[-1]
            parallel_attrs = info_stack.pop(-1)
            is_start = False
            elem.clear()
    
        if len(tags_stack) == 0:
            if flag:
                flag = False
                
            else:
                drugbank_id = parallel_attrs['drug'][0]['drugbank-id'][0]
                idx = int(drugbank_id[2:])
                num += 1
                
                if (idx - 1) % 1000 == 0:
                    last_db_id = drugbank_id
                elif (idx - 1) % 1000 == 999:
                    print('%s - %s detected' % (last_db_id, drugbank_id))
                
                interact_info = parallel_attrs['drug'][0]['drug-interactions'][0]
                if interact_info is not None:
                    drug_pairs = OrderedSet()
                    ddi_num += len(interact_info['drug-interaction'])
                    for item in interact_info['drug-interaction']:
                        drug2 = item['drugbank-id'][0]
                        drug_pairs.add((drugbank_id, drug2))
                    ddi_set.union(drug_pairs)
            
                is_start = True

    print("%d drugs deteced" % num)
    ddi_df = pd.DataFrame(ddi_set, columns = ['Drug1', 'Drug2'])
    return ddi_df

def ShuffleDDi(filename):
    dirname = 'data/DDI'
    ddi_df = pd.read_csv(os.path.join(dirname, filename), sep = '\t')
    ddi_df = shuffle(ddi_df)
    ddi_df.reset_index(inplace = True, drop = True)
    ddi_df.to_csv(os.path.join(dirname, filename), sep = '\t', index = 0)

def DDIUnion(filenames):
    dirname = 'data/DDI'
    
    ddi_full_set = set()
    for filename in filenames:
        ddi_df = pd.read_csv(os.path.join(dirname, filename), sep = '\t')
        for pair in ddi_df.values:
            ddi_full_set.add(tuple(pair))
    
    ddi_full_mat = list()
    for pair in ddi_full_set:
        ddi_full_mat.append(list(pair))
    ddi_full_mat = np.array(ddi_full_mat)
    
    ddi_full_df = pd.DataFrame(ddi_full_mat, columns = ['Drug1', 'Drug2'])
    ddi_full_df.to_csv(os.path.join(dirname, 'ddi_full.txt'), sep = '\t', index = 0)

if __name__ == "__main__":
    #ShuffleDDi('ddi_twosides.txt')
    
    filenames = ['ddi_drugbank_v5.1.9_all.txt', 'ddi_twosides.txt']
    DDIUnion(filenames)
    
    