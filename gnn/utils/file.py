# -*- coding: utf-8 -*-
"""
Created on Wed May  4 14:56:53 2022
@author: QI YU
@email: yq123456leo@outlook.com
"""

import pickle
import json
import numpy as np

def pickle_dump(filename, obj):
    with open(filename, 'wb') as f:
        pickle.dump(obj, f)
    print('File saved: %s' % filename)
    
def pickle_load(filename):
    try:
        with open(filename, 'rb') as f:
            obj = pickle.load(f)
        print('File loaded: %s' % filename)
    except EOFError:
        print('Cannot load %s' % filename)
        obj = None
    return obj

def npy_dump(filename, obj):
    np.save(filename, obj)
    print('File saved: %s' % filename)

def npy_load(filename):
    try:
        obj = np.load(filename)
        print('File loaded: %s' % filename)
    except EOFError:
        print('Cannot load %s' % filename)
        obj = None
    return obj

def write_log(filename, log, mode = 'w'):
    with open(filename, mode) as writers:
        writers.write('\n')
        json.dump(log, writers, indent = 4, ensure_ascii = False)