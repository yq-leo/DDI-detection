# -*- coding: utf-8 -*-
"""
Created on Mon May 30 00:29:39 2022
@author: surface
"""


import numpy as np
import pandas as pd
import os

dirname = 'inductive_result'
filename = 'rotate_nb_lr_rf_semi-ind_result.csv'
dst_dir = 'summary'

res_df = pd.read_csv(os.path.join(dirname, filename))

from collections import defaultdict

methods = list(res_df.method.unique())
evaluations = list(res_df.columns)[1:-3]

stat = {method:defaultdict(list) for method in methods}
for i in range(res_df.shape[0]):
  line = res_df.loc[i]
  method = line['method']
  for eva in evaluations:
    stat[method][eva].append(line[eva])

stat_mat = list()
for key in stat:
  sub_mat = [key]
  for eva in evaluations:
    data_vec = np.array(stat[key][eva])
    avg = data_vec.mean()
    sub_mat.append(round(avg, 4))
  stat_mat.append(sub_mat)
stat_mat = np.array(stat_mat)
columns = ['method'] + evaluations
stat_df = pd.DataFrame(stat_mat, columns = columns)

dst_file = filename[:-4] + '_summary.csv'
stat_df.to_csv(os.path.join(dst_dir, dst_file))