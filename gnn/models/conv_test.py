# -*- coding: utf-8 -*-
"""
Created on Sat May 28 21:12:24 2022

@author: surface
"""

from keras.layers import *
from keras.regularizers import l2
from keras.models import Model
from keras import backend as K  # use computable function
import numpy as np

drug1_embed, drug2_embed = np.random.rand(1024, 1, 32), np.random.rand(1024, 1, 32)
drug_embed = Concatenate()([drug1_embed, drug2_embed])
drug_reshaped = Reshape((drug_embed.shape[-1], -1))(drug_embed)
drug_conv_out = Conv1D(32, kernel_size = 3, strides = 2, activation = 'relu', input_shape = (64, 1))(drug_reshaped)
drug_drop_out = Dropout(0.5, input_shape = (drug_conv_out.shape[1], drug_conv_out.shape[2]))(drug_conv_out)
drug_maxpool_out = MaxPooling1D(2)(drug_drop_out)
drug_flattened = Flatten()(drug_maxpool_out)
drug_result = Dense(2)(Dense(128, activation = 'relu')(drug_flattened))
drug_drug_score = Softmax()(drug_result)[:, 0:-1]