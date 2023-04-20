# -*- coding: utf-8 -*-
"""
Created on Sun Apr 17 22:03:11 2022
@author: QI YU
@email: yq123456leo@outlook.com
"""

import torch
import torch.nn as nn

def test1():
    m = nn.Conv1d(16, 33, 3, stride=2)
    inputs = torch.randn(20, 16, 50)
    outputs = m(inputs)
    print(outputs.shape)

def test2():
    pass


if __name__ == "__main__":
    test1()