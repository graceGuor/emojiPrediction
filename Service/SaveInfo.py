# -*- coding: UTF-8 -*-
from __future__ import print_function
from sklearn.metrics import f1_score
import os
import numpy as np
import sys
import tensorflow as tf

import time
import glob

# 加载id特征向量文件至dict(embeddings_index){word:embedding}
def loadFeatures(feaFile):
    print('Indexing word vectors.')
    labels = []  # list of label ids
    ids = []
    feaVec = []
    f = open(feaFile)
    for line in f:
        values = line.split()
        # if values[0] == '0':
        #     labels.append([1, 0])
        # else:
        #     labels.append([0, 1])
        if values[0] == '0':
            labels.append([0])
        else:
            labels.append([1])
        ids.append(values[1])
        coefs = np.asarray(values[2:], dtype='float32')
        feaVec.append(coefs)
    f.close()
    return np.array(labels),ids,np.array(feaVec)