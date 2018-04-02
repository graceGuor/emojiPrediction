# -*- coding: UTF-8 -*-
from __future__ import print_function
import numpy as np
import random
import ptb.conf as conf
from sklearn import preprocessing
import pdb
import csv

#得到文件中句子的最大长度
def getMaxLexOfSeq(filename):
    f = open(filename,'r',encoding='UTF-8')
    maxLenOfSeq = 0
    for line in f:
        values = line.split()
        l = len(values)
        if l > maxLenOfSeq:
            maxLenOfSeq = l
    f.close()
    return  maxLenOfSeq

#得到文件中每个句子的长度
def getLenOfSentences(filename):
    f = open(filename, 'r', encoding='UTF-8')
    lens = []
    for line in f:
        values = line.split()
        l = len(values)
        lens.append(l)
    f.close()
    return lens

#得到文件中句子的平均长度
def getAverageLenOfSentences(filename):
    f = open(filename, 'r', encoding='UTF-8')
    sum_lens = 0
    lines = f.readlines()
    for line in lines:
        values = line.split()
        l = len(values)
        sum_lens = sum_lens + l
    lenOfLines = len(lines)
    f.close()
    averageLen = sum_lens/lenOfLines
    return averageLen

# 加载csv向量文件至dict{word(第一个词):embedding（除第一个词外）}
# 将“×”表示为1
def loadDict_csv(feaFile):
    words = []
    feaVec = []
    lines = csv.reader(open(feaFile, 'r', encoding='utf8'))
    for line in lines:
        line = str(line).replace("'", "").replace('[', '').replace(']', '')
        values = line.split(',')
        words.append(values[0])
        # coefs = np.asarray(values[1:])
        coefs = values[1:]
        feas = []
        for item in coefs:
            if 'X' in item:
                item = 1
            else:
                item = 0
            feas.append(item)
        feaVec.append(feas)
    return dict(zip(words, feaVec))

# 加载id特征向量文件至dict(embeddings_index){word:embedding}
def loadEmbeddings(feaFile):
    words = []
    feaVec = []
    f = open(feaFile, encoding="utf-8")
    for line in f:
        values = line.split()
        words.append(values[0])
        coefs = np.asarray(values[1:])
        # coefs = np.asarray(values[:], dtype='float32')
        feaVec.append(coefs)
    f.close()
    return dict(zip(words, feaVec))
    # return np.array(words),np.array(feaVec)


# 加载id特征向量文件至dict(embeddings_index){word:embedding}
def loadFeatures(feaFile):
    labels = []  # list of label ids
    ids = []
    feaVec = []
    f = open(feaFile)
    for line in f:
        values = line.split()
        if values[0] == '0':
            labels.append([1.0, 0.0])
        else:
            labels.append([0.0, 1.0])
        # if values[0] == '0':
        #     labels.append([0])
        # else:
        #     labels.append([1])
        ids.append(values[1])
        coefs = np.asarray(values[2:], dtype='float32')
        feaVec.append(coefs)
    f.close()
    # return np.array(labels)[700:900],ids,np.array(feaVec)[700:900]
    return np.array(labels),ids,np.array(feaVec)

# 加载id特征向量文件至dict(embeddings_index){word:embedding},特征为随意生成的
def loadFeaturesRandom(feaFile):
    labels = []  # list of label ids
    ids = []
    feaVec = []
    f = open(feaFile)
    for line in f:
        values = line.split()
        if values[0] == '0':
            labels.append([1.0, 0.0])
        else:
            labels.append([0.0, 1.0])
        ids.append(values[1])
        for i in range(2,len(values)):#将特征随机化
            values[i] = random.random()
        coefs = np.asarray(values[2:], dtype='float32')
        feaVec.append(coefs)
    f.close()
    return np.array(labels),ids,np.array(feaVec)

# 加载id特征向量文件至dict(embeddings_index){word:embedding},每一维特征进行标准化
def loadFeaturesScale(feaFile):
    labels = []  # list of label ids
    ids = []
    feaVec = []
    f = open(feaFile)
    for line in f:
        values = line.split()
        if values[0] == '0':
            labels.append([1.0, 0.0])
        else:
            labels.append([0.0, 1.0])
        # for i in range(2,len(values)):
        ids.append(values[1])
        coefs = np.asarray(values[2:], dtype='float32')
        feaVec.append(coefs)
    f.close()
    feaVec = np.array(feaVec)
    scaler = preprocessing.StandardScaler(copy=False,with_mean=True,with_std=True).fit(feaVec)
    #StandardScaler() 的参数with_mean 默认为True表示使用密集矩阵，使用稀疏矩阵则会报错 ，with_mean= False 适用于稀疏矩阵
    # with_std默认为True,如果为True，则将数据缩放为单位方差（单位标准偏差）
    # copy默认为True,如果为False，避免产生一个副本，并执行inplace缩放。 如果数据不是NumPy数组或scipy.sparseCSR矩阵，则仍可能返回副本

    feaVec_scaled = scaler.transform(feaVec)
    return np.array(labels),np.array(ids),feaVec_scaled,scaler

# 加载id特征向量文件至dict(embeddings_index){word:embedding},用训练数据的scaler对验证数据，测试数据进行标准化
def loadFeaturesScaled(feaFile,scaler):
    labels = []  # list of label ids
    ids = []
    feaVec = []
    f = open(feaFile)
    for line in f:
        values = line.split()
        if values[0] == '0':
            labels.append([1.0, 0.0])
        else:
            labels.append([0.0, 1.0])
        ids.append(values[1])
        coefs = np.asarray(values[2:], dtype='float32')
        feaVec.append(coefs)
    f.close()
    feaVec = np.array(feaVec)
    # scaler = preprocessing.StandardScaler(copy=True,with_mean=True,with_std=True).fit(feaVec)
    feaVec_scaled = scaler.transform(feaVec)
    return np.array(labels),ids,feaVec_scaled

# 加载id特征向量文件至dict(embeddings_index){word:embedding},根据sequence的顺序来返回ids
def loadFeaturesScaleFollowSeq(feaFile,sequence):
    lines = []
    labels = []  # list of label ids
    ids = []
    feaVec = []
    f = open(feaFile)
    for line in f:
        values = line.split()
        lines.append(values[:])
    f.close()

    for item in sequence:
        for line in lines:
            if item == line[1]:
                if line[0] == '0':
                    labels.append([1.0, 0.0])
                else:
                    labels.append([0.0, 1.0])
                # for i in range(2,len(values)):
                ids.append(line[1])
                coefs = np.asarray(line[2:], dtype='float32')
                feaVec.append(coefs)
    feaVec = np.array(feaVec)
    scaler = preprocessing.StandardScaler(copy=False, with_mean=True, with_std=True).fit(feaVec)
    feaVec_scaled = scaler.transform(feaVec)
    return np.array(labels), ids, feaVec_scaled, scaler


# 加载id特征向量文件至dict(embeddings_index){word:embedding},每一维特征进行正则化
def loadFeaturesNormalize(feaFile):
    labels = []  # list of label ids
    ids = []
    feaVec = []
    f = open(feaFile)
    for line in f:
        values = line.split()
        if values[0] == '0':
            labels.append([1.0, 0.0])
        else:
            labels.append([0.0, 1.0])
        # for i in range(2,len(values)):
        ids.append(values[1])
        coefs = np.asarray(values[2:], dtype='float32')
        feaVec.append(coefs)
    f.close()
    feaVec = np.array(feaVec)
    normalizer = preprocessing.Normalizer(copy=False,norm='l2').fit(feaVec)
    feaVec_normalized = normalizer.transform(feaVec)
    return np.array(labels),ids,feaVec_normalized,normalizer

# 加载id特征向量文件至dict(embeddings_index){word:embedding},用训练数据的normalizer对验证数据，测试数据进行正则化
def loadFeaturesNormalized(feaFile,normalizer):
    labels = []  # list of label ids
    ids = []
    feaVec = []
    f = open(feaFile)
    for line in f:
        values = line.split()
        if values[0] == '0':
            labels.append([1.0, 0.0])
        else:
            labels.append([0.0, 1.0])
        # for i in range(2,len(values)):
        ids.append(values[1])
        coefs = np.asarray(values[2:], dtype='float32')
        feaVec.append(coefs)
    f.close()
    feaVec = np.array(feaVec)
    # normalizer = preprocessing.Normalizer(copy=False,norm='l2').fit(feaVec)
    feaVec_normalized = normalizer.transform(feaVec)
    return np.array(labels),ids,feaVec_normalized

# 加载id特征向量文件至dict(embeddings_index){word:embedding},每一维特征压缩到0-1
def loadFeaturesMinmaxscale(feaFile):
    labels = []  # list of label ids
    ids = []
    feaVec = []
    f = open(feaFile)
    for line in f:
        values = line.split()
        if values[0] == '0':
            labels.append([1.0, 0.0])
        else:
            labels.append([0.0, 1.0])
        # for i in range(2,len(values)):
        ids.append(values[1])
        coefs = np.asarray(values[2:], dtype='float32')
        feaVec.append(coefs)
    f.close()
    feaVec = np.array(feaVec)
    min_max_scaler = preprocessing.MinMaxScaler()
    feaVec_minmax = min_max_scaler.fit_transform(feaVec)
    return np.array(labels),ids,feaVec_minmax,min_max_scaler

# 加载id特征向量文件至dict(embeddings_index){word:embedding},每一维特征压缩到0-1
def loadFeaturesMinmaxscaled(feaFile,min_max_scaler):
    labels = []  # list of label ids
    ids = []
    feaVec = []
    f = open(feaFile)
    for line in f:
        values = line.split()
        if values[0] == '0':
            labels.append([1.0, 0.0])
        else:
            labels.append([0.0, 1.0])
        # for i in range(2,len(values)):
        ids.append(values[1])
        coefs = np.asarray(values[2:], dtype='float32')
        feaVec.append(coefs)
    f.close()
    feaVec = np.array(feaVec)
    # min_max_scaler = preprocessing.MinMaxScaler()
    feaVec_minmax = min_max_scaler.fit_transform(feaVec)
    return np.array(labels),ids,feaVec_minmax

# 从概率输出转化为分类结果
def fromPtoR(pre_pro):
    pre_cla = []
    for pre in pre_pro:
        pre = np.array(pre)
        pre_cla_one = np.zeros(len(pre))
        index = pre.argmax(axis=0)
        pre_cla_one[index] = 1.
        pre_cla.append(pre_cla_one)
    return np.array(pre_cla)

#将dict中的可以与value对调
def from_wordIdDict_to_idWordDict(word_to_id):
    id_to_word = dict(zip(word_to_id.values(), word_to_id.keys()))
    return id_to_word

#将embedding按照word_to_id中的顺序排列，如果该词没有embedding，则随机产生该词对应的embedding
def getDictEmb_rand(word_to_id,embedding):
    ids = []
    embs = []
    count = 0
    for (k, v) in embedding.items():
        l = len(v)
        # print(v)
        typeOfEmb = type(v[0])
        print("typeOfEmb:" + str(typeOfEmb))
        print("embedding size:" + str(l))
        break
    for (word, id) in word_to_id.items():
        ids.append(id)
        if word in embedding:
            embs.append(embedding[word])
        else:
            count = count + 1
            # print(word)
            rand = np.random.randn(l)#维度要与embedding一致
            # print(rand)
            embs.append(rand)
    print("随机初始化的词：" + str(count))
    # print(len(embs))
    # print(type(embs))
    # print(len(ids))
    # ids = np.reshape(ids, [len(ids), 1])
    dict_id_emb = np.concatenate([embs], axis=1)#不进行拼接，直接返回embeddings
    res = []
    for f in dict_id_emb:
        res.append([float(i) for i in f])
    return res



#将embedding按照word_to_id中的顺序排列，如果该词没有embedding，则该词对应的embedding均置为0
def getDictEmb_0(word_to_id,embedding):
    ids = []
    embs = []
    count = 0
    for (k, v) in embedding.items():
        l = len(v)
        typeOfEmb = type(v[0])
        print("typeOfEmb:" + str(typeOfEmb))
        print("embedding size:" + str(l))
        break
    for (word, id) in word_to_id.items():
        ids.append(id)
        if word in embedding:
            embs.append(embedding[word])
        else:
            count = count + 1
            # print(word)
            zeros = np.zeros([l], dtype=typeOfEmb)#维度与embedding一致
            zeros = np.zeros([l])  # 维度与embedding一致
            embs.append(zeros)
    print("dict中没有的词：" + str(count))
    print()
    dict_id_emb = np.concatenate([embs], axis=1)#不进行拼接，直接返回embeddings
    res = []
    for f in dict_id_emb:
        res.append([float(i) for i in f])
    return res