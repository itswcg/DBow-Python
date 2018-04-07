# -*- coding: utf-8 -*-
import cv2
import numpy as np
from orb import orb_features
from voc_tree import constructTree


N = 10 #图片的数量
K = 10 #聚类K类
L = 5 #字典树L层、、、、

image_descriptors = orb_features(N) #提取特征
# print image_descriptors

FEATS = []

for feats in image_descriptors:
    FEATS += [np.array(fv, dtype='float32') for fv in feats]

FEATS = np.vstack(FEATS) #将特征转化为np的数组
# print FEATS

treeArray = constructTree(K, L, np.vstack(FEATS)) #建立字典树，并打印结果

