# -*- coding: utf-8 -*-
import cv2
import numpy as np
from orb import orb_features, update_image
from voc_tree import constructTree
from matcher import *


N = 10 #图片的数量
K = 5 #聚类K类
L = 3 #字典树L层
n = 10 #增加的图片
T = 0.95

image_descriptors = orb_features(N) #提取特征
# print image_descriptors

FEATS = []

for feats in image_descriptors:
    FEATS += [np.array(fv, dtype='float32') for fv in feats]

FEATS = np.vstack(FEATS) #将特征转化为np的数组
# print FEATS

treeArray = constructTree(K, L, np.vstack(FEATS)) #建立字典树，并打印结果
tree = Tree(K, L, treeArray)
tree.build_tree(N, image_descriptors)
print "the vector of image:"
print tree.transform(0)
# print tree.imageIDs, tree.dbLengths

matcher = Matcher(N, image_descriptors, tree)
# print matcher.query(4)

# add images
des = update_image(n)
tree.update_tree(n, des)
print tree.transform(10)
# print tree.imageIDs

# 比较
print "compute cosine similarity:"
res = {}
for i in range(tree.N-1):
    print 'Image {} vs Image {}: {}'.format(n, i, matcher.cos_sim(tree.transform(n), tree.transform(i)))
    if matcher.cos_sim(tree.transform(n), tree.transform(i)) > T:
        res[i] = matcher.cos_sim(tree.transform(n), tree.transform(i))
if res:
    r = max(res.items(), key=lambda x:x[1])[0]
    print ("相似度最高的图片为{}.jpg".format(r))
else:
    print("None")
