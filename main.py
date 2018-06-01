# -*- coding: utf-8 -*-
import cv2
import numpy as np
from orb import orb_features, update_image
from voc_tree import constructTree
from matcher import *


N = 20 #训练字典图片的数量
K = 12 #聚类K类
L = 5 #字典树L层
n = 21 #测试的图片的数量
T = 1 #相似度阈值

image_descriptors = orb_features(N) #提取特征
# print image_descriptors

FEATS = []

for feats in image_descriptors:
    FEATS += [np.array(fv, dtype='float32') for fv in feats]

FEATS = np.vstack(FEATS) #将特征转化为np的数组
# print FEATS

treeArray = constructTree(K, L, np.vstack(FEATS)) #建立字典树，并打印结果
tree = Tree(K, L, treeArray)
# tree.build_tree(N, image_descriptors)
# print "the vector of image:"
# print tree.transform(2)
# print tree.imageIDs, tree.dbLengths

# matcher = Matcher(N, image_descriptors, tree)
# # print matcher.query(4)

# # add images

# for i in range(n):
#     des = update_image(i)
#     tree.update_tree(i, des)
# # print tree.imageIDs

# # 比较
#     print "{}.jpg compute cosine similarity:".format(i)
#     res = {}
#     for j in range(tree.N-1):
#         print 'Image {} vs Image {}: {}'.format(i, j, matcher.cos_sim(tree.transform(i), tree.transform(j)))
#         if matcher.cos_sim(tree.transform(i), tree.transform(j)) >= T:
#             res[j] = matcher.cos_sim(tree.transform(i), tree.transform(j))
#     if res:
#         r = max(res.items(), key=lambda x:x[1])[0]
#         print ("相似度最高的图片为{}.jpg".format(r))
#         print tree.transform(r)
#     else:
#         print("None")

