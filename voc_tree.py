# -*- coding: utf-8 -*-
import cv2
import math
import numpy as np
from collections import deque
from kmeans import KMeansClassifier


class Cluster(object):
    """
    聚类中心类
    """
    def __init__(self, i, l, data):
        self.i = i #节点id
        self.l = l #在树中的深度
        self.data = data #聚类点


class Node(object):
    """
    节点类
    """
    def __init__(self):
        self.cen = None
        self.index = None
        self.inverted_index = None


class Tree(object):
    """
    树类
    """
    def __init__(self, K, L, treeArray):
        self.treeArray = treeArray
        self.L = L
        self.K = K
        self.N = 0 # 图片的数量
        self.imageIDs = [] # 图片的id
        self.dbLengths = {} # 图片对应的tf-idf值

    def build_tree(self, N, db_descriptors):
        """
        建树
        """
        f_i = 0
        for i in range(N):
            self.fill_tree(i, db_descriptors[f_i])
            f_i += 1
        self.set_lengths()

    def propagate(self, pt):
        """
        计算特征点到节点的距离
        返回距离最近的叶子节点i
        """
        i = 0 # 初始化节点id
        l = 0 # 初始化树的深度
        closeChild = 0
        while l != self.L:
            curDist = np.inf # 最小
            minDist = np.inf
            for x in range(0,self.K):
                childPos = findChild(self.K,i,x)
                testPT = self.treeArray[childPos].cen
                if testPT is None:
                    continue
                # 计算欧几里得距离
                curDist = np.linalg.norm(testPT - pt)
                if curDist < minDist:
                    minDist = curDist
                    closeChild = childPos
            i = closeChild
            l += 1
        return i

    def fill_tree(self, imageID, features):
        """
        填充反向索引
        叶子节点的反向索引字典包含图片id以及对应的出现的次数
        """
        for feat in features:
            leaf_node = self.propagate(feat)
            # 增加反向索引
            if imageID not in self.treeArray[leaf_node].inverted_index:
                self.treeArray[leaf_node].inverted_index[imageID] = 1
            else:
                self.treeArray[leaf_node].inverted_index[imageID] += 1
        self.N += 1 # 增加图片的数量
        self.imageIDs.append(imageID)

    def set_lengths(self):
        """
        图片id对应的tf-idf值
        用于查询
        """
        num_nodes = len(self.treeArray)
        num_leafs = self.K ** self.L
        for imageID in self.imageIDs:
            cum_sum = float(0)
            # 只迭代叶子节点
            for lf in range(num_nodes-1, num_nodes-num_leafs-1, -1):
                if self.treeArray[lf].inverted_index == None:
                    continue
                if imageID in self.treeArray[lf].inverted_index:
                    # tf是lf单词在图像中的词频
                    tf = self.treeArray[lf].inverted_index[imageID]
                    # df是包含lf单词的图片数量
                    df = len(self.treeArray[lf].inverted_index)
                    idf = math.log( float(self.N) / float(df) )
                    cum_sum += math.pow( tf*idf , 2)
            self.dbLengths[imageID] = math.sqrt( cum_sum )

    def transform(self, imageID):
        """
        把图像转换为单词向量
        """
        vecList = []
        num_nodes = len(self.treeArray)
        num_leafs = self.K ** self.L
        for lf in range(num_nodes-1, num_nodes-num_leafs-1, -1):
            # print self.treeArray[lf].inverted_index
            if self.treeArray[lf].inverted_index is None:
                continue
            if imageID in self.treeArray[lf].inverted_index:
                vecList.append(self.treeArray[lf].inverted_index[imageID])
            else:
                vecList.append(0)
        vec = np.array(vecList)
        return vec

    def process_query(self, features, n):
        """
        查询图像库
        返回得分最高的n幅图像
        """
        scores = {}
        for feat in features:
            leaf_node = self.propagate(feat)
            idx = self.treeArray[leaf_node].inverted_index.items()
            for (ID,count) in idx:
                df = len(idx)
                idf = math.log( float(self.N) / float(df) )
                idf_sq = idf * idf
                tf  = count
                score = float(tf * idf_sq)
                if ID not in scores:
                    scores[ID] = score
                else:
                    scores[ID] += score
        scores = scores.items()
        final_scores = []
        for i in range(len(scores)):
            (ID,score) = scores[i]
            nmz_score = float(score) / float(self.dbLengths[ID])
            final_scores.append((ID, nmz_score))
        final_scores.sort(key=lambda pair : pair[1], reverse=True)
        return final_scores[0:n]


def findChild(K, i, x):
    """
    返回节点i的第x个节点
    """
    return (K*(i+1)-(K-2)+x-1)


def constructTree(K, L, data):
    """
    构建字典树
    """
    print "building tree: K = " + str(K) + ", L = " + str(L)

    NUM_NODES = (K**(L+1)-1)/(K-1) #总节点数

    treeArray = [Node() for i in range(NUM_NODES)]
    NUM_LEAFS = 0

    cv2_iter = cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER
    criteria = (cv2_iter, 10, 1.0)

    queue = deque() #采用队列
    queue.appendleft( Cluster(0,0,data) )
    with open('voc.txt', 'w') as f: #保存txt文件
        f.writelines('{} {}  0 3'.format(K, L))
        f.write('\n')

        while len(queue):
            clust = queue.pop()

            # print clust.data
            if K <= len(clust.data):
                # opencv实现

                compactness, label, center=cv2.kmeans(clust.data,
                                                     K,
                                                     None,
                                                     criteria,
                                                     10,
                                                     cv2.KMEANS_RANDOM_CENTERS)

                # kmeans实现

                # clf = KMeansClassifier(K)
                # clf.fit(clust.data)
                # center = clf._centroids
                # label = clf._labels

                if clust.l+1 < L:
                    # print "NOT LEAF"
                    for x in range(0,K):
                        des = center[x].astype(int)
                        d = des.tolist()

                        childPos = findChild(K,clust.i,x)

                        # opencv
                        queue.appendleft(Cluster(childPos,
                                                 clust.l+1,
                                                 clust.data[label.ravel()==x]))

                        # kmeans
                        # queue.appendleft(Cluster(childPos,
                        #                          clust.l+1,
                        #                          clust.data[label==x]))

                        treeArray[childPos].cen = center[x,:]

                        f.writelines('{} {} {} {}'.format(clust.i, 0, ' '.join(str(i) for i in d), 0))
                        f.write('\n')

                else:
                    # print "LEAF"
                    for x in range(0,K):
                        des = center[x].astype(int)
                        d = des.tolist()

                        childPos = findChild(K,clust.i,x)

                        f.writelines('{} {} {} {}'.format(clust.i, 1, ' '.join(str(i) for i in d), 1))
                        f.write('\n')

                        treeArray[childPos].inverted_index = {}
                        treeArray[childPos].cen = center[x,:]
                        if clust.data.size == 0:
                            print "ZERO CLUSTER"
                        NUM_LEAFS += 1
            else:
                x = 0
                childPos = findChild(K,clust.i,x)
                treeArray[childPos].cen = np.zeros(len(clust.data[0,:]),
                                                   dtype='float32')
                if clust.l+1 != L:
                    queue.appendleft(Cluster(childPos,
                                             clust.l+1,
                                             clust.data))
                else:
                    treeArray[childPos].inverted_index = {}

    print "num leafs: " + str(NUM_LEAFS)
    print 'save voc.txt ... done'
    return treeArray
