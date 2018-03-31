# -*- coding: utf-8 -*-
import cv2
import numpy as np
from collections import deque
from kmeans import KMeansClassifier

class Cluster(object):
    """
    聚类中心
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
    with open('orb.txt', 'w') as f: #保存txt文件
        f.writelines('{} {}  0 0'.format(K, L))
        f.write('\n')

        while len(queue):
            clust = queue.pop()

            # print clust.data
            if K <= len(clust.data):
                # opencv实现

                # compactness, label, center=cv2.kmeans(clust.data,
                #                                      K,
                #                                      None,
                #                                      criteria,
                #                                      10,
                #                                      cv2.KMEANS_RANDOM_CENTERS)

                # kmeans实现

                clf = KMeansClassifier(K)
                clf.fit(clust.data)
                center = clf._centroids
                label = clf._labels

                if clust.l+1 != L:
                    # print "NOT LEAF"
                    for x in range(0,K):
                        des = center[x].astype(int)
                        d = des.tolist()

                        childPos = findChild(K,clust.i,x)
                        queue.appendleft(Cluster(childPos,
                                                 clust.l+1,
                                                 clust.data[label.ravel()==x]))
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

    # print "num leafs: " + str(NUM_LEAFS)
    print 'save orb.txt ... done'
    return treeArray
