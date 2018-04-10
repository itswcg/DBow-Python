# -*- coding: utf-8 -*-
import cv2
import math
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


class Tree(object):
    """
    Hierarchical k-means tree.
        C: num children per node
        L: num levels in tree (root node not included)
        treeArray: 1-d array of node objects
    """

    def __init__(self, K, L, treeArray):
        self.treeArray = treeArray
        self.L = L # levels in tree
        self.K = K # branching factor: children per internal node
        self.N = 0 # num images contributing to tree
        self.imageIDs = []
        self.dbLengths = {}

    def build_tree(self, N, db_descriptors):
        """
        db_names: list of image names
        db_descriptors: list of db image feature descriptors
        """
        f_i = 0
        for i in range(N):
            self.fill_tree(i, db_descriptors[f_i])
            f_i += 1
        self.set_lengths()

    def propagate(self, pt):
        """
        propogate a feature descriptor (pt) down the tree until
        reaching leaf node. Path down tree is dictated by euclidean
        distance between pt and cluster centroids (tree nodes).
        Array index of leaf node is returned (i).
        """
        i = 0 # initialize position to top node
        l = 0 # initialize position to top level
        closeChild = 0
        while l != self.L:
            curDist = np.inf
            minDist = np.inf
            for x in range(0,self.K):
                childPos = findChild(self.K,i,x)
                testPT = self.treeArray[childPos].cen
                if testPT is None:
                    continue
                # euclidean distance between child x and pt
                curDist = np.linalg.norm(testPT - pt)
                if curDist < minDist:
                    minDist = curDist
                    closeChild = childPos
            i = closeChild
            l += 1
        return i

    def fill_tree(self, imageID, features):
        """
        updates inverted indexes with imageID and corresponding features
        """
        for feat in features:
            # quantize feat to leaf node
            leaf_node = self.propagate(feat)
            # add to inverted index
            if imageID not in self.treeArray[leaf_node].inverted_index:
                self.treeArray[leaf_node].inverted_index[imageID] = 1
            else:
                self.treeArray[leaf_node].inverted_index[imageID] += 1
        self.N += 1 # increase num images contributing to tree
        self.imageIDs.append(imageID)

    def set_lengths(self):
        """
        find database image vector lengths (used in score normalization)
        """
        # process db vector lengths:
        num_nodes = len(self.treeArray)
        num_leafs = self.K ** self.L
        for imageID in self.imageIDs:
            cum_sum = float(0)
            # iterate over only leaf nodes:
            for lf in range(num_nodes-1, num_nodes-num_leafs-1, -1):
                if self.treeArray[lf].inverted_index == None:
                    continue
                if imageID in self.treeArray[lf].inverted_index:
                    # tf is frequency of lf in imageID
                    tf = self.treeArray[lf].inverted_index[imageID]
                    # df is num images containing lf visual word
                    df = len(self.treeArray[lf].inverted_index)
                    idf = math.log( float(self.N) / float(df) )
                    cum_sum += math.pow( tf*idf , 2)
            self.dbLengths[imageID] = math.sqrt( cum_sum )

    def transform(self, imageID):
        vecList = []
        num_nodes = len(self.treeArray)
        num_leafs = self.K ** self.L
        for lf in range(num_nodes-1, num_nodes-num_leafs-1, -1):
            if imageID in self.treeArray[lf].inverted_index:
                vecList.append(self.treeArray[lf].inverted_index[imageID])
            else:
                vecList.append(0)
        vec = np.array(vecList)
        return vec


    def process_query(self, features, n):
        """
        features: list of features in query image
        n: return top n scores
        """
        scores = {} # dict of imageID to score
        for feat in features:
            leaf_node = self.propagate(feat)
            idx = self.treeArray[leaf_node].inverted_index.items()
            for (ID,count) in idx:
                df = len(idx) # document frequency in inverted index
                idf = math.log( float(self.N) / float(df) )
                idf_sq = idf * idf
                tf  = count
                score = float(tf * idf_sq)
                if ID not in scores:
                    scores[ID] = score
                else:
                    scores[ID] += score
        # normalize scores by scaling by norm of db vectors
        scores = scores.items()
        final_scores = [] # TODO: change this to a heap so it can sort by score
        for i in range(len(scores)):
            (ID,score) = scores[i]
            nmz_score = float(score) / float(self.dbLengths[ID])
            # TODO: change this to push onto heap so we can auto sort
            final_scores.append((ID, nmz_score))
        # sort final scores and return
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
    with open('orb.txt', 'w') as f: #保存txt文件
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

                if clust.l+1 != L:
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

    # print "num leafs: " + str(NUM_LEAFS)
    print 'save orb.txt ... done'
    return treeArray
