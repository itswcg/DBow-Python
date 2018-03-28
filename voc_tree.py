import cv2
import numpy as np
from matplotlib import pyplot as plt
from collections import deque


class Cluster(object):
    def __init__(self, i, l, data):
        self.i = i
        self.l = l
        self.data = data


class Node(object):
    def __init__(self):
        self.cen = None
        self.index = None
        self.inverted_index = None


def findChild(K, i, x):
    return (K*(i+1)-(K-2)+x-1)

def constructTree(K, L, data):
    print "building tree: K = " + str(K) + ", L = " + str(L)

    NUM_NODES = (K**(L+1)-1)/(K-1)

    # initialize tree array with empty nodes
    treeArray = [Node() for i in range(NUM_NODES)]
    NUM_LEAFS = 0

    # KMEANS PARAM INPUTS
    cv2_iter = cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER
    criteria = (cv2_iter, 10, 1.0) #biaozun

    queue = deque()
    queue.appendleft( Cluster(0,0,data) )
    with open('orb.txt', 'a') as f:
        f.writelines('{} {}  0 0'.format(K, L))
        f.write('\n')

        while len(queue):
            clust = queue.pop()
            # KMEANS FUNCTION CALL
            # let N be then number of points we seek to cluster using KMeans
            # let d be the dimensionality of each data point
            # compactness is not used by us
            # label: Nx1 numpy array with labels \in [0,C)
            # center: Cxd numpy array with rows as cluster centroids

            if K <= len(clust.data):
                # on mac (opencv 2.4.5):
                #compactness, label, center = cv2.kmeans(clust.data,
                #                                        C,
                #                                        criteria,
                #                                        10,
                #                                        0)
                # # on CAEN (opencv 3.1.0):
                compactness, label, center=cv2.kmeans(clust.data,
                                                     K,
                                                     None,
                                                     criteria,
                                                     10,
                                                     cv2.KMEANS_RANDOM_CENTERS)
                # custom kmeans implementation:
                #label, center = kmeans(cv2.TERM_CRITERIA_EPS,
                #                       cv2.TERM_CRITERIA_MAX_ITER,
                #                       clust.data,
                #                       C)

                if clust.l+1 != L:
                    # print "NOT LEAF"
                    for x in range(0,K):
                        childPos = findChild(K,clust.i,x)
                        print childPos
                        queue.appendleft(Cluster(childPos,
                                                 clust.l+1,
                                                 clust.data[label.ravel()==x]))
                        treeArray[childPos].cen = center[x,:]
                        f.writelines('{} {} {} {}'.format(clust.i, 0, clust.data[label.ravel()==x], 0))
                        f.write('\n')
                        # print treeArray[childPos].cen
                        # print clust.data
                else:
                    # print "LEAF"
                    for x in range(0,K):
                        childPos = findChild(K,clust.i,x)
                        f.writelines('{} {} {} {}'.format(clust.i, 1, clust.data[label.ravel()==x], 0))
                        f.write('\n')
                        # print childPos
                        #treeArray[childPos].index = clust.data[label.ravel()==x]
                        treeArray[childPos].inverted_index = {}
                        treeArray[childPos].cen = center[x,:]
                        if clust.data.size == 0:
                            print "ZERO CLUSTER ========="
                        NUM_LEAFS += 1
            else:
                # pass down data to first (0th) child;
                # pass down Nones to other children
                # (aka do nothing since they were initialized with None)
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
    return treeArray
