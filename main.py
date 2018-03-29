import cv2
import numpy as np
from orb import orb_features
from voc_tree import *


N = 10
K = 10
L = 5

image_descriptors = orb_features(N)
# print image_descriptors

FEATS = []

for feats in image_descriptors:
    FEATS += [np.array(fv, dtype='float32') for fv in feats]

FEATS = np.vstack(FEATS)
# print FEATS

treeArray = constructTree(K, L, np.vstack(FEATS))

