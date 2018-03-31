# -*- coding: utf-8 -*-
import cv2
import numpy as np
from matplotlib import pyplot as plt


def orb_features(n):
    """
    ORB特征提取
    """
    print 'reading images ...'

    descriptors = []
    orb = cv2.ORB_create()

    for i in range(n):
        img = cv2.imread('images/{0}.png'.format(i), 0)
        keypoint, descriptor = orb.detectAndCompute(img, None)
        #特征转int型
        descriptor = descriptor.astype(int)
        #特征转列表
        descriptor = descriptor.tolist()
        descriptors.append(descriptor)

    return descriptors
