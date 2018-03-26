import cv2
import numpy as np
from matplotlib import pyplot as plt


def orb_features(n):
    print 'reading images ...'

    descriptors = []
    orb = cv2.ORB_create()
    for i in range(n):
        img = cv2.imread('images/{0}.png'.format(i), 0)
        keypoint, descriptor = orb.detectAndCompute(img, None)
        descriptor = descriptor.astype(int)
        descriptor = descriptor.tolist()
        descriptors.append(descriptor)

    return descriptors
