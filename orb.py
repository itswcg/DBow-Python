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
    orb = cv2.ORB_create(500) #表示提取一幅图像的n个特征, 如需生成字典文件，改为10000

    for i in range(n):
        img = cv2.imread('images_0/{0}.png'.format(i), 0)
        keypoint, descriptor = orb.detectAndCompute(img, None)
        #特征转int型
        # print descriptor
        descriptor = descriptor.astype(int)
        #特征转列表
        descriptor = descriptor.tolist()
        descriptors.append(descriptor)

    return descriptors


def update_image(n):
    orb = cv2.ORB_create(500)
    img = cv2.imread('images_0/{0}.png'.format(n), 0)
    keypoint, descriptor = orb.detectAndCompute(img, None)
    descriptor = descriptor.astype(int)
    descriptor = descriptor.tolist()
    return descriptor

if __name__ == '__main__':
    img1=cv2.imread('images_0/0.png', cv2.IMREAD_GRAYSCALE)
    img2=cv2.imread('images_0/1.png', cv2.IMREAD_GRAYSCALE)

    orb = cv2.ORB_create()
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1,des2)
    matches = sorted(matches, key=lambda x:x.distance)
    img3 = cv2.drawMatches(img1, kp1, img2, kp2, matches[:50], img2,flags=2)
    plt.imshow(img3), plt.show()
