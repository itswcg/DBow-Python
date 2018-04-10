# -*- coding: utf-8 -*-
import time
import numpy as np
from voc_tree import *

MIN_MATCH_COUNT = 10
N_DECIMAL = 4


def gv_pair(q_kp, q_des, db_kp, db_des):
    """
    Perform geometric verification on a query-database
        image pair.
    q_kp: query image feature keypoints (locations)
    q_des: query image feature descriptors
    db_kp: database image feature keypoints (locations)
    db_des: database image feature descriptors
    M: homography matrix
    n_inliers: number of inliers for best consensus model using RANSAC
    ransac_matches: features deemed as matches after gv
    """
    M = None
    n_inliers = 0
    ransac_matches = []

    q_des = np.asarray(q_des,np.float32)
    db_des= np.asarray(db_des,np.float32)

    matcher = cv2.BFMatcher() # brute-force matcher
    matches = matcher.knnMatch(q_des, db_des, 2)
    good_matches = []
    for m,n in matches:
        if m.distance < 0.65*n.distance:
            good_matches.append([m])
    # find & compute homography if there are enough good_matches
    if len(good_matches) > MIN_MATCH_COUNT:
        src_pts = [q_kp[m[0].queryIdx] for m in good_matches]
        src_pts = np.float32(src_pts).reshape(-1,1,2)
        dst_pts = [db_kp[m[0].trainIdx] for m in good_matches]
        dst_pts = np.float32(dst_pts).reshape(-1,1,2)
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        if mask is None:
            return M, n_inliers, ransac_matches
        matchesMask = mask.ravel().tolist()
        n_inliers = len( [1 for m in matchesMask if m==1] )
        ransac_matches = [good_matches[i] for i in range(len(matchesMask)) \
                          if matchesMask[i]==1]
    return M, n_inliers, ransac_matches


class Matcher(object):
    """
    module for finding match given query and database descriptors.
    """
    def __init__(self, N, db_des):
        self.tree = None
        self.N = N
        self.db_descriptors = db_des


    def update_tree(self, in_tree):
        """
        set the tree
        """
        self.tree = in_tree

    def query_features(self, q_des, m=5):
        """
        q_des: features of query image
        q_kp: query feature keypoints
        m: do geometric reranking on top m from BOW list
        r: return ranked list of r results to user
        """
        bow_list = self.tree.process_query(q_des,m)
        return bow_list


    def query(self, q_id):
        """
        query using the id number of the query image
        """
        assert isinstance(q_id, int)
        if q_id > self.N or q_id < 0:
            print "query id out of range"
            return
        q_des= self.db_descriptors[q_id]
        return self.query_features(q_des, 10)

    def cos_sim(self, vec_a, vec_b):
        vec_a = np.mat(vec_a)
        vec_b = np.mat(vec_b)
        cos = float(vec_a * vec_b.T) / (np.linalg.norm(vec_a) * np.linalg.norm(vec_b))
        sim = 0.5 + 0.5 * cos
        return sim
