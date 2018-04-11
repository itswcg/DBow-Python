# -*- coding: utf-8 -*-
import time
import numpy as np
from voc_tree import *


class Matcher(object):
    """
    比配查询类
    """
    def __init__(self, N, db_des, tree):
        self.tree = tree
        self.N = N
        self.db_descriptors = db_des

    def query_features(self, q_des, m=5):
        """
        查询特征
        """
        bow_list = self.tree.process_query(q_des,m)
        return bow_list

    def query(self, q_id):
        """
        查询图片id
        """
        assert isinstance(q_id, int)
        if q_id > self.N or q_id < 0:
            print "query id out of range"
            return
        q_des= self.db_descriptors[q_id]
        return self.query_features(q_des, 10)

    def cos_sim(self, vec_a, vec_b):
        """
        计算余弦相似度
        """
        vec_a = np.mat(vec_a)
        vec_b = np.mat(vec_b)
        cos = float(vec_a * vec_b.T) / (np.linalg.norm(vec_a) * np.linalg.norm(vec_b))
        sim = 0.5 + 0.5 * cos
        return sim
