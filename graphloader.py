# !/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Author              : Ziyuan Ye
@Email               : ziyuanye9801@gmail.com
'''

import dgl

import numpy as np


def build_knngraph_from_weight_matrix(weight_matrix: np.ndarray, k: int) -> dgl.DGLGraph:
    """
    Build graph using k-NN with weight matrix.
    :param k: top k neighbours
    :param weight_matrix:
    :return:
    """
    src_node = []
    dst_node = []
    for i in range(weight_matrix.shape[0]):
        """Find top k neighbors with functional correlation matrix"""
        dst = weight_matrix[i].argsort()[-k:][::-1]
        src_node.extend([i]*len(dst))
        dst_node.extend(dst)
    G = dgl.graph((src_node, dst_node))
    """Make the directed graph become undirected graph"""
    G = dgl.to_bidirected(G)
    return G


# if __name__ == '__main__':
#     fc, time_series, label = dataloader.dataloader(cate='motor')
#     con_G = build_knngraph_from_weight_matrix(fc[0], 5)
#     # print(con_G)
#     in_degs = con_G.in_degrees().float().clamp(min=1)  # shape [num_nodes]
#     out_degs = con_G.out_degrees().float().clamp(min=1)  # shape [num_nodes]