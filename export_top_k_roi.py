# !/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Author              : Ziyuan Ye
@Email               : ziyuanye9801@gmail.com
'''

import numpy as np
import pandas as pd

import os
import heapq

# cate_list = ['0bk_body', '0bk_faces', '0bk_places', '0bk_tools', '2bk_body', '2bk_faces', '2bk_places', '2bk_tools',
#                  'loss', 'win', 't', 'lf', 'lh', 'rh', 'rf', 'math', 'story', 'mental', 'rnd', 'match', 'relation',
#                  'fear', 'neut']

# cate_list = ['emotion(emotional)', 'gambling', 'language(auditory)', 'motion(movement)', 'relational',
#              'social(social cognition)', 'working memory']

cate_list = ['body', 'faces', 'places', 'tools']



def export_top_k(cate, k, pth):
    file = pd.read_csv(os.path.join(pth, cate+".csv"))
    result = [float(file.keys()[0])]
    result.extend(list(file[file.keys()[0]]))
    result = [float(x) for x in result]
    result = np.array(result)
    topk = heapq.nlargest(k, range(len(result)), result.take)
    topk = [str((x+1)%180) for x in topk]
    return topk

# pth = r'.\result_cv\MMP\stpgcn\saliency_result\yeo17\mix'
# pth = r'./result_cv/MMP/stgcn/saliency_result/yeo17/mix_new'
# pth = r'result_cv/MMP/mlp_mixer/saliency_result/yeo17/mix'
# pth = r'.\result_cv\TNNLS\Neurosynth_groundtruth'
pth = r'.\result_cv\MMP\WM_task\diff_stimuli\stpgcn\saliency_result\yeo17\mix'

region = pd.read_csv(r'result_cv\region.csv')

for cate in cate_list:
    k = 10
    topk = export_top_k(cate, k, pth)
    for i in range(len(topk)):
        if topk[i] == '0':
            topk[i] = '180'
    top_region = set()
    for k in topk:
        id = [str(x) for x in list(region['regionID'])].index(k)
        top_region.add(region['cortex'][id])
    print(cate, ':', top_region)

# region = pd.read_csv(r'result_cv\region.csv')
# region['cortex']
# region['regionID']

