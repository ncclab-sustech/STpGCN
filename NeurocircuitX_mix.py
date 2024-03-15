# !/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Author              : Ziyuan Ye
@Email               : ziyuanye9801@gmail.com
'''
import numpy as np
import os


def stdlize(data: np.ndarray):
    data = np.array([(_data - np.min(data)) / (np.max(data) - np.min(data))
              for _data in data]).reshape(-1, 1)
    return data

def mix_acc():
    cate_list = ['0bk_body', '0bk_faces', '0bk_places', '0bk_tools', '2bk_body', '2bk_faces', '2bk_places', '2bk_tools',
                 'loss', 'win', 't', 'lf', 'lh', 'rh', 'rf', 'math', 'story', 'mental', 'rnd', 'match', 'relation',
                 'fear', 'neut']
    for cate in cate_list:

        # ab_weight = np.loadtxt(r'.\result_cv\gcn\saliency_result\yeo17\ablation\{}.csv'.format(cate))
        # ke_weight = np.loadtxt(r'.\result_cv\gcn\saliency_result\yeo17\keep\{}.csv'.format(cate))
        # ab_weight = np.loadtxt(r'.\result_cv\gat\saliency_result\yeo17\ablation\{}.csv'.format(cate))
        # ke_weight = np.loadtxt(r'.\result_cv\gat\saliency_result\yeo17\keep\{}.csv'.format(cate))
        # ab_weight = np.loadtxt(r'.\result_cv\stgcn\saliency_result\yeo17\ablation\{}.csv'.format(cate))
        # ke_weight = np.loadtxt(r'.\result_cv\stgcn\saliency_result\yeo17\keep\{}.csv'.format(cate))
        # ab_weight = np.loadtxt(r'.\result_cv\stpgcn\saliency_result\yeo17\ablation\{}.csv'.format(cate))
        # ke_weight = np.loadtxt(r'.\result_cv\stpgcn\saliency_result\yeo17\keep\{}.csv'.format(cate))

        ab_weight = np.loadtxt(r'.\result_cv\MMP\mlp_mixer\saliency_result\yeo17\ablation\{}.csv'.format(cate))
        ke_weight = np.loadtxt(r'.\result_cv\MMP\mlp_mixer\saliency_result\yeo17\keep\{}.csv'.format(cate))

        save_array = []
        for i in range(379):
            mix_weight = ab_weight[i] * 0.5 + ke_weight[i] * 0.5
            save_array.append(mix_weight)
        save_array = np.array(save_array)
        save_array = stdlize(save_array)
        # pth = r'.\result_cv\gcn\saliency_result\yeo17\mix'
        # pth = r'.\result_cv\gat\saliency_result\yeo17\mix'
        # pth = r'.\result_cv\stgcn\saliency_result\yeo17\mix'
        # pth = r'.\result_cv\stpgcn\saliency_result\yeo17\mix'
        pth = r'.\result_cv\MMP\mlp_mixer\saliency_result\yeo17\mix'
        np.savetxt(os.path.join(pth,'{}.csv'.format(cate)),save_array,delimiter=",")
    print('finish')

mix_acc()