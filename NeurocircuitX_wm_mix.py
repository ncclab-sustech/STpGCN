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


def mix_02():
    cate_list = ['bk_body', 'bk_faces', 'bk_places', 'bk_tools']
    model_list = ['gcn', 'gat', 'stgcn', 'stpgcn']
    strategy = ['keep', 'ablation']
    for cate in cate_list:
        for model_name in model_list:
            for stra in strategy:
                bk_0_pth = r'.\result_cv\WM_task\diff_stimuli\{}\saliency_result\yeo17\{}\{}{}.csv'.format(model_name,
                                                                                                 stra,
                                                                                                 '0',
                                                                                                 cate)
                bk_2_pth = r'.\result_cv\WM_task\diff_stimuli\{}\saliency_result\yeo17\{}\{}{}.csv'.format(model_name,
                                                                                                 stra,
                                                                                                 '2',
                                                                                                 cate)
                save_pth = r'.\result_cv\WM_task\diff_stimuli\{}\saliency_result\yeo17\{}\{}.csv'.format(model_name,
                                                                                                 stra,
                                                                                                 cate[3:])
                bk_0_weight = np.loadtxt(bk_0_pth)
                bk_2_weight = np.loadtxt(bk_2_pth)
                save_array = []
                for i in range(379):
                    mix_weight = bk_0_weight[i] * 0.5 + bk_2_weight[i] * 0.5
                    save_array.append(mix_weight)
                save_array = np.array(save_array)
                save_array = stdlize(save_array)
                np.savetxt(save_pth,
                           save_array,
                       delimiter=",")
    print('finish')


def mix_acc():
    cate_list = ['body', 'faces', 'places', 'tools']
    model_list = ['gcn', 'gat', 'stgcn', 'stpgcn']
    for cate in cate_list:
        for model_name in model_list:
            keep_pth = r'.\result_cv\WM_task\diff_stimuli\{}\saliency_result\yeo17\keep\{}.csv'.format(model_name,
                                                                                                       cate)
            ablation_pth = r'.\result_cv\WM_task\diff_stimuli\{}\saliency_result\yeo17\ablation\{}.csv'.format(model_name,
                                                                                                       cate)
            save_pth = r'.\result_cv\WM_task\diff_stimuli\{}\saliency_result\yeo17\mix\{}.csv'.format(model_name,
                                                                                                     cate)
            bk_0_weight = np.loadtxt(keep_pth, encoding='gbk')
            bk_2_weight = np.loadtxt(ablation_pth, encoding='gbk')
            save_array = []
            for i in range(379):
                mix_weight = bk_0_weight[i] * 0.5 + bk_2_weight[i] * 0.5
                save_array.append(mix_weight)
            save_array = np.array(save_array)
            save_array = stdlize(save_array)
            np.savetxt(save_pth,
                       save_array,
                       delimiter=",")
    print('finish')


if __name__ == '__main__':
    # mix_02()
    mix_acc()