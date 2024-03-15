# !/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Author              : Ziyuan Ye
@Email               : ziyuanye9801@gmail.com
'''
import random

import numpy as np
import pandas as pd

import os
import heapq
import time

import torch

import dataloader

from sklearn.svm import SVC
from sklearn import metrics


# cate_list = ['0bk_body', '0bk_faces', '0bk_places', '0bk_tools', '2bk_body', '2bk_faces', '2bk_places', '2bk_tools',
#                  'loss', 'win', 't', 'lf', 'lh', 'rh', 'rf', 'math', 'story', 'mental', 'rnd', 'match', 'relation',
#                  'fear', 'neut']
cate_list = ['0bk_body', '0bk_faces', '0bk_places', '0bk_tools', '2bk_body', '2bk_faces', '2bk_places', '2bk_tools']

non_vis_idx = [7, 8, 9, 10, 11, 13, 14, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42,
               43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68,
               69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94,
               95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116,
               117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 138,
               139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 154, 160, 161, 163, 164, 165, 166, 167, 168,
               169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 187, 188, 189, 190, 191, 193, 194, 203, 204, 205,
               206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224, 225, 226,
               227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 247,
               248, 249, 250, 251, 252, 253, 254, 255, 256, 257, 258, 259, 260, 261, 262, 263, 264, 265, 266, 267, 268,
               269, 270, 271, 272, 273, 274, 275, 276, 277, 278, 279, 280, 281, 282, 283, 284, 285, 286, 287, 288, 289,
               290, 291, 292, 293, 294, 295, 296, 297, 298, 299, 300, 301, 302, 303, 304, 305, 306, 307, 308, 309, 310,
               311, 312, 313, 314, 315, 316, 318, 319, 320, 321, 322, 323, 324, 325, 326, 327, 328, 329, 330, 334, 340,
               341, 343, 344, 345, 346, 347, 348, 349, 350, 351, 352, 353, 354, 355, 356, 357, 358, 359]


def export_top_k(cates, k, pth):

    final_set = set()
    for cate in cates:
        file = pd.read_csv(os.path.join(pth, cate + ".csv"))
        result = [float(file.keys()[0])]
        result.extend(list(file[file.keys()[0]]))
        result = [float(x) for x in result]
        for index in range(len(result)):
            if index not in non_vis_idx:
                result[index] = 0
        result = np.array(result)
        topk = heapq.nlargest(k, range(len(result)), result.take)
        topk = [str((x+1)) for x in topk]
        for value in topk:
            final_set.add(value)
    return list(final_set)

def neurocircuit_export_top_k(pth, k):
    file = pd.read_csv(pth)
    result = [float(file.keys()[0])]
    result.extend(list(file[file.keys()[0]]))
    result = [float(x) for x in result]
    for index in range(len(result)):
        if index not in non_vis_idx:
            result[index] = 0
    result = np.array(result)
    cate_num = random.randint(4,8)
    topk = heapq.nlargest(k*8, range(len(result)), result.take)
    return topk

def main(run_index, k,):
    neurosynthflag = True
    if neurosynthflag:
        pth = r'.\result_cv\TNNLS\Neurosynth_groundtruth\working memory.csv'
        roi = neurocircuit_export_top_k(pth, k)
    else:
        # pth = r'.\result_cv\MMP\mlp_mixer\saliency_result\yeo17\mix'
        # pth = r'.\result_cv\MMP\stgcn\saliency_result\yeo17\mix_new'
        pth = r'.\result_cv\MMP\stpgcn\saliency_result\yeo17\mix'
        roi = export_top_k(cate_list, k, pth)

    T=15
    num_node = 360

    gfc_pth = r".\fc\mmp_gfc_360.npy"

    task_dict, task_len_dict, key_index = dataloader.dataloader(cate='wm')
    dataset = dataloader.dataset(task_dict, task_len_dict, gfc_pth, 15, run_index, key_index)
    train_dataset_x, train_dataset_y, test_dataset_x, test_dataset_y = dataset.total_train_x.cpu(), \
                                                                       torch.max(dataset.total_train_y.cpu().data, 1)[1].numpy(), \
                                                                       dataset.total_test_x, \
                                                                       torch.max(dataset.total_test_y.cpu().data, 1)[1].numpy()
    train_mask = torch.tensor(torch.zeros([train_dataset_x.shape[0],1,T,num_node]))
    for item in roi:
        train_mask[:,:,:,int(item)] = torch.ones([train_dataset_x.shape[0],1,T])

    test_mask = torch.tensor(torch.zeros([test_dataset_x.shape[0],1,T,num_node]))
    for item in roi:
        test_mask[:,:,:,int(item)] = torch.ones([test_dataset_x.shape[0],1,T])


    train_dataset_x = torch.mul(train_mask, train_dataset_x.cpu()).reshape([train_dataset_x.shape[0], -1]).numpy()
    test_dataset_x = torch.mul(test_mask, test_dataset_x.cpu()).reshape([test_dataset_x.shape[0], -1]).numpy()

    train_dataset_x = np.nan_to_num(train_dataset_x)
    test_dataset_x = np.nan_to_num(test_dataset_x)

    model = SVC(max_iter=64, decision_function_shape='ovr')
    model.fit(train_dataset_x, train_dataset_y)

    t = time.time()
    pred = model.predict(test_dataset_x)
    print("ACC: {}".format(sum(test_dataset_y == pred) / test_dataset_y.shape[0]))
    print("Time cost", time.time() - t)
    return test_dataset_y, pred

if __name__ == '__main__':

    top_k_roi = [1,2,3,4,5,6,7,8,9,10]

    acc_list = []
    macro_pre_list = []
    macro_f1_list = []
    macro_r_list = []
    for k in top_k_roi:
        print("========================================================================")
        print("================== Start #k={} of SVM. =====================".format(k))
        print("========================================================================")

        for run_index in range(10):
            print("Start # {} Training SVM.".format(run_index+1))
            total_target_y, total_pred_y = main(run_index + 1, k)
            acc = round(metrics.accuracy_score(total_target_y,total_pred_y), 3)
            macro_pre = round(metrics.precision_score(total_target_y,total_pred_y,average='macro'), 3)
            macro_f1 = round(metrics.f1_score(total_target_y,total_pred_y,average='macro'), 3)
            macro_r = round(metrics.recall_score(total_target_y,total_pred_y,average='macro'), 3)
            acc_list.append(acc)
            macro_pre_list.append(macro_pre)
            macro_f1_list.append(macro_f1)
            macro_r_list.append(macro_r)

    file = pd.DataFrame({'acc': acc_list,
                         'macro_pre': macro_pre_list,
                         'macro_r': macro_r_list,
                         'macro_f1': macro_f1_list})
    save_pth = r'.\result_cv\MMP\WM_task'
    file.to_csv(os.path.join(save_pth, 'top_k_roi_svm_neurosynth.csv'), encoding='utf8')

# for item in a:
#     b[:,:,:,int(item)] = torch.ones([batch_size,1,T])