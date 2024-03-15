# !/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Author              : Ziyuan Ye
@Email               : ziyuanye9801@gmail.com
'''
import numpy as np
import torch

import pandas as pd

from sklearn.svm import SVC
from sklearn import metrics
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier
from sklearn.multiclass import OneVsRestClassifier

import dataloader

import time
import os
import sys


def main(run_index: int, min_l: int, model_name: str):
    """================ Hyper-parameters Setting ================="""
    best_test_acc = 0
    # TODO ================= check ================
    gfc_pth = r".\fc\mmp_gfc_360.npy"
    # gfc_pth = r".\fc\mmp_gfc_308.npy"
    # gfc_pth = r".\fc\aal_gfc_116.npy"
    # gfc_pth = r".\fc\aal_gfc_90.npy"
    cate = 'wm'
    print("Current cate is {}".format(cate))
    task_dict, task_len_dict, key_index = dataloader.dataloader(cate)
    dataset = dataloader.dataset(task_dict, task_len_dict, gfc_pth, min_l, run_index, key_index)
    train_x, train_y, test_x, test_y = dataset.total_train_x.reshape((dataset.total_train_x.shape[0], -1)).cpu().numpy(), \
                                        torch.max(dataset.total_train_y.cpu().data, 1)[1].numpy(), \
                                        dataset.total_test_x.reshape((dataset.total_test_x.shape[0], -1)).cpu().numpy(),\
                                        torch.max(dataset.total_test_y.cpu().data, 1)[1].numpy()
    test_x = np.nan_to_num(test_x)
    train_x = np.nan_to_num(train_x)

    t = time.time()
    if model_name == 'svm':
        model = SVC(max_iter=64, decision_function_shape='ovr')
    elif model_name == 'xgboost':
        train_y = OneHotEncoder(sparse=False).fit_transform(train_y.reshape(-1, 1))
        print(train_y[:5])
        sys.exit()
        test_y = OneHotEncoder(sparse=False).fit_transform(test_y.reshape(-1, 1))
        model = OneVsRestClassifier(XGBClassifier(), n_jobs=-1)
    elif model_name == 'mlp':
        pass
    model.fit(train_x, train_y)

    pred = model.predict(test_x)
    print("ACC: {}".format(sum(test_y == pred) / test_x.shape[0]))
    print("Time cost", time.time() - t)
    return test_y, pred


if __name__ == '__main__':

    model_name = "svm"
    pth = r'result_cv/TNNLS/metric/AAL'

    nframe = [15]

    acc_list = []
    micro_pre_list = []
    macro_pre_list = []
    micro_f1_list = []
    macro_f1_list = []
    micro_r_list = []
    macro_r_list = []
    for run_index in range(10):
        print("========================================================================")
        print("================== Start # {} run of SVM. =====================".format(run_index + 1))
        print("========================================================================")

        for frame in nframe:
            print("Start # {} frame of Training SVM.".format(frame))
            total_target_y, total_pred_y = main(run_index + 1, frame, model_name)
            acc = round(metrics.accuracy_score(total_target_y,total_pred_y), 3)
            macro_pre = round(metrics.precision_score(total_target_y,total_pred_y,average='macro'), 3)
            macro_f1 = round(metrics.f1_score(total_target_y,total_pred_y,labels=list(range(23)),average='macro'), 3)
            macro_r = round(metrics.recall_score(total_target_y,total_pred_y,average='macro'), 3)
            acc_list.append(acc)
            macro_pre_list.append(macro_pre)
            macro_f1_list.append(macro_f1)
            macro_r_list.append(macro_r)

    file = pd.DataFrame({'acc': acc_list,
                         'macro_pre': macro_pre_list,
                         'macro_f1': macro_f1_list,
                         'macro_r': macro_r_list})
    file.to_csv(os.path.join(pth, '{}_new.csv'.format(model_name)), encoding='utf8')