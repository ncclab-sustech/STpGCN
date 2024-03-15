# !/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Author              : Ziyuan Ye
@Email               : ziyuanye9801@gmail.com
'''
import random

import torch
from torch.utils.data import TensorDataset
from  torch.utils.data.dataset import ConcatDataset
from torch.utils.data import Dataset

from sklearn.model_selection import train_test_split

from graphloader import build_knngraph_from_weight_matrix

import random

import numpy as np

import os
import sys
import time
import warnings
warnings.filterwarnings("ignore")
warnings.simplefilter(action='ignore', category=FutureWarning)

graph_dir = r'..\dataset\HCP\processed'

rest_motor_fc = r'fc_rest_motor.npy'

# TODO================================ Check if use cut visual =====================================
# fmri_dir = r'T:\HCP_fMRI_event\all_fix15_cut_visual'
# fmri_dir = r'/home/ncclab306/database7/HCP_fMRI_event/all_fix15'
fmri_dir = r'T:\HCP_fMRI_event\all_fix15'
# fmri_dir = r'S:\HCP_task\subtask\all_AAL'

cate_list = ['0bk_body', '0bk_faces', '0bk_places', '0bk_tools', '2bk_body', '2bk_faces', '2bk_places', '2bk_tools',
             'loss', 'win', 't', 'lf', 'lh', 'rh', 'rf', 'math', 'story', 'mental', 'rnd', 'match', 'relation', 'fear',
             'neut']

wm_cate_list = ['0bk_body', '0bk_faces', '0bk_places', '0bk_tools', '2bk_body', '2bk_faces', '2bk_places', '2bk_tools']

# cate_list = ['0bk_body', '0bk_faces', '0bk_places', '0bk_tools', '2bk_body', '2bk_faces', '2bk_places', '2bk_tools',
#              't', 'lf', 'lh', 'rh', 'rf', 'math', 'story', 'mental', 'rnd', 'match', 'relation', 'fear',
#              'neut']

k_knn = 5  # graph setting


class Subset(list):
    """
    Subset of a dataset at specified indices.
    Arguments:
        dataset (list): The whole Dataset
        indices (sequence): Indices in the whole set selected for subset
    """
    def __init__(self, dataset, indices):
        super().__init__()
        self.dataset = dataset
        self.indices = indices

    def get_len(self):
        self.get_data()
        if self.indices.shape == ():
            print('this happens: Subset')
            return 1
        else:
            return len(self.indices)

    def get_data(self):
        re = []
        for idx in range(len(self.dataset)):
            if idx in self.indices:
                re.append(self.dataset[idx])
        return re


def dataloader(cate: str):
    start_time = time.time()
    task_dict = {}
    task_len_dict = {}
    print("Loading time series...")
    if cate == 'all':
        for i in cate_list:
            task_dict[i] = np.load(os.path.join(fmri_dir, i + '.npy'))
            task_len_dict[i] = len(task_dict[i])
        print("Time series loading is complete! Takes {} seconds".format(time.time() - start_time))
        return task_dict, task_len_dict, 0
    elif cate == 'wm':
        for i in wm_cate_list:
            task_dict[i] = np.load(os.path.join(fmri_dir, i + '.npy'))
            task_len_dict[i] = len(task_dict[i])
        print("Time series loading is complete! Takes {} seconds".format(time.time() - start_time))
        return task_dict, task_len_dict, 0
    else:
        key_index = 0
        for i in cate_list:
            if i == cate:
                task_dict[i] = np.load(os.path.join(fmri_dir, i+'.npy'))
                task_len_dict[i] = len(task_dict[i])
                print("Time series loading is complete! Takes {} seconds".format(time.time() - start_time))
                return task_dict, task_len_dict, key_index
            key_index += 1


class dataset(object):
    def __init__(self, time_series, task_len_dict, gfc_pth, min_l, run_index, key_index):
        self.time_series = time_series
        self.nclass = len(time_series.keys())
        # TODO================== Check if modify =========================
        self.num_node = 308
        self.min_l = min_l
        self.run_index = run_index - 1
        self.key_index = key_index
        self.task_len_dict = task_len_dict
        self.all_check_delete_none()
        # self.all_construct_graph()
        self.all_build_graph(pth=gfc_pth)
        start_time = time.time()
        print("Loading time series...")
        self.all_train_test_build_dataset()

        print("Dataset construction is complete! Takes {} seconds".format(time.time() - start_time))

    def all_check_delete_none(self):
        for key in self.time_series.keys():
            for i in range(self.time_series[key].shape[0]):
                if self.time_series[key][i] is None:
                    self.time_series[key][i] = None
                    print("Check None when key = {}, i = {}".format(key, i))

    def all_construct_graph(self):
        sum_ = 0
        count = 0
        for i in range(1075):
            sum_ += self.fc[i][:self.num_node, :self.num_node]
            count += 1
        gfc = sum_ / count
        np.save(r"..\dataset\HCP\processed\gfc.npy", gfc)

    def all_build_graph(self, pth):

        gfc = np.load(pth)
        self.g = build_knngraph_from_weight_matrix(gfc, k=k_knn)

    def all_train_test_build_dataset(self):
        key_index = self.key_index

        total_train_x = []
        total_train_y = []
        total_test_x = []
        total_test_y = []

        # TODO================================ check =====================================
        robust = True
        # For WM sub task classifier
        if self.nclass == 8:
            # 0/2 bk label: 0->[1,0]; 2->[0,1]
            # stimulation label: body->[1, 0, 0, 0]; faces->[0, 1, 0, 0]; places->[0, 0, 1, 0]; tools->[0, 0, 0, 1]
            # TODO================================ check =====================================
            wm_2flag = False
            if wm_2flag:
                for key in self.time_series.keys():
                    total_x = []
                    for task_index in range(len(self.time_series[key])):
                        total_x.append(self.time_series[key][task_index][np.newaxis, :])
                    if key[0] == '0':
                        task_label = [1, 0]
                    elif key[0] == '2':
                        task_label = [1, 0]

                    valid_length = int(len(total_x) * 0.1)
                    valid_idx = np.arange(len(total_x))[self.run_index * valid_length:(self.run_index + 1) * valid_length]
                    train_idx = np.concatenate(
                        [np.arange(len(total_x))[:self.run_index * valid_length],
                         np.arange(len(total_x))[(self.run_index + 1) * valid_length:]],
                        axis=0)

                    cur_train = Subset(total_x, train_idx).get_data()
                    cur_test = Subset(total_x, valid_idx).get_data()

                    cur_train_label = [task_label for x in range(len(cur_train))]
                    cur_test_label = [task_label for x in range(len(cur_test))]

                    total_train_x.extend(cur_train)
                    total_train_y.extend(cur_train_label)
                    total_test_x.extend(cur_test)
                    total_test_y.extend(cur_test_label)

            else:
                for key in self.time_series.keys():
                    total_x = []
                    for task_index in range(len(self.time_series[key])):
                        total_x.append(self.time_series[key][task_index][np.newaxis, :])
                    if key[-5:] == '_body':
                        task_label = [1, 0, 0, 0]
                    elif key[-5:] == 'faces':
                        task_label = [0, 1, 0, 0]
                    elif key[-5:] == 'laces':
                        task_label = [0, 0, 1, 0]
                    elif key[-5:] == 'tools':
                        task_label = [0, 0, 0, 1]

                    valid_length = int(len(total_x) * 0.1)
                    valid_idx = np.arange(len(total_x))[self.run_index * valid_length:(self.run_index + 1) * valid_length]
                    train_idx = np.concatenate(
                        [np.arange(len(total_x))[:self.run_index * valid_length],
                         np.arange(len(total_x))[(self.run_index + 1) * valid_length:]],
                        axis=0)

                    cur_train = Subset(total_x, train_idx).get_data()
                    cur_test = Subset(total_x, valid_idx).get_data()

                    cur_train_label = [task_label for x in range(len(cur_train))]
                    cur_test_label = [task_label for x in range(len(cur_test))]

                    total_train_x.extend(cur_train)
                    total_train_y.extend(cur_train_label)
                    total_test_x.extend(cur_test)
                    total_test_y.extend(cur_test_label)
        # TODO================================ Check=====================================
        elif robust:
            for key in self.time_series.keys():
                total_x = []
                for task_index in range(len(self.time_series[key])):
                    total_x.append(self.time_series[key][task_index][np.newaxis, :])
                task_label = [0] * key_index + [1] + [0] * (self.nclass - key_index - 1)
                valid_length = int(len(total_x) * 0.90)
                valid_idx = np.arange(len(total_x))[:valid_length]
                train_idx = np.arange(len(total_x))[valid_length:]
                seed = self.run_index
                random.seed(seed)
                random.shuffle(total_x)
                cur_train = Subset(total_x, train_idx).get_data()
                cur_test = Subset(total_x, valid_idx).get_data()

                cur_train_label = [task_label for x in range(len(cur_train))]
                cur_test_label = [task_label for x in range(len(cur_test))]

                total_train_x.extend(cur_train)
                total_train_y.extend(cur_train_label)
                total_test_x.extend(cur_test)
                total_test_y.extend(cur_test_label)
                key_index += 1
        else:
            # TODO================================ Check WM =====================================
            wm_flag = False
            wm_2flag = False
            for key in self.time_series.keys():
                total_x = []
                for task_index in range(len(self.time_series[key])):
                    total_x.append(self.time_series[key][task_index][np.newaxis, :])
                if wm_flag:
                    if wm_2flag:
                        if key[0] == '0':
                            task_label = [1, 0]
                        elif key[0] == '2':
                            task_label = [1, 0]
                    else:
                        if key[-5:] == '_body':
                            task_label = [1, 0, 0, 0]
                        elif key[-5:] == 'faces':
                            task_label = [0, 1, 0, 0]
                        elif key[-5:] == 'laces':
                            task_label = [0, 0, 1, 0]
                        elif key[-5:] == 'tools':
                            task_label = [0, 0, 0, 1]
                else:
                    task_label = [0] * key_index + [1] + [0] * (self.nclass - key_index - 1)

                valid_length = int(len(total_x) * 0.1)
                valid_idx = np.arange(len(total_x))[self.run_index * valid_length:(self.run_index + 1) * valid_length]
                train_idx = np.concatenate(
                    [np.arange(len(total_x))[:self.run_index * valid_length], np.arange(len(total_x))[(self.run_index + 1) * valid_length:]],
                    axis=0)

                cur_train = Subset(total_x, train_idx).get_data()
                cur_test = Subset(total_x, valid_idx).get_data()

                cur_train_label = [task_label for x in range(len(cur_train))]
                cur_test_label = [task_label for x in range(len(cur_test))]

                total_train_x.extend(cur_train)
                total_train_y.extend(cur_train_label)
                total_test_x.extend(cur_test)
                total_test_y.extend(cur_test_label)
                key_index += 1

        self.total_train_x = torch.tensor([x[:, :self.min_l, :self.num_node] for x in total_train_x]).cuda()
        self.total_train_y = torch.Tensor(total_train_y).cuda()
        self.total_test_x = torch.tensor([x[:, :self.min_l, :self.num_node] for x in total_test_x]).cuda()
        self.total_test_y = torch.Tensor(total_test_y).cuda()

        print(len(total_train_x), len(total_train_y), len(total_test_x), len(total_test_y))

        self.train_dataset = TensorDataset(self.total_train_x,self. total_train_y)
        self.test_dataset = TensorDataset(self.total_test_x, self.total_test_y)
        self.val_dataset = self.test_dataset
        self.total_dataset = ConcatDataset([self.train_dataset, self.test_dataset])
        # val_size = int(1 / 3 * len(self.test_dataset))
        # rest_size = len(self.test_dataset) - val_size
        # self.val_dataset, _ = torch.utils.data.random_split(self.test_dataset, [val_size, rest_size])

    def all_build_dataset(self):
        torch.manual_seed(0)
        key_index = self.key_index
        keys = self.time_series.keys()
        total_x = []
        total_y = []
        for key in self.time_series.keys():
            task_label = [0] * key_index + [1] + [0] * (self.nclass - key_index)
            for task_index in range(len(self.time_series[key])):
                total_x.append(self.time_series[key][task_index][np.newaxis, :])
            total_y.extend([task_label for x in self.time_series[key]])
            key_index += 1
        total_x = torch.tensor(total_x).cuda()  # [batch, 1, T, n]
        total_y = torch.Tensor(total_y).cuda()

        total_data = TensorDataset(total_x, total_y)
        train_size = int(0.7 * len(total_data))
        test_size = len(total_data) - train_size
        self.train_dataset, self.test_dataset = torch.utils.data.random_split(total_data, [train_size, test_size])
        val_size = int(1 / 3 * len(self.test_dataset))
        rest_size = len(self.test_dataset) - val_size
        self.val_dataset, _ = torch.utils.data.random_split(self.test_dataset, [val_size, rest_size])

    # def all_train_test_build_dataset(self):
    #     torch.manual_seed(0)
    #     seed = 2333
    #     key_index = 0
    #     keys = self.time_series.keys()
    #
    #     total_train_x = []
    #     total_train_y = []
    #     total_test_x = []
    #     total_test_y = []
    #     for key in self.time_series.keys():
    #         total_x = []
    #         for task_index in range(len(self.time_series[key])):
    #             total_x.append(self.time_series[key][task_index][np.newaxis, :])
    #         task_label = [0] * key_index + [1] + [0] * (len(list(keys)) - key_index - 1)
    #
    #         cur_train, cur_test = train_test_split(total_x, test_size=0.3, random_state=seed)
    #         cur_train_label = [task_label for x in range(len(cur_train))]
    #         cur_test_label = [task_label for x in range(len(cur_test))]
    #
    #         total_train_x.extend(cur_train)
    #         total_train_y.extend(cur_train_label)
    #         total_test_x.extend(cur_test)
    #         total_test_y.extend(cur_test_label)
    #         key_index += 1
    #     total_train_x = torch.tensor([x[:, :self.min_l, :] for x in total_train_x]).cuda()
    #     total_train_y = torch.Tensor(total_train_y).cuda()
    #     total_test_x = torch.tensor([x[:, :self.min_l, :] for x in total_test_x]).cuda()
    #     total_test_y = torch.Tensor(total_test_y).cuda()
    #
    #     self.train_dataset = TensorDataset(total_train_x, total_train_y)
    #     self.test_dataset = TensorDataset(total_test_x, total_test_y)
    #     val_size = int(1 / 3 * len(self.test_dataset))
    #     rest_size = len(self.test_dataset) - val_size
    #     self.val_dataset, _ = torch.utils.data.random_split(self.test_dataset, [val_size, rest_size])





if __name__ == '__main__':
    # TODO================================ Check if use cut visual =====================================
    # gfc_pth = r".\fc\mmp_gfc_360.npy"
    # gfc_pth = r".\fc\mmp_gfc_308.npy"
    # gfc_pth = r".\fc\aal_gfc_116.npy"
    gfc_pth = r".\fc\aal_gfc_90.npy"
    task_dict, task_len_dict, key_index = dataloader('all')
    print(task_dict[cate_list[1]].shape)
    a = dataset(task_dict, task_len_dict, gfc_pth, 15, 8, key_index)

