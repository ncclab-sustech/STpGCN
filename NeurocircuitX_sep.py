# !/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Author              : Ziyuan Ye
@Email               : ziyuanye9801@gmail.com
'''

import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.nn as nn

from stgcn import STpGCN, STGCN
from gcn import GCN
from gat import GAT
from mlp_mixer_pytorch import MLPMixer

import dataloader

import numpy as np

import gc
import sys
import time
import os
import heapq
import copy

import hcp_utils as hcp


def z_score(data: np.ndarray):
    std = np.std(data)
    mean = np.mean(data)
    data = (data-mean)/std

    return data.reshape(-1, 1)


def stdlize(data: np.ndarray):
    data = np.array([(_data - np.min(data)) / (np.max(data) - np.min(data))
              for _data in data]).reshape(-1, 1)
    return data


def fisher(data: np.ndarray):
    data = np.arctanh(data)
    largest_index = heapq.nlargest(10, range(data.shape[0]), data.take)
    max_ = -1
    copy_data = copy.deepcopy(largest_index)
    copy_data.sort(reverse=True)
    for d in copy_data:
        if d != np.inf and d > max_:
            max_ = d
    for i in range(data.shape[0]):
        if data[i] == np.inf:
            data[i] = max_
    return data.reshape(-1, 1)


def ablation_acc(data: np.ndarray):
    for i in range(360):
        data[i] = 1 - data[i]
    return data.reshape(-1, 1)

atlas_name = 'yeo17'
gc.collect()

model_structure = 'TSTNTSTN'
channels = [1, 32, 16, 32, 32, 16, 32]  # Need to modify when model structure change

# =============== Training params ===============
lr = 1e-3
batch_size = 25
epochs = 20

drop_prob = 0.5
k_knn = 5
nframe = 15
loss_alpha = 0.0005

# =============== Device params ===============
DisableGPU = False
device = torch.device("cuda") if torch.cuda.is_available() and not DisableGPU else torch.device("cpu")

start_time = time.time()

# TODO================================ Check the category of used dataset =====================================

cate_list = ['0bk_body', '0bk_faces', '0bk_places', '0bk_tools', '2bk_body', '2bk_faces', '2bk_places', '2bk_tools',
             'loss', 'win', 't', 'lf', 'lh', 'rh', 'rf', 'math', 'story', 'mental', 'rnd', 'match', 'relation',
             'fear', 'neut']

# cate_list = ['0bk_body', '0bk_faces', '0bk_places', '0bk_tools', '2bk_body', '2bk_faces', '2bk_places', '2bk_tools']

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

graph_dir = r'..\dataset\HCP\processed'

# TODO================================ Check graph =====================================
gfc_pth = r"./fc/mmp_gfc_360.npy"

for cate in cate_list:
    # TODO================================ Check save path =====================================
    # pth = r'.\result_cv\gcn\saliency_result\yeo17\ablation'
    # pth = r'.\result_cv\gcn\saliency_result\yeo17\keep'

    # pth = r'.\result_cv\gat\saliency_result\yeo17\ablation'
    # pth = r'.\result_cv\gat\saliency_result\yeo17\keep'

    # pth = r'.\result_cv\stgcn\saliency_result\yeo17\ablation'
    # pth = r'.\result_cv\stgcn\saliency_result\yeo17\keep'

    # pth = r'.\result_cv\stpgcn\saliency_result\yeo17\ablation'
    # pth = r'.\result_cv\stpgcn\saliency_result\yeo17\keep'

    pth = r'./result_cv/MMP/mlp_mixer/saliency_result/yeo17/ablation'
    # pth = r'./result_cv/MMP/mlp_mixer/saliency_result/yeo17/keep'

    # pth = r'.\result_cv\WM_task\diff_stimuli\gcn\saliency_result\yeo17\ablation'
    # pth = r'.\result_cv\WM_task\diff_stimuli\gcn\saliency_result\yeo17\keep'

    # pth = r'.\result_cv\WM_task\diff_stimuli\gat\saliency_result\yeo17\ablation'
    # pth = r'.\result_cv\WM_task\diff_stimuli\gat\saliency_result\yeo17\keep'
    #
    # pth = r'.\result_cv\WM_task\diff_stimuli\stgcn\saliency_result\yeo17\ablation'
    # pth = r'.\result_cv\WM_task\diff_stimuli\stgcn\saliency_result\yeo17\keep'

    # pth = r'.\result_cv\WM_task\diff_stimuli\stpgcn\saliency_result\yeo17\ablation'
    # pth = r'.\result_cv\WM_task\diff_stimuli\stpgcn\saliency_result\yeo17\keep'
    cate_result = np.array([])

    for model_index in range(1, 11):
        print("========================================================================")
        print("============== Start calculate {} task salicency map. ===============".format(cate))
        print("========================================================================")
        gc.collect()
        task_dict, task_len_dict, key_index = dataloader.dataloader(cate=cate)
        dataset = dataloader.dataset(task_dict, task_len_dict, gfc_pth, 15, model_index, key_index)
        train_dataset, test_dataset, g = dataset.train_dataset, dataset.test_dataset, dataset.g
        g = g.to(device)

        num_node = dataset.num_node
        window = dataset.min_l
        horizon = int(window / 4)  # Need to modify when model structure change
        num_class = 23

        # TODO================================ Check the well trained models =====================================
        # model_pth = r'checkpoints_cut_visual/gcn/run_1/gcn-length=15.pt'
        # model_pth = r'checkpoints_cut_visual/gat/run_1/gat-length=15.pt'
        # model_pth = r'checkpoints_cut_visual/stgcn/run_1/stgcn-length=15.pt'
        # model_pth = r'checkpoints_cut_visual\stpgcn\run_1\stpgcn-length=15.pt'
        model_pth = r'checkpoints_cv/MMP/mlp_mixer/run_{}/mlp_mixer-length=15.pt'.format(model_index)

        # model_pth = r'checkpoints_cut_visual\WM_task\diff_stimuli/gcn/run_{}/gcn-length=15.pt'.format(model_index)
        # model_pth = r'checkpoints_cut_visual\WM_task\diff_stimuli/gat/run_{}/gat-length=15.pt'.format(model_index)
        # model_pth = r'checkpoints_cut_visual\WM_task\diff_stimuli/stgcn/run_{}/stgcn-length=15.pt'.format(model_index)
        # model_pth = r'checkpoints_cut_visual\WM_task\diff_stimuli\stpgcn\run_{}\stpgcn-length=15.pt'.format(model_index)

        model_config = torch.load(model_pth)

        # TODO================================ Check model=====================================
        # model = GCN(c_in=1, c_hid=16, c_out=1, g=g, ts=window, nclass=num_class, num_node=num_node).to(device)
        # model = GAT(g, nlayers=2, in_dim=1, nhidden=32, nclass=num_class, heads=[4, 4],
        #                             activation=nn.ReLU(inplace=True), feat_drop=0.3, attn_drop=0.3,
        #                             negative_slope=0.2, residual=False, T=window, num_node=num_node).to(device)
        # model = STGCN(channels, window, horizon, num_node, g, drop_prob, model_structure, num_class).to(device)
        model = STpGCN(channels, window, horizon, num_node, g, drop_prob, model_structure, num_class, True).to(device)
        model = MLPMixer(image_size=(15, num_node),
                         channels=1,
                         patch_size=3,
                         dim=512,
                         depth=12,
                         num_classes=num_class,
                         dropout=drop_prob
                         ).to(device)

        model.load_state_dict(model_config, strict=False)
        test_iter = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
        save_result = []

        model.eval()

        bnet_list = []
        if atlas_name == 'yeo7':
            for network_index in range(7):
                # small_label = np.array([0] * network_index + [1] + (6 - network_index) * [0]).reshape(1, -1)
                # # small_label = np.array([1] * network_index + [0] + (6 - network_index) * [1]).reshape(1, -1)
                # yeo_label = hcp.unparcellate(small_label, hcp.yeo7)
                # mmp_360 = torch.Tensor(hcp.parcellate(yeo_label, hcp.mmp).squeeze(0)[non_vis_idx].reshape(-1, 1)).cuda()
                # bnet_list.append(mmp_360)
                pass
        elif atlas_name == 'yeo17':
            for network_index in range(17):
                # TODO================================ Check atlas label =====================================
                # small_label = np.array([0] * network_index + [1] + (16 - network_index) * [0]).reshape(1, -1)
                small_label = np.array([1] * network_index + [0] + (16 - network_index) * [1]).reshape(1, -1)
                yeo_label = hcp.unparcellate(small_label, hcp.yeo17)
                a = hcp.parcellate(yeo_label, hcp.mmp).squeeze(0)[:360]
                # TODO================================ Check if use mlp_mixer =====================================
                # mmp_360 = torch.Tensor(hcp.parcellate(yeo_label, hcp.mmp).squeeze(0)[non_vis_idx].reshape(-1, 1)).cuda()
                mmp_360 = torch.Tensor(hcp.parcellate(yeo_label, hcp.mmp).squeeze(0)[:360].reshape(-1, 1)).cuda()
                bnet_list.append(mmp_360)

        for keep_index in range(len(bnet_list)):
            gc.collect()
            l_sum, n = 0.0, 0
            total_pred_y = []
            total_target_y = []
            with torch.no_grad():
                for x, y in test_iter:
                    clone_x = x.clone()[:, :, :, :].to(torch.float32)
                    clone_x = torch.mul(clone_x.permute(0, 1, 3, 2), bnet_list[keep_index]).permute(0, 1, 3, 2)
                    output = F.log_softmax(model(clone_x.cuda()), dim=-1)
                    l = F.nll_loss(output, y.argmax(dim=-1))

                    l2_reg = torch.tensor(0.).cuda()
                    for name, param in model.named_parameters():
                        if 'bias' not in name:
                            l2_reg += loss_alpha * torch.norm(param)
                    l += l2_reg

                    l_sum += l.item() * y.shape[0]
                    n += y.shape[0]

                    pred_y = torch.max(output.cpu(), 1)[1].numpy().squeeze().tolist()
                    target_y = torch.max(y.cpu().data, 1)[1].numpy().tolist()

                    total_pred_y += pred_y
                    total_target_y += target_y

                total_pred_y = np.array(total_pred_y)
                total_target_y = np.array(total_target_y)

                # TODO================================ Check result =====================================
                result = 1 - (sum(total_target_y == total_pred_y) / n)
                # result = (sum(total_target_y == total_pred_y) / n)

                save_result.append(result)
                print(keep_index, ':', result)

        normd_acc_data = np.array([(_data - np.min(save_result)) / (np.max(save_result) - np.min(save_result))
                                   for _data in save_result]).reshape(1, -1)

        normd_acc_data2yeo = hcp.unparcellate(normd_acc_data, hcp.yeo17)
        normd_acc_data2mmp = hcp.parcellate(normd_acc_data2yeo, hcp.mmp).squeeze(0)
        final_data = stdlize(normd_acc_data2mmp)

        if model_index == 1:
            cate_result = final_data
        else:
            cate_result = np.concatenate((cate_result, final_data), axis=1)

    final_result = cate_result.mean(axis=1).reshape((-1, 1))
    np.savetxt(os.path.join(pth,
                            '{}.csv'.format(cate)),
               final_result,
               delimiter=",")
    print(time.time() - start_time)