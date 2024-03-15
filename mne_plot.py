# !/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Author              : Ziyuan Ye
@Email               : ziyuanye9801@gmail.com
'''

import matplotlib
import mne
import time
import os
import numpy as np
import warnings

import matplotlib.pyplot as plt
from matplotlib import cm


def normalize_data(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))


def scale_data(data):
    return (data) * (1 / np.max(data))


def color_map_color(value, cmap_name='RdBu_r', vmin=0, vmax=1):
    # norm = plt.Normalize(vmin, vmax)
    norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
    cmap = cm.get_cmap(cmap_name)  # PiYG
    rgb = cmap(norm(abs(value)))[:3]  # will return rgba, we take only first 3 so we get rgb
    color = matplotlib.colors.rgb2hex(rgb)
    return color


def plot_label(weights, brain, labels, mmp_label):
    for i, weight in enumerate(weights):
        for label in labels:
            if label.name == mmp_label[i]:
                # if weight > 0.motor:
                #     print(weight, label.name)
                temp_label = label
                color = color_map_color(weight)
                brain.add_label(temp_label, color=color, borders=False)
    return


def save_image(task, store_name, pos, f_weight):

    mmp_label = np.load(r'.\mmp_label.npy')[:360]

    Brain = mne.viz.get_brain_class()

    subjects_dir = mne.datasets.sample.data_path() + '/subjects'
    print('subjects_dir:', subjects_dir)
    mne.datasets.fetch_hcp_mmp_parcellation(subjects_dir=subjects_dir, verbose=True)

    mne.datasets.fetch_aparc_sub_parcellation(subjects_dir=subjects_dir, verbose=True)

    labels = mne.read_labels_from_annot('fsaverage', 'HCPMMP1', pos, subjects_dir=subjects_dir)[:360]
    print(labels)

    brain = Brain('fsaverage', pos, 'inflated', subjects_dir=subjects_dir, offset=False, cortex='low_contrast',
                  background='white', size=(6000, 4500))
    brain.add_annotation('HCPMMP1', borders=True, alpha=0.8)

    plot_label(f_weight, brain, labels, mmp_label)
    # aud_label = [label for label in labels if label.name == 'L_A1_ROI-lh'][0]
    # medial lateral
    # brain.show_view('lateral')
    brain.show_view('lateral')
    if not os.path.exists('result_cut_visual/miccai/' + task + '/'):
        os.makedirs('result_cut_visual/miccai/' + task + '/')
    brain.save_image('result_cut_visual/miccai/' + task + '/' + store_name + pos + '_lateral_reds.png')
    brain.show_view('medial')
    brain.save_image('result_cut_visual/miccai/' + task + '/' + store_name + pos + '_medial_reds.png')
    brain.show_view('dorsal')
    brain.save_image('result_cut_visual/miccai/' + task + '/' + store_name + pos + '_dorsal_reds.png')
    brain.show_view('ventral')
    brain.save_image('result_cut_visual/miccai' + task + '/' + store_name + pos + '_ventral_reds.png')
    brain.close()


def plot_task_topk(weight, topk):
    f_weight = [0 if i < 0 else i for i in weight]
    f_weight = normalize_data(f_weight)
    idx = f_weight.argsort()[(-1 * topk):][::-1]
    for index, value in enumerate(f_weight):
        if index in idx:
            f_weight[index] = 1
        else:
            f_weight[index] = 0
    save_image(task, 'top_' + str(topk) + '_', 'lh', f_weight)
    save_image(task, 'top_' + str(topk) + '_', 'rh', f_weight)


def plot_task(weight, name):
    f_weight = [0 if i < 0 else i for i in weight]
    f_weight = normalize_data(f_weight)
    save_image(task, name, 'lh', f_weight)
    save_image(task, name, 'rh', f_weight)


def plot_time_task(weight):
    time = weight.shape[0]
    for i in range(time):
        name = 'time_' + str(i + 1) + '_'
        plot_task(weight[i], name)


warnings.filterwarnings('ignore')
task_ls = ['0bk_body', '0bk_faces', '0bk_places', '0bk_tools', '2bk_body', '2bk_faces', '2bk_places', '2bk_tools',
             'loss', 'win', 't', 'lf', 'lh', 'rh', 'rf', 'math', 'story', 'mental', 'rnd', 'match', 'relation', 'fear',
             'neut']
pth = r'.\result_cv\saliency_result\yeo17\mix'

for task in task_ls:
    print('processing task:', task)
    # weight = np.load('result/vis_weight/' + task + '.npy')
    weight = np.loadtxt(os.path.join(pth, '{}.csv'.format(task)))[:360]
    plot_task(weight, task)
    # weight = np.load('result/weight/' + task + '.npy')
    # plot_task(task)
    # plot_task_topk(task, 10)
    # plot_task_topk(task, 20)
    # plot_task_topk(task, 30)
    # plot_task_topk(task, 40)
    # plot_task_topk(task, 100)
# f_weight = ScaleData(f_weight)
# print(f_weight)

# array = [0 for x in range(360)]
#
# with open('dataset/indexl.txt') as f:
#     for line in f:
#         region = line.split()[0]
#         index = int(line.split()[motor])
#         array[index - motor] = region
#
# with open('dataset/indexr.txt') as f:
#     for line in f:
#         region = line.split()[0]
#         index = int(line.split()[motor])
#         array[index +179] = region
#
# for a in array:
#     print(a)
#
# np.save('dataset/mmp_label', array)
# lh rh





