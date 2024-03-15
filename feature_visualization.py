# !/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Author              : Ziyuan Ye
@Email               : ziyuanye9801@gmail.com
'''

import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F

from stgcn import STGCN_pyramid_feature, STGCN_pyramid_layer_feature
import dataloader

import numpy as np
from sklearn.manifold import TSNE
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

import gc
import sys
import time
import os


def cal_stpgcn_layer_embed():
    gc.collect()
    pth1 = r'./result_cv/stpgcn/stpgcn_feature1'
    pth2 = r'./result_cv/stpgcn/stpgcn_feature2'
    pth3 = r'./result_cv/stpgcn/stpgcn_feature3'
    model_pth = r'.\checkpoints_cv\stgcnp\run_8\stgcnp-length=15.pt'
    model_config = torch.load(model_pth)
    model_structure = 'TSTNTSTN'
    channels = [1, 32, 16, 32, 32, 16, 32]  # Need to modify when model structure change

    # =============== Training params ===============
    batch_size = 25
    drop_prob = 0.5

    # =============== Device params ===============
    DisableGPU = False
    device = torch.device("cuda") if torch.cuda.is_available() and not DisableGPU else torch.device("cpu")

    start_time = time.time()

    cate_list = ['0bk_body', '0bk_faces', '0bk_places', '0bk_tools', '2bk_body', '2bk_faces', '2bk_places', '2bk_tools',
             'loss', 'win', 't', 'lf', 'lh', 'rh', 'rf', 'math', 'story', 'mental', 'rnd', 'match', 'relation', 'fear',
             'neut']
    gfc_pth = r"..\dataset\HCP\processed\gfc.npy"

    for cate in cate_list:
        print("========================================================================")
        print("============== Start calculate {} task embedding feature. ===============".format(cate))
        print("========================================================================")
        gc.collect()
        min_l = 15
        run_index = 8
        task_dict, task_len_dict, key_index = dataloader.dataloader(cate)
        dataset = dataloader.dataset(task_dict, task_len_dict, gfc_pth, min_l, run_index, key_index)
        train_dataset, test_dataset, total_dataset, g = dataset.train_dataset, \
                                                        dataset.test_dataset, \
                                                        dataset.total_dataset, \
                                                        dataset.g
        g = g.to(device)

        num_node = dataset.num_node
        window = dataset.min_l
        horizon = int(window / 4)  # Need to modify when model structure change
        num_class = 23
        model = STGCN_pyramid_layer_feature(channels,
                                            window,
                                            horizon,
                                            num_node,
                                            g,
                                            drop_prob,
                                            model_structure,
                                            num_class,
                                            True).to(device)
        model.load_state_dict(model_config, strict=False)

        test_iter = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

        save_result1 = None
        save_result2 = None
        save_result3 = None

        model.eval()
        with torch.no_grad():
            for x, y in test_iter:
                gc.collect()
                shape = x.shape[0]
                x = x[:, :, :, :360]
                out1, out2, out3 = model(x.cuda())
                out1 = out1.reshape([shape, -1]).cpu().numpy()
                out2 = out2.reshape([shape, -1]).cpu().numpy()
                out3 = out3.reshape([shape, -1]).cpu().numpy()

                if save_result1 is None:
                    save_result1 = out1
                else:
                    save_result1 = np.concatenate((save_result1, out1), axis=0)

                if save_result2 is None:
                    save_result2 = out2
                else:
                    save_result2 = np.concatenate((save_result2, out2), axis=0)

                if save_result3 is None:
                    save_result3 = out3
                else:
                    save_result3 = np.concatenate((save_result3, out3), axis=0)

        np.save(os.path.join(pth1, cate), save_result1)
        np.save(os.path.join(pth2, cate), save_result2)
        np.save(os.path.join(pth3, cate), save_result3)

        print(time.time()-start_time)

def cal_embed():
    gc.collect()
    pth = r'./result_cv/stpgcn/stpgcn_feature'
    model_pth = r'.\checkpoints_cv\stgcnp\run_8\stgcnp-length=15.pt'
    model_config = torch.load(model_pth)
    model_structure = 'TSTNTSTN'
    channels = [1, 32, 16, 32, 32, 16, 32]  # Need to modify when model structure change

    # =============== Training params ===============
    batch_size = 25
    drop_prob = 0.5

    # =============== Device params ===============
    DisableGPU = False
    device = torch.device("cuda") if torch.cuda.is_available() and not DisableGPU else torch.device("cpu")

    start_time = time.time()

    cate_list = ['0bk_body', '0bk_faces', '0bk_places', '0bk_tools', '2bk_body', '2bk_faces', '2bk_places', '2bk_tools',
             'loss', 'win', 't', 'lf', 'lh', 'rh', 'rf', 'math', 'story', 'mental', 'rnd', 'match', 'relation', 'fear',
             'neut']
    gfc_pth = r"..\dataset\HCP\processed\gfc.npy"

    for cate in cate_list:
        print("========================================================================")
        print("============== Start calculate {} task embedding feature. ===============".format(cate))
        print("========================================================================")
        gc.collect()
        min_l = 15
        run_index = 8
        task_dict, task_len_dict, key_index = dataloader.dataloader(cate)
        dataset = dataloader.dataset(task_dict, task_len_dict, gfc_pth, min_l, run_index, key_index)
        train_dataset, test_dataset, total_dataset, g = dataset.train_dataset, dataset.test_dataset, dataset.total_dataset, dataset.g
        g = g.to(device)

        num_node = dataset.num_node
        window = dataset.min_l
        horizon = int(window / 4)  # Need to modify when model structure change
        num_class = 23
        model = STGCN_pyramid_feature(channels, window, horizon, num_node, g, drop_prob, model_structure, num_class, True).to(device)
        model.load_state_dict(model_config, strict=False)

        test_iter = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

        save_result = None

        model.eval()
        with torch.no_grad():
            for x, y in test_iter:
                gc.collect()
                shape = x.shape[0]
                x = x[:, :, :, :360]
                output = model(x.cuda()).reshape([shape, -1]).cpu().numpy()

                if save_result is None:
                    save_result = output
                else:
                    save_result = np.concatenate((save_result, output), axis=0)
        np.save(os.path.join(pth, cate), save_result)

        print(time.time()-start_time)


def save_feature():
    gc.collect()
    pth = r'./result_cv/stgcnp/original_feature'
    model_pth = r'.\checkpoints_cv\stpgcn\run_8\stgcnp-length=15.pt'
    model_config = torch.load(model_pth)
    model_structure = 'TSTNTSTN'
    channels = [1, 32, 16, 32, 32, 16, 32]  # Need to modify when model structure change

    # =============== Training params ===============
    lr = 1e-3
    batch_size = 25
    epochs = 20

    drop_prob = 0.5
    k_knn = 5
    nframe = 160
    loss_alpha = 0.0005

    # =============== Device params ===============
    DisableGPU = False
    device = torch.device("cuda") if torch.cuda.is_available() and not DisableGPU else torch.device("cpu")

    start_time = time.time()

    cate_list = ['0bk_body', '0bk_faces', '0bk_places', '0bk_tools', '2bk_body', '2bk_faces', '2bk_places', '2bk_tools',
                 'loss', 'win', 't', 'lf', 'lh', 'rh', 'rf', 'math', 'story', 'mental', 'rnd', 'match', 'relation',
                 'fear',
                 'neut']
    gfc_pth = r"..\dataset\HCP\processed\gfc.npy"

    for cate in cate_list:
        print("========================================================================")
        print("============== Start calculate {} task embedding feature. ===============".format(cate))
        print("========================================================================")
        gc.collect()
        min_l = 15
        run_index = 8
        task_dict, task_len_dict, key_index = dataloader.dataloader(cate)
        dataset = dataloader.dataset(task_dict, task_len_dict, gfc_pth, min_l, run_index, key_index)
        train_dataset, test_dataset, total_dataset, g = dataset.train_dataset, dataset.test_dataset, dataset.total_dataset, dataset.g
        g = g.to(device)


        num_node = dataset.num_node
        window = dataset.min_l
        horizon = int(window / 4)  # Need to modify when model structure change
        num_class = 23
        model = STGCN_pyramid_feature(channels, window, horizon, num_node, g, drop_prob, model_structure, num_class, True).to(device)
        model.load_state_dict(model_config, strict=False)

        test_iter = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

        save_result = None

        model.eval()
        with torch.no_grad():
            for x, y in test_iter:
                gc.collect()
                shape = x.shape[0]
                x = x.squeeze(1).reshape([shape, -1]).cpu().numpy()
                if save_result is None:
                    save_result = x
                else:
                    save_result = np.concatenate((save_result, x), axis=0)
        np.save(os.path.join(pth, cate), save_result)

        print(time.time()-start_time)


def tsne_hidden(n):
    pth1 = r'./result_cv/stpgcn/stpgcn_feature1'
    pth2 = r'./result_cv/stpgcn/stpgcn_feature2'
    pth3 = r'./result_cv/stpgcn/stpgcn_feature3'
    pth = r'./result_cv/stpgcn/stpgcn_feature'
    pth_ori = r'./result_cv/stpgcn/original_feature'
    params = {'legend.fontsize': 18,
              'figure.figsize': (20, 15),
              'axes.labelsize': 18,
              'axes.titlesize': 18,
              'xtick.labelsize': 16,
              'ytick.labelsize': 16,
              'figure.dpi': 300.0,
              'axes.grid': False}

    plt.rcParams.update(params)
    """========================= Data preprocessing ========================="""
    cate_list = ['0bk_body', '0bk_faces', '0bk_places', '0bk_tools', '2bk_body', '2bk_faces', '2bk_places', '2bk_tools',
             'loss', 'win', 't', 'lf', 'lh', 'rh', 'rf', 'math', 'story', 'mental', 'rnd', 'match', 'relation', 'fear',
             'neut']
    x = None
    y = []
    for cate in cate_list:

        if x is None:
            x = np.load(os.path.join(pth_ori, cate+'.npy'))
            if cate == '0bk_body':
                cate = '0bk body'
            elif cate == '0bk_faces':
                cate = '0bk faces'
            elif cate == '0bk_places':
                cate = '0bk places'
            elif cate == '0bk_tools':
                cate = '0bk tools'
            elif cate == '2bk_body':
                cate = '2bk body'
            elif cate == '2bk_faces':
                cate = '2bk faces'
            elif cate == '2bk_places':
                cate = '2bk places'
            elif cate == '2bk_tools':
                cate = '2bk tools'
            elif cate == 't':
                cate = 'Tongue'
            elif cate == 'lf':
                cate = 'Left foot'
            elif cate == 'lh':
                cate = 'Left hand'
            elif cate == 'rf':
                cate = 'Right foot'
            elif cate == 'rh':
                cate = 'Right hand'
            elif cate == 'rnd':
                cate = 'Random'
            elif cate == 'relation':
                cate = 'Relational'
            elif cate == 'fear':
                cate = 'Emotional face'
            elif cate == 'neut':
                cate = 'Shape'

            # if cate in ['0bk_body', '0bk_faces', '0bk_places', '0bk_tools',
            #             '2bk_body', '2bk_faces', '2bk_places', '2bk_tools']:
            #     cate = 'Working memory'
            # elif cate in ['loss', 'win']:
            #     cate = 'Gambling'
            # elif cate in ['t', 'lf', 'lh', 'rh', 'rf']:
            #     cate = 'Motor'
            # elif cate in ['math', 'story']:
            #     cate = 'Language'
            # elif cate in ['mental', 'rnd']:
            #     cate = 'Social'
            # elif cate in ['match', 'relation']:
            #     cate = 'Relational'
            # elif cate in ['fear', 'neut']:
            #     cate = 'Emotion'
            y = [cate] * x.shape[0]

        else:
            x_data = np.load(os.path.join(pth_ori, cate+'.npy'))
            if cate == '0bk_body':
                cate = '0bk body'
            elif cate == '0bk_faces':
                cate = '0bk faces'
            elif cate == '0bk_places':
                cate = '0bk places'
            elif cate == '0bk_tools':
                cate = '0bk tools'
            elif cate == '2bk_body':
                cate = '2bk body'
            elif cate == '2bk_faces':
                cate = '2bk faces'
            elif cate == '2bk_places':
                cate = '2bk places'
            elif cate == '2bk_tools':
                cate = '2bk tools'
            elif cate == 't':
                cate = 'Tongue'
            elif cate == 'lf':
                cate = 'Left foot'
            elif cate == 'lh':
                cate = 'Left hand'
            elif cate == 'rf':
                cate = 'Right foot'
            elif cate == 'rh':
                cate = 'Right hand'
            elif cate == 'rnd':
                cate = 'Random'
            elif cate == 'relation':
                cate = 'Relational'
            elif cate == 'fear':
                cate = 'Emotional face'
            elif cate == 'neut':
                cate = 'Shape'
            elif cate == 'loss':
                cate = 'Loss'
            elif cate == 'win':
                cate = 'Win'
            elif cate == 'math':
                cate = 'Math'
            elif cate == 'story':
                cate = 'Story'
            elif cate == 'mental':
                cate = 'Mental'
            elif cate == 'match':
                cate = 'Match'

            # if cate in ['0bk_body', '0bk_faces', '0bk_places', '0bk_tools',
            #             '2bk_body', '2bk_faces', '2bk_places', '2bk_tools']:
            #     cate = 'Working memory'
            # elif cate in ['loss', 'win']:
            #     cate = 'Gambling'
            # elif cate in ['t', 'lf', 'lh', 'rh', 'rf']:
            #     cate = 'Motor'
            # elif cate in ['math', 'story']:
            #     cate = 'Language'
            # elif cate in ['mental', 'rnd']:
            #     cate = 'Social'
            # elif cate in ['match', 'relation']:
            #     cate = 'Relational'
            # elif cate in ['fear', 'neut']:
            #     cate = 'Emotion'
            y_data = [cate] * x_data.shape[0]
            y.extend(y_data)
            x = np.concatenate((x, x_data), axis=0)
        # y_label += 1
    y = np.array(y)

    """========================= T-SNE ========================="""
    n_components = 2
    tsne = TSNE(random_state=42, n_components=n_components)
    tsne_result = tsne.fit_transform(x)

    # Plot the result of our TSNE with the label color coded
    # A lot of the stuff here is about making the plot look pretty and not TSNE
    tsne_result_df = pd.DataFrame({'tsne_1': tsne_result[:, 0], 'tsne_2': tsne_result[:, 1], 'label': y})
    fig, ax = plt.subplots(1)
    sns.scatterplot(x='tsne_1',
                    y='tsne_2',
                    hue='label',
                    data=tsne_result_df,
                    ax=ax,
                    s=50,
                    palette="gist_rainbow"
                    # palette = "rainbow"
                    )
    lim = (tsne_result.min() - 5, tsne_result.max() + 5)
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_aspect('equal')
    ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0)
    plt.xlabel("1st-Component")
    plt.ylabel("2nd-Component")
    plt.gcf().subplots_adjust(bottom=0.15)
    plt.savefig(os.path.join(r'result_cv/stpgcn/tsne_original_result/tsne_{}.png'.format(n)))
    # plt.show()


def tsne_original():
    pth = r'.\result\original_feature'
    params = {'legend.fontsize': 18,
              'figure.figsize': (8, 5),
              'axes.labelsize': 12,
              'axes.titlesize': 18,
              'xtick.labelsize': 10,
              'ytick.labelsize': 10,
              'figure.dpi': 300.0,
              'axes.grid': False}

    plt.rcParams.update(params)
    """========================= Data preprocessing ========================="""
    cate_list = ['motor', 'language', 'gambling', 'wm', 'emotion', 'social', 'relational']
    x = None
    y = []
    for cate in cate_list:

        if x is None:
            x = np.load(os.path.join(pth, cate+'.npy'))
            y = [cate] * x.shape[0]
        else:
            x_data = np.load(os.path.join(pth, cate+'.npy'))
            y_data = [cate] * x_data.shape[0]
            y.extend(y_data)
            x = np.concatenate((x, x_data), axis=0)
        # y_label += 1
    y = np.array(y)

    """========================= T-SNE ========================="""
    n_components = 2
    tsne = TSNE(n_components, early_exaggeration=5)
    tsne_result = tsne.fit_transform(x)

    # Plot the result of our TSNE with the label color coded
    # A lot of the stuff here is about making the plot look pretty and not TSNE
    tsne_result_df = pd.DataFrame({'tsne_1': tsne_result[:, 0], 'tsne_2': tsne_result[:, 1], 'label': y})
    fig, ax = plt.subplots(1)
    sns.scatterplot(x='tsne_1',
                    y='tsne_2',
                    hue='label',
                    data=tsne_result_df,
                    ax=ax,
                    s=50,
                    palette="Dark2")
    lim = (tsne_result.min() - 5, tsne_result.max() + 5)
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_aspect('equal')
    ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0)
    plt.xlabel("1st-Component")
    plt.ylabel("2nd-Component")
    plt.gcf().subplots_adjust(bottom=0.15)
    plt.savefig(os.path.join(pth, r'result\tsne_original1.png'))
    # plt.show()


if __name__ == '__main__':
    # cal_embed()
    # cal_stpgcn_layer_embed()
    tsne_hidden(1)
    # save_feature()
    # save_original_data()
    # tsne_original()