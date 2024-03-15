# !/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Author              : Ziyuan Ye
@Email               : ziyuanye9801@gmail.com
'''
import nilearn.plotting as plotting
import hcp_utils as hcp
import numpy as np
import matplotlib.pyplot as plt
import os
from nilearn import datasets

# cate_list = ['Emotion', 'Gambling', 'Language', 'Motor', 'Relational', 'Social', 'WM']
# cate_list = ['Relational']
# pth_list = [r'.\result_cv\WM_task\diff_stimuli\gcn\saliency_result\yeo17\mix',
#             r'.\result_cv\WM_task\diff_stimuli\gat\saliency_result\yeo17\mix',
#             r'.\result_cv\WM_task\diff_stimuli\stgcn\saliency_result\yeo17\mix',
#             r'.\result_cv\WM_task\diff_stimuli\stpgcn\saliency_result\yeo17\mix']
# cate_list = ['body', 'faces', 'places', 'tools']
pth = r'result_cv/TNNLS/NeurocircuitX/mlp_mixer'
cate_list = ['0bk_body', '0bk_faces', '0bk_places', '0bk_tools', '2bk_body', '2bk_faces', '2bk_places', '2bk_tools',
                 'loss', 'win', 't', 'lf', 'lh', 'rh', 'rf', 'math', 'story', 'mental', 'rnd', 'match', 'relation',
                 'fear', 'neut']
# pth = r'result_cv/MMP/mlp_mixer/saliency_result/yeo17/mix'
pth_list = [r'./result_cv/MMP/mlp_mixer/saliency_result/yeo17/mix']

fsaverage = datasets.fetch_surf_fsaverage()

hemi_list = ['left', 'right']
view_list = ['lateral', 'medial']

for pth in pth_list:
    for view_ in view_list:
        for hemi_ in hemi_list:
            for cate in cate_list:
                f_weight = np.loadtxt(os.path.join(pth,'{}.csv'.format(cate)))

                if hemi_ == 'left':
                    temp = plotting.plot_surf(hcp.mesh.inflated_left,
                                              hcp.cortex_data(hcp.unparcellate(f_weight, hcp.mmp).reshape(-1))[:32492],
                                              bg_map=hcp.mesh.sulc_left, hemi=hemi_, view=view_, cmap=plt.get_cmap('RdBu_r'),
                                              symmetric_cmap=False)
                elif hemi_ == 'right':
                    temp = plotting.plot_surf(hcp.mesh.inflated_right,
                                              hcp.cortex_data(hcp.unparcellate(f_weight, hcp.mmp).reshape(-1))[32492:],
                                              bg_map=hcp.mesh.sulc_right, hemi=hemi_, view=view_, cmap=plt.get_cmap('RdBu_r'),
                                              symmetric_cmap=False)
                temp.savefig(os.path.join(pth, '{}_{}_{}.png'.format(cate, hemi_, view_)), dpi=300)

print('finish')

    # ========================== Save as HTML ==========================
    # ========================== full brain ==========================
    # temp = plotting.view_surf(hcp.mesh.inflated,
    #                           hcp.cortex_data(hcp.unparcellate(f_weight, hcp.mmp).reshape(-1)),
    #                           bg_map=hcp.mesh.sulc, cmap=plt.get_cmap('RdBu_r'), symmetric_cmap=False)  # RdBu_r
    # data = hcp.cortex_data(hcp.unparcellate(f_weight, hcp.mmp).reshape(-1))[:32492]
    # print(data.shape)

    # ========================== half brain ==========================
    # temp = plotting.view_surf(hcp.mesh.inflated_right, data,
    #                            bg_map=hcp.mesh.sulc_right, cmap=plt.get_cmap('BuGn'), symmetric_cmap=False)
    # # temp.open_in_browser()
    # temp.save_as_html(os.path.join(pth, '{}_right.html'.format(cate)))


