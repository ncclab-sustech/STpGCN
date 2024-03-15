# !/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Author              : Ziyuan Ye
@Email               : ziyuanye9801@gmail.com
'''
import numpy as np

import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import DataLoader

from mlp_mixer_pytorch import MLPMixer
from braingnn import BrainGNN

import tqdm

import dataloader

#  Model Selection
from stpgcn_variants import STGCN, STpGCN, STpGCN_ab_top, STpGCN_ab_mid, STpGCN_ab_bottom_up
from gcn import GCN
from gat import GAT
from gin import GIN

from torch_geometric.utils.convert import to_networkx, from_networkx
from torch_geometric.data import DataLoader
from torch_geometric.data import Data
import dgl

import gc
import time
import sys
import warnings
warnings.filterwarnings("ignore")
warnings.simplefilter(action='ignore', category=FutureWarning)

loss_alpha = 0.0005


def evaluate_model(model, data_iter, nnode: int, judge_save: bool):

    model.eval()
    l_sum, n = 0.0, 0
    total_pred_y = []
    total_target_y = []

    with torch.no_grad():
        for x, y in data_iter:            
            x = x[:, :, :, :nnode].to(torch.float32) # for mlp_mixer, gin
            output = F.log_softmax(model(x.cuda()), dim=-1)
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

        if judge_save:
            return l_sum / n, sum(total_target_y == total_pred_y) / n, total_target_y, total_pred_y
        else:
            return l_sum / n, sum(total_target_y == total_pred_y) / n


def main(run_index: int, min_l: int, model_name: str):
    """================ Hyper-parameters Setting ================="""
    # TODO ============================ check =====================
    best_test_acc = 0
    # gfc_pth = r".\fc\mmp_gfc_360.npy"
    gfc_pth = r".\fc\mmp_gfc_308.npy"
    # gfc_pth = r".\fc\aal_gfc_116.npy"
    # gfc_pth = r".\fc\aal_gfc_90.npy"
    # =============== Model params ===================
    model_structure = 'TSTNTSTN'
    channels = [1, 32, 16, 32, 32, 16, 32]  # Need to modify when model structure change

    # =============== Training params ===============
    lr = 1e-3
    batch_size = 25
    epochs = 32

    drop_prob = 0.5
    k_knn = 5

    # =============== Device params ===============
    DisableGPU = False
    device = torch.device("cuda") if torch.cuda.is_available() and not DisableGPU else torch.device("cpu")

    # TODO ============================ check =====================
    cate = 'wm'
    # cate = 'all'
    print("Current cate is {}".format(cate))
    task_dict, task_len_dict, key_index = dataloader.dataloader(cate)
    dataset = dataloader.dataset(task_dict, task_len_dict, gfc_pth, min_l, run_index, key_index)
    train_dataset, test_dataset, val_dataset, g,  = dataset.train_dataset, \
                                                  dataset.test_dataset, \
                                                  dataset.val_dataset, \
                                                  dataset.g
    g = g.to(device)

    num_node = dataset.num_node
    window = dataset.min_l
    horizon = int(window / 4)  # Need to modify when model structure change
    num_class = 4

    val_iter = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
    test_iter = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
    train_iter = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    """=================== STGCN Training ====================="""

    # model definition
    t = time.localtime()
    # ==================================== All tasks ==========================================
    if model_name == 'stgcn':
        model = STGCN(channels, window, horizon, num_node, g, drop_prob, model_structure, num_class).to(device)
        save_pth = r'.\checkpoints_cv\AAL\stgcn\run_{}\stgcn-length={}.pt'.format(str(run_index), str(window))
        train_name = r'.\result_cv\AAL\stgcn\run_{}\length={}-train.txt'.format(str(run_index), str(window))
        test_name = r'.\result_cv\AAL\stgcn\run_{}\length={}-test.txt'.format(str(run_index), str(window))
        time_name = r'.\result_cv\AAL\stgcn\run_{}\length={}-time.txt'.format(str(run_index), str(window))
        target_name = r'.\result_cv\AAL\stgcn\run_{}\length={}-target.npy'.format(str(run_index), str(window))
        pred_name = r'.\result_cv\AAL\stgcn\run_{}\length={}-pred.npy'.format(str(run_index), str(window))
    
    elif model_name == 'gcn':
        model = GCN(c_in=1, c_hid=16, c_out=1, g=g, ts=window, nclass=num_class, num_node=num_node).to(device)
        save_pth = r'.\checkpoints_cv\AAL\gcn\run_{}\gcn-length={}.pt'.format(str(run_index), str(window))
        train_name = r'.\result_cv\AAL\gcn\run_{}\length={}-train.txt'.format(str(run_index), str(window))
        test_name = r'.\result_cv\AAL\gcn\run_{}\length={}-test.txt'.format(str(run_index), str(window))
        time_name = r'.\result_cv\AAL\gcn\run_{}\length={}-time.txt'.format(str(run_index), str(window))
        target_name = r'.\result_cv\AAL\gcn\run_{}\length={}-target.npy'.format(str(run_index), str(window))
        pred_name = r'.\result_cv\AAL\gcn\run_{}\length={}-pred.npy'.format(str(run_index), str(window))
    
    elif model_name == 'stpgcn':
        model = STpGCN(channels, window, horizon, num_node, g, drop_prob, model_structure, num_class, True).to(device)
        save_pth = r'.\checkpoints_cv\MMP\robust\stpgcn\10\run_{}\stpgcn-length={}.pt'.format(str(run_index), str(window))
        train_name = r'.\result_cv\MMP\robust\stpgcn\10\run_{}\length={}-train.txt'.format(str(run_index), str(window))
        test_name = r'.\result_cv\MMP\robust\stpgcn\10\run_{}\length={}-test.txt'.format(str(run_index), str(window))
        time_name = r'.\result_cv\MMP\robust\stpgcn\10\run_{}\length={}-time.txt'.format(str(run_index), str(window))
        target_name = r'.\result_cv\MMP\robust\stpgcn\10\run_{}\length={}-target.npy'.format(str(run_index), str(window))
        pred_name = r'.\result_cv\MMP\robust\stpgcn\10\run_{}\length={}-pred.npy'.format(str(run_index), str(window))
    
    elif model_name == 'stpgcn_ab_top':
        model = STpGCN_ab_top(channels, window, horizon, num_node,
                              g, drop_prob, model_structure, num_class, True).to(device)
        save_pth = r'.\checkpoints_cv\AAL\stpgcn_ab_top\run_{}\stpgcn-length={}.pt'.format(str(run_index), str(window))
        train_name = r'.\result_cv\AAL\stpgcn_ab_top\run_{}\length={}-train.txt'.format(str(run_index), str(window))
        test_name = r'.\result_cv\AAL\stpgcn_ab_top\run_{}\length={}-test.txt'.format(str(run_index), str(window))
        time_name = r'.\result_cv\AAL\stpgcn_ab_top\run_{}\length={}-time.txt'.format(str(run_index), str(window))
        target_name = r'.\result_cv\AAL\stpgcn_ab_top\run_{}\length={}-target.npy'.format(str(run_index), str(window))
        pred_name = r'.\result_cv\AAL\stpgcn_ab_top\run_{}\length={}-pred.npy'.format(str(run_index), str(window))
    
    elif model_name == 'stpgcn_ab_mid':
        model = STpGCN_ab_mid(channels, window, horizon, num_node,
                              g, drop_prob, model_structure, num_class, True).to(device)
        save_pth = r'.\checkpoints_cv\AAL\stpgcn_ab_mid\run_{}\stpgcn-length={}.pt'.format(str(run_index), str(window))
        train_name = r'.\result_cv\AAL\stpgcn_ab_mid\run_{}\length={}-train.txt'.format(str(run_index), str(window))
        test_name = r'.\result_cv\AAL\stpgcn_ab_mid\run_{}\length={}-test.txt'.format(str(run_index), str(window))
        time_name = r'.\result_cv\AAL\stpgcn_ab_mid\run_{}\length={}-time.txt'.format(str(run_index), str(window))
        target_name = r'.\result_cv\AAL\stpgcn_ab_mid\run_{}\length={}-target.npy'.format(str(run_index), str(window))
        pred_name = r'.\result_cv\AAL\stpgcn_ab_mid\run_{}\length={}-pred.npy'.format(str(run_index), str(window))
    
    elif model_name == 'stpgcn_ab_bottom_up':
        model = STpGCN_ab_bottom_up(channels, window, horizon, num_node,
                              g, drop_prob, model_structure, num_class, True).to(device)
        save_pth = r'.\checkpoints_cv\MMP\stpgcn_ab_bottom_up\run_{}\stpgcn-length={}.pt'.format(str(run_index), str(window))
        train_name = r'.\result_cv\MMP\stpgcn_ab_bottom_up\run_{}\length={}-train.txt'.format(str(run_index), str(window))
        test_name = r'.\result_cv\MMP\stpgcn_ab_bottom_up\run_{}\length={}-test.txt'.format(str(run_index), str(window))
        time_name = r'.\result_cv\MMP\stpgcn_ab_bottom_up\run_{}\length={}-time.txt'.format(str(run_index), str(window))
        target_name = r'.\result_cv\MMP\stpgcn_ab_bottom_up\run_{}\length={}-target.npy'.format(str(run_index), str(window))
        pred_name = r'.\result_cv\MMP\stpgcn_ab_bottom_up\run_{}\length={}-pred.npy'.format(str(run_index), str(window))
    
    elif model_name == 'gat':
        model = GAT(g, nlayers=2, in_dim=1, nhidden=32, nclass=num_class, heads=[4, 4],
                    activation=nn.ReLU(inplace=True), feat_drop=0.3, attn_drop=0.3,
                    negative_slope=0.2, residual=False, T=window, num_node=num_node).to(device)
        save_pth = r'.\checkpoints_cv\AAL\gat\run_{}\gat-length={}.pt'.format(str(run_index),
                                                                                      str(window))
        train_name = r'.\result_cv\AAL\gat\run_{}\length={}-train.txt'.format(str(run_index), str(window))
        test_name = r'.\result_cv\AAL\gat\run_{}\length={}-test.txt'.format(str(run_index), str(window))
        time_name = r'.\result_cv\AAL\gat\run_{}\length={}-time.txt'.format(str(run_index), str(window))
        target_name = r'.\result_cv\AAL\gat\run_{}\length={}-target.npy'.format(str(run_index),
                                                                                      str(window))
        pred_name = r'.\result_cv\AAL\gat\run_{}\length={}-pred.npy'.format(str(run_index), str(window))
    
    elif model_name == 'mlp_mixer':
        model = MLPMixer(image_size=(15, num_node),
                        channels=1,
                        patch_size=3,
                        dim=512,
                        depth=12,
                        num_classes=num_class,
                         dropout= drop_prob
                        ).to(device)
        save_pth = r'.\checkpoints_cv\AAL\mlp_mixer\run_{}\mlp_mixer-length={}.pt'.format(str(run_index),
                                                                          str(window))
        train_name = r'.\result_cv\AAL\mlp_mixer\run_{}\length={}-train.txt'.format(str(run_index), str(window))
        test_name = r'.\result_cv\AAL\mlp_mixer\run_{}\length={}-test.txt'.format(str(run_index), str(window))
        time_name = r'.\result_cv\AAL\mlp_mixer\run_{}\length={}-time.txt'.format(str(run_index), str(window))
        target_name = r'.\result_cv\AAL\mlp_mixer\run_{}\length={}-target.npy'.format(str(run_index),
                                                                            str(window))
        pred_name = r'.\result_cv\AAL\mlp_mixer\run_{}\length={}-pred.npy'.format(str(run_index), str(window))
    
    elif model_name == 'gin':
        model = GIN(g, num_layers=5, num_mlp_layers=2, input_dim=1, hidden_dim=64,
                 output_dim=32, final_dropout=0.3, learn_eps=0, graph_pooling_type='sum',
                 neighbor_pooling_type='sum').to(device)
        save_pth = r'.\checkpoints_cv\AAL\gin\run_{}\gin-length={}.pt'.format(str(run_index),
                                                                          str(window))
        train_name = r'.\result_cv\AAL\gin\run_{}\length={}-train.txt'.format(str(run_index), str(window))
        test_name = r'.\result_cv\AAL\gin\run_{}\length={}-test.txt'.format(str(run_index), str(window))
        time_name = r'.\result_cv\AAL\gin\run_{}\length={}-time.txt'.format(str(run_index), str(window))
        target_name = r'.\result_cv\AAL\gin\run_{}\length={}-target.npy'.format(str(run_index),
                                                                            str(window))
        pred_name = r'.\result_cv\AAL\gin\run_{}\length={}-pred.npy'.format(str(run_index), str(window))

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                           'min',
                                                           0.5,
                                                           patience=5,
                                                           threshold=0.5)
    print("Details: lr = {}, batch size = {}, epochs = {}, horizon = {}, drop_prob = {}, k_knn = {}".format(lr,
                                                                                                   batch_size,
                                                                                                   epochs,
                                                                                                   horizon,
                                                                                                   drop_prob,
                                                                                                   k_knn))

    train_save_list = []
    test_save_list = []
    start_time = time.time()
    training_time = []
    for epoch in range(1, epochs + 1):
        model.train()
        l_sum, n = 0.0, 0  # 'l_sum' is epoch sum loss, 'n' is epoch instance number
        total_pred_y = []
        total_target_y = []
        for x, y in tqdm.tqdm(train_iter):
            # TODO ============================== Check =================================
            if model_name == 'braingnn':
                l = []
                for index in range(x.shape[0]):
                    l.append(Data(x[index, 0, :, :].squeeze().squeeze().permute(1, 0).long(),
                                  torch.tensor((g.edges()[0].cpu().numpy(), g.edges()[1].cpu().numpy()),
                                               dtype=torch.long).contiguous(), y[index], g.adj().long()).cuda())

                loader = DataLoader(l, batch_size=x.shape[0])
                for data in loader:
                    output = F.log_softmax(model(data.x, data.edge_index, data.batch, data.edge_attr, data.pos), dim=-1)
            else:
                x = x[:, :, :, :num_node].to(torch.float32)
                output = F.log_softmax(model(x.cuda()), dim=-1)
            l = F.nll_loss(output, y.argmax(dim=-1))
            l2_reg = torch.tensor(0.).cuda()
            for name, param in model.named_parameters():
                if 'bias' not in name:
                    l2_reg += loss_alpha * torch.norm(param)
            l += l2_reg

            optimizer.zero_grad()
            l.backward()
            optimizer.step()
            l_sum += l.item() * y.shape[0]
            n += y.shape[0]
            pred_y = torch.max(output.cpu(), 1)[1].numpy().squeeze().tolist()
            target_y = torch.max(y.cpu().data, 1)[1].numpy().tolist()

            total_pred_y += pred_y
            total_target_y += target_y

        training_time.append(round(time.time()-start_time, 1))

        scheduler.step(l_sum/n)

        # Epoch validation
        if epoch == epochs:
            val_loss, val_acc, save_target_y, save_pred_y = evaluate_model(model, test_iter, num_node, True)
            np.save(target_name, save_target_y)
            np.save(pred_name, save_pred_y)
        else:
            val_loss, val_acc = evaluate_model(model, test_iter, num_node, False)

        # GPU mem usage
        gpu_mem_alloc = torch.cuda.max_memory_allocated() / 1000000 if torch.cuda.is_available() else 0

        # Save model when test acc is greater than the best
        if best_test_acc < val_acc:
            best_test_acc = val_acc
            save_flag = True
        if save_flag:
            torch.save(model.state_dict(), save_pth)
            save_flag = False

        # If set batch size >= 1
        total_pred_y = np.array(total_pred_y)
        total_target_y = np.array(total_target_y)
        print('Epoch {:03d}|'
              ' lr {:.6f} |'
              ' Train Loss {:.5f} |'
              ' Train Acc {:.5f} |'
              ' Val Loss {:.5f} |'
              ' Val Acc {:.5f} |'
              ' GPU {:.1f} MiB'.format(
            epoch, optimizer.param_groups[0]['lr'], l_sum / n, sum(total_target_y == total_pred_y) / n,
            val_loss, val_acc, gpu_mem_alloc))
        train_save_list.append(round(sum(total_target_y == total_pred_y) / n, 3))
        test_save_list.append(round(val_acc, 3))

    with open(train_name, 'w') as f:
        f.write(str(train_save_list))
        f.close()

    with open(test_name, 'w') as f:
        f.write(str(test_save_list))
        f.close()

    with open(time_name, 'w') as f:
        f.write(str(training_time))
        f.close()


if __name__ == '__main__':
    nframe = list(range(4,16,3)) + [15]
    for run_index in range(10):
        print("========================================================================")
        print("================== Start # {} run of Training STpGCN. =====================".format(run_index + 1))
        print("========================================================================")
        for frame in nframe:
            print("Start # {} frame of Training STpGCN.".format(frame))
            main(run_index + 1, frame, 'stpgcn')
    print('\nAll 10 runs finished.\n')