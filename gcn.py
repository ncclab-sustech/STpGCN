# !/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Author              : Ziyuan Ye
@Email               : ziyuanye9801@gmail.com
'''


import dgl
import dgl.function as fn

import torch.nn as nn
import torch.nn.functional as F


class GCNLayer(nn.Module):
    def __init__(self, in_feats, out_feats):
        super(GCNLayer, self).__init__()
        self.linear = nn.Linear(in_feats, out_feats)

    def forward(self, g: dgl.DGLGraph, feature):

        with g.local_scope():
            gcn_msg = fn.copy_u(u='h', out='m')
            gcn_reduce = fn.sum(msg='m', out='h')

            g.ndata['h'] = feature
            g.update_all(gcn_msg, gcn_reduce)

            h = g.ndata['h']
            return self.linear(h)


class GCN(nn.Module):
    def __init__(self, c_in, c_hid, c_out, g, ts, nclass, num_node):
        super(GCN, self).__init__()

        self.ts = ts
        self.layer1 = GCNLayer(c_in, c_hid)
        self.layer2 = GCNLayer(c_hid, c_out)
        self.nnodes = num_node
        self.outlayer = OutputLayer(ts, self.nnodes, nclass)

        self.g = g

    def forward(self, features):
        # [batch, c_in, ts, nodes] --> [nodes, c_in, ts, batch]
        x = features.transpose(0, 3)

        # [nodes, c_in, ts, batch] --> [nodes, batch, ts, c_in]
        x = x.transpose(1, 3)

        # output: [nodes, batch, ts, c_out]
        output = F.relu(self.layer1(self.g, x.float()))
        output = self.layer2(self.g, output.float())

        # [nodes, batch, ts, c_out] --> [nodes, c_out, ts, batch]
        output = output.transpose(1, 3)

        # [nodes, c_out, ts, batch] --> [batch, c_out, ts, nodes]
        output = output.transpose(0, 3)

        return self.outlayer(output)


class OutputLayer(nn.Module):
    def __init__(self, ts, nnodes, nclass):
        super(OutputLayer, self).__init__()
        self.T = ts
        self.nnodes = nnodes
        self.nclass = nclass
        self.tconv1 = nn.Conv2d(in_channels=1,
                                out_channels=1,
                                kernel_size=(ts, 1))
        self.ln1 = nn.LayerNorm([nnodes, 1])
        self.tconv2 = nn.Conv2d(in_channels=1,
                                out_channels=1,
                                kernel_size=(1, 1))
        self.ln2 = nn.LayerNorm([nnodes, 1])
        self.fc = nn.Conv2d(in_channels=1,
                            out_channels=1,
                            kernel_size=(1, nnodes - nclass + 1))

    def forward(self, x):
        # maps multi-steps to one
        # [batch, c_in, ts, nodes] --> [batch, c_out_1, 1, nodes]

        x_t1 = self.tconv1(x)
        x_ln1 = self.ln1(x_t1.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

        # [batch, c_out_1, 1, nodes] --> [batch, nodes]
        x_t2 = self.tconv2(x_ln1)
        x_ln2 = self.ln2(x_t2.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

        x_r = self.fc(x_ln2).squeeze(1).squeeze(1)

        return x_r

# class GCNLayer(nn.Module):
#     def __init__(self, in_feats, out_feats):
#         super(GCNLayer, self).__init__()
#         self.linear = nn.Linear(in_feats, out_feats)
#
#     def forward(self, g: dgl.DGLGraph, feature):
#
#         with g.local_scope():
#             gcn_msg = fn.copy_u(u='h', out='m')
#             gcn_reduce = fn.sum(msg='m', out='h')
#
#             g.ndata['h'] = feature
#             g.update_all(gcn_msg, gcn_reduce)
#
#             h = g.ndata['h']
#             return self.linear(h)
#
#
# class GCN(nn.Module):
#     def __init__(self, c_in, c_hid, c_out, g, ts, nclass):
#         super(GCN, self).__init__()
#
#         self.ts = ts
#
#         self.layer1 = GCNLayer(c_in, c_hid)
#         self.layer2 = GCNLayer(c_hid, c_out)
#
#         self.sumpool = SumPooling()
#
#         self.linear1 = nn.Linear(self.ts, int(self.ts/2))
#         self.linear2 = nn.Linear(int(self.ts/2), nclass)
#
#         self.g = g
#
#     def forward(self, features):
#         # [batch, c_in, ts, nodes] --> [nodes, c_in, ts, batch]
#         x = features.transpose(0, 3)
#
#         # [nodes, c_in, ts, batch] --> [nodes, batch, ts, c_in]
#         x = x.transpose(1, 3)
#
#         # output: [nodes, batch, ts, c_out]
#         output = F.relu(self.layer1(self.g, x.float()))
#         output = self.layer2(self.g, output.float())
#
#         # [node=1, batch, ts, c_out=1]  Here, the node indicate the graph
#         output = self.sumpool(self.g, output)
#
#         # [nodes, batch, ts, c_out] --> [nodes, c_out, ts, batch]
#         output = output.transpose(1, 3)
#
#         # [nodes, c_out, ts, batch] --> [batch, c_out, ts, nodes]
#         output = output.transpose(0, 3)
#
#         # maps multi-channels to two
#         x = output.squeeze(1).squeeze(2)
#
#         x = self.linear1(x)
#         x = self.linear2(x)
#         return x

