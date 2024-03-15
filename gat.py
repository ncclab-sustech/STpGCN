# !/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Author              : Ziyuan Ye
@Email               : ziyuanye9801@gmail.com
'''

import torch
import torch.nn as nn
import dgl.function as fn
from dgl.nn import GATConv

import sys


class GAT(nn.Module):
    def __init__(self, g, nlayers, in_dim, nhidden, nclass, heads,
                 activation, feat_drop, attn_drop, negative_slope, residual, T, num_node):
        super(GAT, self).__init__()
        self.g = g
        self.nlayers = nlayers
        self.activation = activation
        self.gat_layers = nn.ModuleList()
        self.num_node = num_node

        # Input layer
        self.gat_layers.append(GATConv(
            in_dim, nhidden, heads[0], feat_drop, attn_drop, negative_slope, False, self.activation
        ))

        # Hidden layer
        for layer in range(nlayers - 1):
            self.gat_layers.append(GATConv(
                nhidden * heads[layer], nhidden, heads[layer + 1], feat_drop,
                attn_drop, negative_slope, residual, self.activation
            ))

        # Output layer
        self.gat_layers.append(GATConv(
            nhidden * heads[-2], 1, heads[-1], feat_drop, attn_drop,
            negative_slope, residual, None
        ))

        self.output = Outputlayer(c=heads[-1], T=T, nnode=self.num_node, nclass=nclass)

    def forward(self, inputs):
        h = inputs.to(torch.float32)
        h = h.transpose(0, 3)
        # [nodes, c_in, ts, batch] --> [nodes, batch, ts, c_in]
        h = h.transpose(1, 3)
        dim_node, dim_batch, dim_ts = h.shape[:3]

        for layer in range(self.nlayers):
            h = self.gat_layers[layer](self.g, h).reshape([dim_node, dim_batch, dim_ts, -1])

        # output = self.gat_layers[-1](self.g, h).mean(0).mean(1).mean(1)
        output = self.gat_layers[-1](self.g, h)
        output = output.transpose(1, 3)

        # [nodes, c_out, ts, batch] --> [batch, c_out, ts, nodes]
        output = output.transpose(0, 3).squeeze(-1)
        return self.output(output)


class Outputlayer(nn.Module):

    def __init__(self, c, T, nnode, nclass):
        super(Outputlayer, self).__init__()
        self.tconv1 = nn.Conv2d(in_channels=c,
                                out_channels=c,
                                kernel_size=(T, 1))
        self.ln1 = nn.LayerNorm([nnode, c])
        # self.tconv2 = nn.Conv2d(in_channels=c,
        #                         out_channels=1,
        #                         kernel_size=(1, 1))
        # self.fc = nn.Linear(90,2)
        self.tconv2 = nn.Conv2d(in_channels=c,
                                out_channels=1,
                                kernel_size=(1, 1))
        self.ln2 = nn.LayerNorm([nnode, 1])
        self.fc = nn.Conv2d(in_channels=1,
                            out_channels=1,
                            kernel_size=(1, nnode - nclass + 1))
        self.T = T

    def forward(self, x):
        # maps multi-steps to one
        # [batch, c_in, ts, nodes] --> [batch, c_out_1, 1, nodes]
        x = x.squeeze(-1)
        x_t1 = self.tconv1(x)
        x_ln1 = self.ln1(x_t1.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

        # [batch, c_out_1, 1, nodes] --> [batch, nodes]
        x_t2 = self.tconv2(x_ln1)
        x_ln2 = self.ln2(x_t2.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

        x_r = self.fc(x_ln2).squeeze(1).squeeze(1)

        return x_r

