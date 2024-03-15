# !/usr/bin/env python
# -*- coding: UTF-8 -*-


import math
import sys

import dgl

import torch.nn as nn
import torch

from layers import TemporalConvLayer_Residual, SpatialConvLayer, OutputLayer, OutputLayer_hidden_feature


class STGCN(nn.Module):
    """
    Inputs:
        c: channels,
        T: window length
        n: num_nodes
        kt: kernel size of temporal conv
        g: fixed DGLGraph
        p: dropout after each 'sandwich', i.e. 'TSTN'
        control_str: model structure controller, e.g. 'TSTNTSTN', where T: Temporal Layer, S: Spatio Layer, N: Norm Layer
        x: input feature matrix with the shape [batch, 1, T, n]

    Return:
        y: output with the shape [batch, n]
    """

    def __init__(self, c: list, T: int, kt: int, n: int, g: dgl.DGLGraph, p: float, control_str: str, num_class:int):
        super(STGCN, self).__init__()
        self.control_str = control_str
        self.num_layers = len(control_str)
        self.num_node = n
        self.layers = nn.ModuleList()
        self.dropout = nn.Dropout(p)
        self.nclass = num_class

        # Temporal conv kernel size set to window size / num('T')
        self.kt = kt

        # c_index controls the change of channels
        c_index = 0
        num_temporal_layers = 0

        for i in range(self.num_layers):

            layer_i = control_str[i]

            # Temporal layer
            if layer_i == 'T':
                self.layers.append(TemporalConvLayer_Residual(c_in=c[c_index],
                                                              c_out=c[c_index + 1],
                                                              kernel=self.kt))
                c_index += 1
                num_temporal_layers += 1

            # Spatial layer
            elif layer_i == 'S':
                self.layers.append(SpatialConvLayer(c[c_index], c[c_index + 1], g))
                c_index += 1

            # Norm layer
            elif layer_i == 'N':
                self.layers.append(nn.LayerNorm([n, c[c_index]]))

        # c[c_index] is the last element in 'c'
        # T - (self.kt - 1) * num_temporal_layers returns the timesteps after previous
        # temporal layer transformations cuz dialiation = 1

        self.output = OutputLayer(c[c_index], T - (self.kt - 1) * num_temporal_layers, self.num_node, self.nclass)

        for layer in self.layers:
            layer.cuda()

    def forward(self, x):
        for i in range(self.num_layers):
            layer_i = self.control_str[i]
            if layer_i == 'N':
                x = self.dropout(self.layers[i](x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2))
            else:
                x = self.layers[i](x)

        return self.output(x)


class STGCN_hidden_feature(nn.Module):
    """
    Inputs:
        c: channels,
        T: window length
        n: num_nodes
        kt: kernel size of temporal conv
        g: fixed DGLGraph
        p: dropout after each 'sandwich', i.e. 'TSTN'
        control_str: model structure controller, e.g. 'TSTNTSTN', where T: Temporal Layer, S: Spatio Layer, N: Norm Layer
        x: input feature matrix with the shape [batch, 1, T, n]

    Return:
        y: output with the shape [batch, n]
    """

    def __init__(self, c: list, T: int, kt: int, n: int, g: dgl.DGLGraph, p: float, control_str: str, num_class:int):
        super(STGCN_hidden_feature, self).__init__()
        self.control_str = control_str
        self.num_layers = len(control_str)
        self.num_node = n
        self.layers = nn.ModuleList()
        self.dropout = nn.Dropout(p)
        self.nclass = num_class

        # Temporal conv kernel size set to window size / num('T')
        self.kt = kt

        # c_index controls the change of channels
        c_index = 0
        num_temporal_layers = 0

        for i in range(self.num_layers):

            layer_i = control_str[i]

            # Temporal layer
            if layer_i == 'T':
                self.layers.append(TemporalConvLayer_Residual(c_in=c[c_index],
                                                              c_out=c[c_index + 1],
                                                              kernel=self.kt))
                c_index += 1
                num_temporal_layers += 1

            # Spatial layer
            elif layer_i == 'S':
                self.layers.append(SpatialConvLayer(c[c_index], c[c_index + 1], g))
                c_index += 1

            # Norm layer
            elif layer_i == 'N':
                self.layers.append(nn.LayerNorm([n, c[c_index]]))

        self.output = OutputLayer_hidden_feature(c[c_index], T - (self.kt - 1) * num_temporal_layers, self.num_node, self.nclass)

        for layer in self.layers:
            layer.cuda()

    def forward(self, x):
        for i in range(self.num_layers):
            layer_i = self.control_str[i]
            if layer_i == 'N':
                x = self.dropout(self.layers[i](x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2))
            else:
                x = self.layers[i](x)

        return x