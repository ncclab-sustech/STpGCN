# !/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Author              : Ziyuan Ye
@Email               : ziyuanye9801@gmail.com
'''


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


class STpGCN(nn.Module):
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

    def __init__(self, c: list, T: int, kt: int, n: int, g: dgl.DGLGraph, p: float, control_str: str, num_class: int,
                 pyramid: bool):
        super(STpGCN, self).__init__()
        self.control_str = control_str
        self.num_layers = len(control_str)
        self.num_node = n
        self.layers = nn.ModuleList()
        self.dropout = nn.Dropout(p)
        self.nclass = num_class
        self.pyramid = pyramid
        self.catconv = nn.Conv2d(32, 32, kernel_size=(1, 2), stride=(1, 2))
        self.finalconv = nn.Conv2d(32, 32, kernel_size=(1, 3), stride=(1, 3))

        # Temporal conv kernel size set to window size / num('T')
        self.kt = kt

        # c_index controls the change of channels
        c_index = 0
        c_index_second = 0
        c_index_third = 0
        num_temporal_layers = 0
        if not pyramid:
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
        else:
            self.first_layers = nn.ModuleList()
            self.second_layers = nn.ModuleList()
            self.third_layers = nn.ModuleList()
            for i in range(self.num_layers):
                layer_i = control_str[i]
                # Temporal layer
                if layer_i == 'T':
                    self.first_layers.append(TemporalConvLayer_Residual(c_in=c[c_index],
                                                                        c_out=c[c_index + 1],
                                                                        kernel=self.kt))
                    c_index += 1
                    num_temporal_layers += 1
                # Spatial layer
                elif layer_i == 'S':
                    self.first_layers.append(SpatialConvLayer(c[c_index], c[c_index + 1], g))
                    c_index += 1
                # Norm layer
                elif layer_i == 'N' and i == self.num_layers - 1:
                    self.first_layers.append(nn.LayerNorm([n, c[c_index]]))
                    self.second_layers.append(nn.LayerNorm([n, c[c_index_second]]))
                    self.third_layers.append(TemporalConvLayer_Residual(c_in=c[c_index_third],
                                                                        c_out=c[c_index_third + 1],
                                                                        kernel=(self.kt-1)*4+1))
                    c_index_third += 1
                    self.third_layers.append(SpatialConvLayer(c[c_index_third], c[c_index_third], g))
                    self.third_layers.append(nn.LayerNorm([n, c[c_index]]))
                elif layer_i == 'N':
                    self.first_layers.append(nn.LayerNorm([n, c[c_index]]))
                    self.second_layers.append(TemporalConvLayer_Residual(c_in=c[c_index_second],
                                                                         c_out=c[c_index_second + 1],
                                                                         kernel=(self.kt-1)*2+1))
                    c_index_second += 1
                    self.second_layers.append(SpatialConvLayer(c[c_index_second], c[c_index_second + 1], g))
                    c_index_second += 1
                    self.second_layers.append(TemporalConvLayer_Residual(c_in=c[c_index_second],
                                                                         c_out=c[c_index_second + 1],
                                                                         kernel=(self.kt-1)*2+1))
                    c_index_second += 1

            self.output = OutputLayer(c[c_index], T - (self.kt - 1) * num_temporal_layers, self.num_node, self.nclass)

            for layer in self.first_layers:
                layer.cuda()
            for layer in self.second_layers:
                layer.cuda()
            for layer in self.third_layers:
                layer.cuda()

    def forward(self, x):
        if not self.pyramid:
            for i in range(self.num_layers):
                layer_i = self.control_str[i]
                if layer_i == 'N':
                    x = self.dropout(self.layers[i](x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2))
                else:
                    x = self.layers[i](x)
            return self.output(x)
        else:
            first_index = 0
            second_index = 0
            third_index = 0
            x11 = self.first_layers[first_index](x)
            x21 = self.second_layers[second_index](x)
            x31 = self.third_layers[third_index](x)
            first_index, second_index, third_index = first_index + 1, second_index + 1, third_index + 1
            x12 = self.first_layers[first_index](x11)
            first_index = first_index + 1
            x13 = self.first_layers[first_index](x12)
            first_index = first_index + 1
            x14 = self.first_layers[first_index](x13.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
            first_index = first_index + 1
            x22 = self.catconv(torch.cat([x21, x14], 3).type(torch.FloatTensor).cuda())
            x23 = self.second_layers[second_index](x22)
            second_index += 1
            x24 = self.second_layers[second_index](x23)
            second_index += 1
            x15 = self.first_layers[first_index](x14)
            first_index = first_index + 1
            x16 = self.first_layers[first_index](x15)
            first_index = first_index + 1
            x17 = self.first_layers[first_index](x16)
            first_index = first_index + 1
            x18 = self.first_layers[first_index](x17.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
            x25 = self.catconv(torch.cat([x24, x18], 3).type(torch.FloatTensor).cuda())
            x26 = self.second_layers[second_index](x25.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
            x32 = self.catconv(torch.cat([x26, x31], 3).type(torch.FloatTensor).cuda())
            x33 = self.third_layers[third_index](x32)
            third_index += 1
            x34 = self.third_layers[third_index](x33.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
            x_final = self.finalconv(torch.cat([x34, x26, x18], 3))
            return self.output(x_final)


class STpGCN_ab_bottom_up(nn.Module):
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

    def __init__(self, c: list, T: int, kt: int, n: int, g: dgl.DGLGraph, p: float, control_str: str, num_class: int,
                 pyramid: bool):
        super(STpGCN_ab_bottom_up, self).__init__()
        self.control_str = control_str
        self.num_layers = len(control_str)
        self.num_node = n
        self.layers = nn.ModuleList()
        self.dropout = nn.Dropout(p)
        self.nclass = num_class
        self.pyramid = pyramid
        self.catconv = nn.Conv2d(32, 32, kernel_size=(1, 2), stride=(1, 2))
        self.finalconv = nn.Conv2d(32, 32, kernel_size=(1, 3), stride=(1, 3))

        # Temporal conv kernel size set to window size / num('T')
        self.kt = kt

        # c_index controls the change of channels
        c_index = 0
        c_index_second = 0
        c_index_third = 0
        num_temporal_layers = 0
        if not pyramid:
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
        else:
            self.first_layers = nn.ModuleList()
            self.second_layers = nn.ModuleList()
            self.third_layers = nn.ModuleList()
            for i in range(self.num_layers):
                layer_i = control_str[i]
                # Temporal layer
                if layer_i == 'T':
                    self.first_layers.append(TemporalConvLayer_Residual(c_in=c[c_index],
                                                                        c_out=c[c_index + 1],
                                                                        kernel=self.kt))
                    c_index += 1
                    num_temporal_layers += 1
                # Spatial layer
                elif layer_i == 'S':
                    self.first_layers.append(SpatialConvLayer(c[c_index], c[c_index + 1], g))
                    c_index += 1
                # Norm layer
                elif layer_i == 'N' and i == self.num_layers - 1:
                    self.first_layers.append(nn.LayerNorm([n, c[c_index]]))
                    self.second_layers.append(nn.LayerNorm([n, c[c_index_second]]))
                    self.third_layers.append(TemporalConvLayer_Residual(c_in=c[c_index_third],
                                                                        c_out=c[c_index_third + 1],
                                                                        kernel=(self.kt-1)*4+1))
                    c_index_third += 1
                    self.third_layers.append(SpatialConvLayer(c[c_index_third], c[c_index_third], g))
                    self.third_layers.append(nn.LayerNorm([n, c[c_index]]))
                elif layer_i == 'N':
                    self.first_layers.append(nn.LayerNorm([n, c[c_index]]))
                    self.second_layers.append(TemporalConvLayer_Residual(c_in=c[c_index_second],
                                                                         c_out=c[c_index_second + 1],
                                                                         kernel=(self.kt-1)*2+1))
                    c_index_second += 1
                    self.second_layers.append(SpatialConvLayer(c[c_index_second], c[c_index_second + 1], g))
                    c_index_second += 1
                    self.second_layers.append(TemporalConvLayer_Residual(c_in=c[c_index_second],
                                                                         c_out=c[c_index_second + 1],
                                                                         kernel=(self.kt-1)*2+1))
                    c_index_second += 1

            self.output = OutputLayer(c[c_index], T - (self.kt - 1) * num_temporal_layers, self.num_node, self.nclass)

            for layer in self.first_layers:
                layer.cuda()
            for layer in self.second_layers:
                layer.cuda()
            for layer in self.third_layers:
                layer.cuda()

    def forward(self, x):
        if not self.pyramid:
            for i in range(self.num_layers):
                layer_i = self.control_str[i]
                if layer_i == 'N':
                    x = self.dropout(self.layers[i](x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2))
                else:
                    x = self.layers[i](x)
            return self.output(x)
        else:
            first_index = 0
            second_index = 0
            third_index = 0
            x11 = self.first_layers[first_index](x)
            x21 = self.second_layers[second_index](x)
            x31 = self.third_layers[third_index](x)
            first_index, second_index, third_index = first_index + 1, second_index + 1, third_index + 1
            x12 = self.first_layers[first_index](x11)
            first_index = first_index + 1
            x13 = self.first_layers[first_index](x12)
            first_index = first_index + 1
            x14 = self.first_layers[first_index](x13.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
            first_index = first_index + 1
            x23 = self.second_layers[second_index](x21)
            second_index += 1
            x24 = self.second_layers[second_index](x23)
            second_index += 1
            x15 = self.first_layers[first_index](x14)
            first_index = first_index + 1
            x16 = self.first_layers[first_index](x15)
            first_index = first_index + 1
            x17 = self.first_layers[first_index](x16)
            first_index = first_index + 1
            x18 = self.first_layers[first_index](x17.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
            x26 = self.second_layers[second_index](x24.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
            x33 = self.third_layers[third_index](x31)
            third_index += 1
            x34 = self.third_layers[third_index](x33.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
            x_final = self.finalconv(torch.cat([x34, x26, x18], 3))
            return self.output(x_final)


class STpGCN_ab_top(nn.Module):
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

    def __init__(self, c: list, T: int, kt: int, n: int, g: dgl.DGLGraph, p: float, control_str: str, num_class: int,
                 pyramid: bool):
        super(STpGCN_ab_top, self).__init__()
        self.control_str = control_str
        self.num_layers = len(control_str)
        self.num_node = n
        self.layers = nn.ModuleList()
        self.dropout = nn.Dropout(p)
        self.nclass = num_class
        self.pyramid = pyramid
        self.catconv = nn.Conv2d(32, 32, kernel_size=(1, 2), stride=(1, 2))
        self.finalconv = nn.Conv2d(32, 32, kernel_size=(1, 2), stride=(1, 2))

        # Temporal conv kernel size set to window size / num('T')
        self.kt = kt

        # c_index controls the change of channels
        c_index = 0
        c_index_second = 0
        c_index_third = 0
        num_temporal_layers = 0
        if not pyramid:
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
        else:
            self.first_layers = nn.ModuleList()
            self.second_layers = nn.ModuleList()
            self.third_layers = nn.ModuleList()
            for i in range(self.num_layers):
                layer_i = control_str[i]
                # Temporal layer
                if layer_i == 'T':
                    self.first_layers.append(TemporalConvLayer_Residual(c_in=c[c_index],
                                                                        c_out=c[c_index + 1],
                                                                        kernel=self.kt))
                    c_index += 1
                    num_temporal_layers += 1
                # Spatial layer
                elif layer_i == 'S':
                    self.first_layers.append(SpatialConvLayer(c[c_index], c[c_index + 1], g))
                    c_index += 1
                # Norm layer
                elif layer_i == 'N' and i == self.num_layers - 1:
                    self.first_layers.append(nn.LayerNorm([n, c[c_index]]))
                    self.second_layers.append(nn.LayerNorm([n, c[c_index_second]]))
                    self.third_layers.append(TemporalConvLayer_Residual(c_in=c[c_index_third],
                                                                        c_out=c[c_index_third + 1],
                                                                        kernel=(self.kt - 1) * 4 + 1))
                    c_index_third += 1
                    self.third_layers.append(SpatialConvLayer(c[c_index_third], c[c_index_third], g))
                    self.third_layers.append(nn.LayerNorm([n, c[c_index]]))
                elif layer_i == 'N':
                    self.first_layers.append(nn.LayerNorm([n, c[c_index]]))
                    self.second_layers.append(TemporalConvLayer_Residual(c_in=c[c_index_second],
                                                                         c_out=c[c_index_second + 1],
                                                                         kernel=(self.kt - 1) * 2 + 1))
                    c_index_second += 1
                    self.second_layers.append(SpatialConvLayer(c[c_index_second], c[c_index_second + 1], g))
                    c_index_second += 1
                    self.second_layers.append(TemporalConvLayer_Residual(c_in=c[c_index_second],
                                                                         c_out=c[c_index_second + 1],
                                                                         kernel=(self.kt - 1) * 2 + 1))
                    c_index_second += 1

            self.output = OutputLayer(c[c_index], T - (self.kt - 1) * num_temporal_layers, self.num_node, self.nclass)

            for layer in self.first_layers:
                layer.cuda()
            for layer in self.second_layers:
                layer.cuda()
            for layer in self.third_layers:
                layer.cuda()

    def forward(self, x):
        if not self.pyramid:
            for i in range(self.num_layers):
                layer_i = self.control_str[i]
                if layer_i == 'N':
                    x = self.dropout(self.layers[i](x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2))
                else:
                    x = self.layers[i](x)
            # print(x.shape)
            return self.output(x)
        else:
            first_index = 0
            second_index = 0
            third_index = 0
            x11 = self.first_layers[first_index](x)
            x21 = self.second_layers[second_index](x)
            first_index, second_index, third_index = first_index + 1, second_index + 1, third_index + 1
            x12 = self.first_layers[first_index](x11)
            first_index = first_index + 1
            x13 = self.first_layers[first_index](x12)
            first_index = first_index + 1
            x14 = self.first_layers[first_index](x13.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
            first_index = first_index + 1
            x22 = self.catconv(torch.cat([x21, x14], 3).type(torch.FloatTensor).cuda())
            x23 = self.second_layers[second_index](x22)
            second_index += 1
            x24 = self.second_layers[second_index](x23)
            second_index += 1
            x15 = self.first_layers[first_index](x14)
            first_index = first_index + 1
            x16 = self.first_layers[first_index](x15)
            first_index = first_index + 1
            x17 = self.first_layers[first_index](x16)
            first_index = first_index + 1
            x18 = self.first_layers[first_index](x17.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
            x25 = self.catconv(torch.cat([x24, x18], 3).type(torch.FloatTensor).cuda())
            x26 = self.second_layers[second_index](x25.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
            x_final = self.finalconv(torch.cat([x26, x18], 3).type(torch.FloatTensor).cuda())
            return self.output(x_final)


class STpGCN_ab_mid(nn.Module):
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

    def __init__(self, c: list, T: int, kt: int, n: int, g: dgl.DGLGraph, p: float, control_str: str, num_class: int,
                 pyramid: bool):
        super(STpGCN_ab_mid, self).__init__()
        self.control_str = control_str
        self.num_layers = len(control_str)
        self.num_node = n
        self.layers = nn.ModuleList()
        self.dropout = nn.Dropout(p)
        self.nclass = num_class
        self.pyramid = pyramid
        self.catconv = nn.Conv2d(32, 32, kernel_size=(1, 2), stride=(1, 2))
        self.finalconv = nn.Conv2d(32, 32, kernel_size=(1, 2), stride=(1, 2))

        # Temporal conv kernel size set to window size / num('T')
        self.kt = kt

        # c_index controls the change of channels
        c_index = 0
        c_index_second = 0
        c_index_third = 0
        num_temporal_layers = 0
        if not pyramid:
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
        else:
            self.first_layers = nn.ModuleList()
            self.second_layers = nn.ModuleList()
            self.third_layers = nn.ModuleList()
            for i in range(self.num_layers):
                layer_i = control_str[i]
                # Temporal layer
                if layer_i == 'T':
                    self.first_layers.append(TemporalConvLayer_Residual(c_in=c[c_index],
                                                                        c_out=c[c_index + 1],
                                                                        kernel=self.kt))
                    c_index += 1
                    num_temporal_layers += 1
                # Spatial layer
                elif layer_i == 'S':
                    self.first_layers.append(SpatialConvLayer(c[c_index], c[c_index + 1], g))
                    c_index += 1
                # Norm layer
                elif layer_i == 'N' and i == self.num_layers - 1:
                    self.first_layers.append(nn.LayerNorm([n, c[c_index]]))
                    self.second_layers.append(nn.LayerNorm([n, c[c_index_second]]))
                    self.third_layers.append(TemporalConvLayer_Residual(c_in=c[c_index_third],
                                                                        c_out=c[c_index_third + 1],
                                                                        kernel=(self.kt - 1) * 4 + 1))
                    c_index_third += 1
                    self.third_layers.append(SpatialConvLayer(c[c_index_third], c[c_index_third], g))
                    self.third_layers.append(nn.LayerNorm([n, c[c_index]]))
                elif layer_i == 'N':
                    self.first_layers.append(nn.LayerNorm([n, c[c_index]]))
                    self.second_layers.append(TemporalConvLayer_Residual(c_in=c[c_index_second],
                                                                         c_out=c[c_index_second + 1],
                                                                         kernel=(self.kt - 1) * 2 + 1))
                    c_index_second += 1
                    self.second_layers.append(SpatialConvLayer(c[c_index_second], c[c_index_second + 1], g))
                    c_index_second += 1
                    self.second_layers.append(TemporalConvLayer_Residual(c_in=c[c_index_second],
                                                                         c_out=c[c_index_second + 1],
                                                                         kernel=(self.kt - 1) * 2 + 1))
                    c_index_second += 1

            self.output = OutputLayer(c[c_index], T - (self.kt - 1) * num_temporal_layers, self.num_node, self.nclass)

            for layer in self.first_layers:
                layer.cuda()
            for layer in self.second_layers:
                layer.cuda()
            for layer in self.third_layers:
                layer.cuda()

    def forward(self, x):
        if not self.pyramid:
            for i in range(self.num_layers):
                layer_i = self.control_str[i]
                if layer_i == 'N':
                    x = self.dropout(self.layers[i](x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2))
                else:
                    x = self.layers[i](x)
            # print(x.shape)
            return self.output(x)
        else:
            first_index = 0
            second_index = 0
            third_index = 0
            x11 = self.first_layers[first_index](x)
            x31 = self.third_layers[third_index](x)
            first_index, second_index, third_index = first_index + 1, second_index + 1, third_index + 1
            x12 = self.first_layers[first_index](x11)
            first_index = first_index + 1
            x13 = self.first_layers[first_index](x12)
            first_index = first_index + 1
            x14 = self.first_layers[first_index](x13.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
            first_index = first_index + 1
            x15 = self.first_layers[first_index](x14)
            first_index = first_index + 1
            x16 = self.first_layers[first_index](x15)
            first_index = first_index + 1
            x17 = self.first_layers[first_index](x16)
            first_index = first_index + 1
            x18 = self.first_layers[first_index](x17.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
            x32 = self.catconv(torch.cat([x18, x31], 3).type(torch.FloatTensor).cuda())
            x33 = self.third_layers[third_index](x32)
            third_index += 1
            x34 = self.third_layers[third_index](x33.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
            x_final = self.finalconv(torch.cat([x34, x18], 3))
            return self.output(x_final)


class STGCN_pyramid_feature(nn.Module):
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

    def __init__(self, c: list, T: int, kt: int, n: int, g: dgl.DGLGraph, p: float, control_str: str, num_class: int,
                 pyramid: bool):
        super(STGCN_pyramid_feature, self).__init__()
        self.control_str = control_str
        self.num_layers = len(control_str)
        self.num_node = n
        self.layers = nn.ModuleList()
        self.dropout = nn.Dropout(p)
        self.nclass = num_class
        self.pyramid = pyramid
        self.catconv = nn.Conv2d(32, 32, kernel_size=(1, 2), stride=(1, 2))
        self.finalconv = nn.Conv2d(32, 32, kernel_size=(1, 3), stride=(1, 3))

        # Temporal conv kernel size set to window size / num('T')
        self.kt = kt

        # c_index controls the change of channels
        c_index = 0
        c_index_second = 0
        c_index_third = 0
        num_temporal_layers = 0
        if not pyramid:
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
        else:
            self.first_layers = nn.ModuleList()
            self.second_layers = nn.ModuleList()
            self.third_layers = nn.ModuleList()
            for i in range(self.num_layers):
                layer_i = control_str[i]
                # Temporal layer
                if layer_i == 'T':
                    self.first_layers.append(TemporalConvLayer_Residual(c_in=c[c_index],
                                                                        c_out=c[c_index + 1],
                                                                        kernel=self.kt))
                    c_index += 1
                    num_temporal_layers += 1
                # Spatial layer
                elif layer_i == 'S':
                    self.first_layers.append(SpatialConvLayer(c[c_index], c[c_index + 1], g))
                    c_index += 1
                # Norm layer
                elif layer_i == 'N' and i == self.num_layers - 1:
                    self.first_layers.append(nn.LayerNorm([n, c[c_index]]))
                    self.second_layers.append(nn.LayerNorm([n, c[c_index_second]]))
                    self.third_layers.append(TemporalConvLayer_Residual(c_in=c[c_index_third],
                                                                        c_out=c[c_index_third + 1],
                                                                        kernel=(self.kt-1)*4+1))
                    c_index_third += 1
                    self.third_layers.append(SpatialConvLayer(c[c_index_third], c[c_index_third], g))
                    self.third_layers.append(nn.LayerNorm([n, c[c_index]]))
                elif layer_i == 'N':
                    self.first_layers.append(nn.LayerNorm([n, c[c_index]]))
                    self.second_layers.append(TemporalConvLayer_Residual(c_in=c[c_index_second],
                                                                         c_out=c[c_index_second + 1],
                                                                         kernel=(self.kt-1)*2+1))
                    c_index_second += 1
                    self.second_layers.append(SpatialConvLayer(c[c_index_second], c[c_index_second + 1], g))
                    c_index_second += 1
                    self.second_layers.append(TemporalConvLayer_Residual(c_in=c[c_index_second],
                                                                         c_out=c[c_index_second + 1],
                                                                         kernel=(self.kt-1)*2+1))
                    c_index_second += 1

            self.output = OutputLayer(c[c_index], T - (self.kt - 1) * num_temporal_layers, self.num_node, self.nclass)

            for layer in self.first_layers:
                layer.cuda()
            for layer in self.second_layers:
                layer.cuda()
            for layer in self.third_layers:
                layer.cuda()

    def forward(self, x):
        if not self.pyramid:
            for i in range(self.num_layers):
                layer_i = self.control_str[i]
                if layer_i == 'N':
                    x = self.dropout(self.layers[i](x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2))
                else:
                    x = self.layers[i](x)
            print(x.shape)
            return self.output(x)
        else:
            first_index = 0
            second_index = 0
            third_index = 0
            x11 = self.first_layers[first_index](x)
            x21 = self.second_layers[second_index](x)
            x31 = self.third_layers[third_index](x)
            first_index, second_index, third_index = first_index + 1, second_index + 1, third_index + 1
            x12 = self.first_layers[first_index](x11)
            first_index = first_index + 1
            x13 = self.first_layers[first_index](x12)
            first_index = first_index + 1
            x14 = self.first_layers[first_index](x13.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
            first_index = first_index + 1
            x22 = self.catconv(torch.cat([x21, x14], 3).type(torch.FloatTensor).cuda())
            x23 = self.second_layers[second_index](x22)
            second_index += 1
            x24 = self.second_layers[second_index](x23)
            second_index += 1
            x15 = self.first_layers[first_index](x14)
            first_index = first_index + 1
            x16 = self.first_layers[first_index](x15)
            first_index = first_index + 1
            x17 = self.first_layers[first_index](x16)
            first_index = first_index + 1
            x18 = self.first_layers[first_index](x17.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
            x25 = self.catconv(torch.cat([x24, x18], 3).type(torch.FloatTensor).cuda())
            x26 = self.second_layers[second_index](x25.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
            x32 = self.catconv(torch.cat([x26, x31], 3).type(torch.FloatTensor).cuda())
            x33 = self.third_layers[third_index](x32)
            third_index += 1
            x34 = self.third_layers[third_index](x33.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
            x_final = self.finalconv(torch.cat([x34, x26, x18], 3))
            print(x_final.shape)

            return x_final


class STGCN_pyramid_layer_feature(nn.Module):
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

    def __init__(self, c: list, T: int, kt: int, n: int, g: dgl.DGLGraph, p: float, control_str: str, num_class: int,
                 pyramid: bool):
        super(STGCN_pyramid_layer_feature, self).__init__()
        self.control_str = control_str
        self.num_layers = len(control_str)
        self.num_node = n
        self.layers = nn.ModuleList()
        self.dropout = nn.Dropout(p)
        self.nclass = num_class
        self.pyramid = pyramid
        self.catconv = nn.Conv2d(32, 32, kernel_size=(1, 2), stride=(1, 2))
        self.finalconv = nn.Conv2d(32, 32, kernel_size=(1, 3), stride=(1, 3))

        # Temporal conv kernel size set to window size / num('T')
        self.kt = kt

        # c_index controls the change of channels
        c_index = 0
        c_index_second = 0
        c_index_third = 0
        num_temporal_layers = 0
        if not pyramid:
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
        else:
            self.first_layers = nn.ModuleList()
            self.second_layers = nn.ModuleList()
            self.third_layers = nn.ModuleList()
            for i in range(self.num_layers):
                layer_i = control_str[i]
                # Temporal layer
                if layer_i == 'T':
                    self.first_layers.append(TemporalConvLayer_Residual(c_in=c[c_index],
                                                                        c_out=c[c_index + 1],
                                                                        kernel=self.kt))
                    c_index += 1
                    num_temporal_layers += 1
                # Spatial layer
                elif layer_i == 'S':
                    self.first_layers.append(SpatialConvLayer(c[c_index], c[c_index + 1], g))
                    c_index += 1
                # Norm layer
                elif layer_i == 'N' and i == self.num_layers - 1:
                    self.first_layers.append(nn.LayerNorm([n, c[c_index]]))
                    self.second_layers.append(nn.LayerNorm([n, c[c_index_second]]))
                    self.third_layers.append(TemporalConvLayer_Residual(c_in=c[c_index_third],
                                                                        c_out=c[c_index_third + 1],
                                                                        kernel=(self.kt-1)*4+1))
                    c_index_third += 1
                    self.third_layers.append(SpatialConvLayer(c[c_index_third], c[c_index_third], g))
                    self.third_layers.append(nn.LayerNorm([n, c[c_index]]))
                elif layer_i == 'N':
                    self.first_layers.append(nn.LayerNorm([n, c[c_index]]))
                    self.second_layers.append(TemporalConvLayer_Residual(c_in=c[c_index_second],
                                                                         c_out=c[c_index_second + 1],
                                                                         kernel=(self.kt-1)*2+1))
                    c_index_second += 1
                    self.second_layers.append(SpatialConvLayer(c[c_index_second], c[c_index_second + 1], g))
                    c_index_second += 1
                    self.second_layers.append(TemporalConvLayer_Residual(c_in=c[c_index_second],
                                                                         c_out=c[c_index_second + 1],
                                                                         kernel=(self.kt-1)*2+1))
                    c_index_second += 1

            self.output = OutputLayer(c[c_index], T - (self.kt - 1) * num_temporal_layers, self.num_node, self.nclass)

            for layer in self.first_layers:
                layer.cuda()
            for layer in self.second_layers:
                layer.cuda()
            for layer in self.third_layers:
                layer.cuda()

    def forward(self, x):
        if not self.pyramid:
            for i in range(self.num_layers):
                layer_i = self.control_str[i]
                if layer_i == 'N':
                    x = self.dropout(self.layers[i](x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2))
                else:
                    x = self.layers[i](x)
            print(x.shape)
            return self.output(x)
        else:
            first_index = 0
            second_index = 0
            third_index = 0
            x11 = self.first_layers[first_index](x)
            x21 = self.second_layers[second_index](x)
            x31 = self.third_layers[third_index](x)
            first_index, second_index, third_index = first_index + 1, second_index + 1, third_index + 1
            x12 = self.first_layers[first_index](x11)
            first_index = first_index + 1
            x13 = self.first_layers[first_index](x12)
            first_index = first_index + 1
            x14 = self.first_layers[first_index](x13.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
            first_index = first_index + 1
            x22 = self.catconv(torch.cat([x21, x14], 3).type(torch.FloatTensor).cuda())
            x23 = self.second_layers[second_index](x22)
            second_index += 1
            x24 = self.second_layers[second_index](x23)
            second_index += 1
            x15 = self.first_layers[first_index](x14)
            first_index = first_index + 1
            x16 = self.first_layers[first_index](x15)
            first_index = first_index + 1
            x17 = self.first_layers[first_index](x16)
            first_index = first_index + 1
            x18 = self.first_layers[first_index](x17.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
            x25 = self.catconv(torch.cat([x24, x18], 3).type(torch.FloatTensor).cuda())
            x26 = self.second_layers[second_index](x25.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
            x32 = self.catconv(torch.cat([x26, x31], 3).type(torch.FloatTensor).cuda())
            x33 = self.third_layers[third_index](x32)
            third_index += 1
            x34 = self.third_layers[third_index](x33.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
            return (x34, x26, x18)


