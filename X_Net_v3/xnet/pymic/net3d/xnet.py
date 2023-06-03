# -*- coding: utf-8 -*-
from __future__ import print_function, division

import torch
import torch.nn as nn
import numpy as np
from pymic.layer.activation import get_acti_func
from pymic.layer.convolution import ConvolutionLayer
from pymic.layer.deconvolution import DeconvolutionLayer

class UNetBlock(nn.Module):
    def __init__(self,in_channels, out_channels, acti_func, acti_func_param):
        super(UNetBlock, self).__init__()
        
        self.in_chns   = in_channels
        self.out_chns  = out_channels
        self.acti_func = acti_func

        self.conv1 = ConvolutionLayer(in_channels,  out_channels, 3, 
                padding = 1, acti_func=get_acti_func(acti_func, acti_func_param))
        self.conv2 = ConvolutionLayer(out_channels, out_channels, 3, 
                padding = 1, acti_func=get_acti_func(acti_func, acti_func_param))

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x

class XNet(nn.Module):
    def __init__(self, params):
        super(XNet, self).__init__()
        self.params = params
        self.in_chns   = self.params['in_chns']
        self.ft_chns   = self.params['feature_chns']
        self.n_class   = self.params['class_num']
        self.acti_func = self.params['acti_func']
        self.dropout   = self.params['dropout']
        self.resolution_level = len(self.ft_chns)
        assert(self.resolution_level == 5 or self.resolution_level == 4)
        
        # CT #######################################################################

        self.block1_ct = UNetBlock(self.in_chns, self.ft_chns[0], 
             self.acti_func, self.params)

        self.block2_ct = UNetBlock(self.ft_chns[0], self.ft_chns[1], 
             self.acti_func, self.params)

        self.block3_ct = UNetBlock(self.ft_chns[1], self.ft_chns[2], 
             self.acti_func, self.params)

        self.block4_ct = UNetBlock(self.ft_chns[2], self.ft_chns[3], 
             self.acti_func, self.params)

        if(self.resolution_level == 5):
            self.block5_ct = UNetBlock(self.ft_chns[3], self.ft_chns[4], 
                 self.acti_func, self.params)

            self.block6_ct = UNetBlock(self.ft_chns[3] * 2, self.ft_chns[3], 
                 self.acti_func, self.params)

        self.block7_ct = UNetBlock(self.ft_chns[2] * 2, self.ft_chns[2], 
             self.acti_func, self.params)

        self.block8_ct = UNetBlock(self.ft_chns[1] * 2, self.ft_chns[1], 
             self.acti_func, self.params)

        self.block9_ct = UNetBlock(self.ft_chns[0] * 2, self.ft_chns[0], 
             self.acti_func, self.params)

        self.down1_ct = nn.MaxPool3d(kernel_size = 2)
        self.down2_ct = nn.MaxPool3d(kernel_size = 2)
        self.down3_ct = nn.MaxPool3d(kernel_size = 2)
        if(self.resolution_level == 5):
            self.down4_ct = nn.MaxPool3d(kernel_size = 2)

            self.up1_ct = DeconvolutionLayer(self.ft_chns[4], self.ft_chns[3], kernel_size = 2,
                stride = 2, acti_func = get_acti_func(self.acti_func, self.params))
        self.up2_ct = DeconvolutionLayer(self.ft_chns[3], self.ft_chns[2], kernel_size = 2,
            stride = 2, acti_func = get_acti_func(self.acti_func, self.params))
        self.up3_ct = DeconvolutionLayer(self.ft_chns[2], self.ft_chns[1], kernel_size = 2,
            stride = 2, acti_func = get_acti_func(self.acti_func, self.params))
        self.up4_ct = DeconvolutionLayer(self.ft_chns[1], self.ft_chns[0], kernel_size = 2,
            stride = 2, acti_func = get_acti_func(self.acti_func, self.params))
        
        # MR #######################################################################
        
        self.block1_mr = UNetBlock(self.in_chns, self.ft_chns[0], 
             self.acti_func, self.params)

        self.block2_mr = UNetBlock(self.ft_chns[0], self.ft_chns[1], 
             self.acti_func, self.params)

        self.block3_mr = UNetBlock(self.ft_chns[1], self.ft_chns[2], 
             self.acti_func, self.params)

        self.block4_mr = UNetBlock(self.ft_chns[2], self.ft_chns[3], 
             self.acti_func, self.params)

        if(self.resolution_level == 5):
            self.block5_mr = UNetBlock(self.ft_chns[3], self.ft_chns[4], 
                 self.acti_func, self.params)

            self.block6_mr = UNetBlock(self.ft_chns[3] * 2, self.ft_chns[3], 
                 self.acti_func, self.params)

        self.block7_mr = UNetBlock(self.ft_chns[2] * 2, self.ft_chns[2], 
             self.acti_func, self.params)

        self.block8_mr = UNetBlock(self.ft_chns[1] * 2, self.ft_chns[1], 
             self.acti_func, self.params)

        self.block9_mr = UNetBlock(self.ft_chns[0] * 2, self.ft_chns[0], 
             self.acti_func, self.params)

        self.down1_mr = nn.MaxPool3d(kernel_size = 2)
        self.down2_mr = nn.MaxPool3d(kernel_size = 2)
        self.down3_mr = nn.MaxPool3d(kernel_size = 2)
        if(self.resolution_level == 5):
            self.down4_mr = nn.MaxPool3d(kernel_size = 2)

            self.up1_mr = DeconvolutionLayer(self.ft_chns[4], self.ft_chns[3], kernel_size = 2,
                stride = 2, acti_func = get_acti_func(self.acti_func, self.params))
        self.up2_mr = DeconvolutionLayer(self.ft_chns[3], self.ft_chns[2], kernel_size = 2,
            stride = 2, acti_func = get_acti_func(self.acti_func, self.params))
        self.up3_mr = DeconvolutionLayer(self.ft_chns[2], self.ft_chns[1], kernel_size = 2,
            stride = 2, acti_func = get_acti_func(self.acti_func, self.params))
        self.up4_mr = DeconvolutionLayer(self.ft_chns[1], self.ft_chns[0], kernel_size = 2,
            stride = 2, acti_func = get_acti_func(self.acti_func, self.params))
        
        #################################################################################

        self.fusion = nn.Conv3d(256, 128, kernel_size = 3, padding = 1)

        if(self.dropout):
             self.drop1 = nn.Dropout(p=0.1)
             self.drop2 = nn.Dropout(p=0.1)
             self.drop3 = nn.Dropout(p=0.2)
             self.drop4 = nn.Dropout(p=0.2)
             if(self.resolution_level == 5):
                  self.drop5 = nn.Dropout(p=0.3)
                  
        self.conv_ct = nn.Conv3d(self.ft_chns[0], self.n_class, 
            kernel_size = 3, padding = 1)
        self.conv_mr = nn.Conv3d(self.ft_chns[0], self.n_class, 
            kernel_size = 3, padding = 1)

    def forward(self, x_ct, x_mr):
        f1_ct = self.block1_ct(x_ct)
        f1_mr = self.block1_mr(x_mr)
        if(self.dropout):     # Pas de dropout donc don't care, utile pour future improvements
             f1 = self.drop1(f1)
        d1_ct = self.down1_ct(f1_ct)
        d1_mr = self.down1_mr(f1_mr)

        f2_ct = self.block2_ct(d1_ct)
        f2_mr = self.block2_mr(d1_mr)
        if(self.dropout):
             f2 = self.drop2(f2)
        d2_ct = self.down2_ct(f2_ct)
        d2_mr = self.down2_mr(f2_mr)

        f3_ct = self.block3_ct(d2_ct)
        f3_mr = self.block3_mr(d2_mr)
        if(self.dropout):
             f3 = self.drop3(f3)
        d3_ct = self.down3_ct(f3_ct)
        d3_mr = self.down3_mr(f3_mr)

        f4_ct = self.block4_ct(d3_ct)
        f4_mr = self.block4_mr(d3_mr)
        
        if(self.dropout):
             f4 = self.drop4(f4)

        if(self.resolution_level == 5):
            d4_ct = self.down4_ct(f4_ct)
            d4_mr = self.down4_mr(f4_mr)
            f5_ct = self.block5_ct(d4_ct)
            f5_mr = self.block5_mr(d4_mr)
            if(self.dropout):
                 f5 = self.drop5(f5)

            f5up_ct  = self.up1_ct(f5_ct)
            f5up_mr  = self.up1_mr(f5_mr)
            f4cat_ct = torch.cat((f4_ct, f5up_ct), dim = 1)
            f4cat_mr = torch.cat((f4_mr, f5up_mr), dim = 1)
            f6_ct    = self.block6_ct(f4cat_ct)
            f6_mr    = self.block6_mr(f4cat_mr)
            
            f6_fus = torch.cat((f6_ct, f6_mr), dim=1)
            f6_fus = self.fusion(f6_fus)
            
            f6up_ct  = self.up2_ct(f6_fus)
            f6up_mr  = self.up2_mr(f6_fus)
            f3cat_ct = torch.cat((f3_ct, f6up_ct), dim = 1)
            f3cat_mr = torch.cat((f3_mr, f6up_mr), dim = 1)
        else:
            f4_ct = torch.cat((f4_ct, f4_mr), dim=1)
            f4_mr = torch.cat((f4_ct, f4_mr), dim=1)
            f4up_ct  = self.up2_ct(f4_ct)
            f4up_mr  = self.up2_mr(f4_mr)
            f3cat_ct = torch.cat((f3_ct, f4up_ct), dim = 1)
            f3cat_mr = torch.cat((f3_mr, f4up_mr), dim = 1)
        f7_ct    = self.block7_ct(f3cat_ct)
        f7_mr    = self.block7_mr(f3cat_mr)

        f7up_ct  = self.up3_ct(f7_ct)
        f7up_mr  = self.up3_mr(f7_mr)
        f2cat_ct = torch.cat((f2_ct, f7up_ct), dim = 1)
        f2cat_mr = torch.cat((f2_mr, f7up_mr), dim = 1)
        f8_ct    = self.block8_ct(f2cat_ct)
        f8_mr    = self.block8_mr(f2cat_mr)

        f8up_ct  = self.up4_ct(f8_ct)
        f8up_mr  = self.up4_mr(f8_mr)
        f1cat_ct = torch.cat((f1_ct, f8up_ct), dim = 1)
        f1cat_mr = torch.cat((f1_mr, f8up_mr), dim = 1)
        f9_ct    = self.block9_ct(f1cat_ct)
        f9_mr    = self.block9_mr(f1cat_mr)

        output_ct = self.conv_ct(f9_ct)
        output_mr = self.conv_mr(f9_mr)
        return output_ct, output_mr

if __name__ == "__main__":
    params = {'in_chns':1,
              'feature_chns':[8, 16, 32, 64, 128],
              'class_num': 2,
              'acti_func': 'leakyrelu',
              'leakyrelu_alpha': 0.01,
              'dropout':True}
    Net = XNet(params)
    Net = Net.double()

    # ça sert à tester le net pour voir la forme de ce qui sort
    x_ct  = np.random.rand(1, 1, 48, 80, 96)
    x_mr  = np.random.rand(1, 1, 48, 80, 96)
    xt_ct = torch.from_numpy(x_ct)
    xt_mr = torch.from_numpy(x_mr)
    xt_ct = torch.tensor(xt_ct)
    xt_mr = torch.tensor(xt_mr)
    
    y_ct, y_mr = Net(xt_ct, xt_mr)
    y_ct = y_ct.detach().numpy()
    y_mr = y_mr.detach().numpy()
    print(y_ct.shape, y_mr.shape)
