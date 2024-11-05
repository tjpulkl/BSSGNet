
import numpy as np
import torch
import math
import torch.nn as nn
from torch.nn import Module, Sequential, Conv2d, ReLU,AdaptiveMaxPool2d, AdaptiveAvgPool2d, \
    NLLLoss, BCELoss, CrossEntropyLoss, AvgPool2d, MaxPool2d, Parameter, Linear, Sigmoid, Softmax, Dropout, Embedding
from torch.nn import functional as F
from torch.autograd import Variable
import functools

from torch.nn import BatchNorm2d as BatchNorm2d
from torch.nn import BatchNorm1d as BatchNorm1d
def InPlaceABNSync(in_channel):
    layers = [       
        BatchNorm2d(in_channel),
        # nn.ReLU(inplace=True),          
        nn.LeakyReLU(inplace=True),   
    ]
    return nn.Sequential(*layers)

def conv2d(in_channel, out_channel, kernel_size):
    layers = [
        nn.Conv2d(in_channel, out_channel, kernel_size, padding=kernel_size // 2, bias=False),
        BatchNorm2d(out_channel),
        # nn.ReLU(inplace=True),
        nn.LeakyReLU(inplace=True),  
    ]
    return nn.Sequential(*layers)

def conv1d(in_channel, out_channel, kernel_size):
    layers = [
        nn.Conv1d(in_channel, out_channel, kernel_size, padding=kernel_size // 2, bias=False),
        BatchNorm1d(out_channel),
        # nn.ReLU(inplace=True),
        nn.LeakyReLU(inplace=True),  
    ]
    return nn.Sequential(*layers)

class HeightWidthAttention(nn.Module):
    def  __init__(self, feat_in=512, feat_out=256, num_classes=20, size=[384//16,384//16], kernel_size =7 ):
        super(HeightWidthAttention, self).__init__()   
        h,w = size[0],size[1]
        kSize = kernel_size
        self.gamma = Parameter(torch.ones(num_classes))
        self.rowpool = nn.AdaptiveAvgPool2d((h,1))
        self.colpool = nn.AdaptiveAvgPool2d((1,w))        
        self.conv_hgt1 =conv1d(feat_in,feat_out//2,3)
        self.conv_hgt2 =conv1d(feat_in,feat_out//2,3)
        self.conv_hwPred1 = nn.Sequential(
            conv1d( feat_out//2,feat_out//2,3),                                                        
            nn.Conv1d(feat_out//2,num_classes,1,stride=1,padding=0,bias=True),
            nn.Sigmoid(),   
        )
        self.conv_hwPred2 = nn.Sequential(
            conv1d( feat_out//2,feat_out//2,3),                                                          
            nn.Conv1d(feat_out//2,num_classes,1,stride=1,padding=0,bias=True),
            nn.Sigmoid(),                                                            
         )
         #========================================================================
        # self.conv_clsUp = nn.Sequential(
        #      conv2d( num_classes, feat_out//2, 1),
        #      conv2d( feat_out//2, feat_out, 1 ),       
        #      nn.Sigmoid(),
        #  )         
        self.conv_clsUp = nn.Sequential(
             nn.Conv2d( num_classes, feat_out//2, 1,stride=1,padding=0,bias=False),
             nn.LeakyReLU(inplace=True),
             nn.Conv2d( feat_out//2, feat_out, 1,stride=1,padding=0,bias=False ),   
            #  nn.Conv2d( feat_out//2, feat_out, 1,stride=1,padding=0,bias=True ),    
             nn.Sigmoid(),
         )    
         #========================================================================
        self.conv_upDim1 = nn.Sequential(  
            conv1d(feat_out//2,feat_out,kSize),             
            nn.Conv1d( feat_out, feat_out, 1, stride=1, bias = True),
            nn.Sigmoid(),                                                                              
        )
        self.conv_upDim2 = nn.Sequential(              
            conv1d(feat_out//2,feat_out,kSize),              
            nn.Conv1d( feat_out, feat_out, 1, stride=1, bias = True),
            nn.Sigmoid(),                                                                            
        )
        self.cmbFea = nn.Sequential(
            conv2d( feat_in*2, feat_in, 3 ),
            conv2d( feat_in, feat_in, 1),
        )        
    def forward(self,fea):
        n,c,h,w = fea.size()
        fea_h = self.rowpool(fea).squeeze(3)      #n,c,h
        fea_w = self.colpool(fea).squeeze(2)      #n,c,w
        fea_h = self.conv_hgt1(fea_h)             #n,c,h
        fea_w = self.conv_hgt2(fea_w) 
        #===========================================================               
        fea_hp = self.conv_hwPred1(fea_h)            #n,class_num,h
        fea_wp = self.conv_hwPred2(fea_w)            #n,class_num,w  
        #===========================================================
        fea_h = self.conv_upDim1(fea_h)                    
        fea_w = self.conv_upDim2(fea_w) 
        #============================================================
        gamma = self.gamma.unsqueeze(0).unsqueeze(2).unsqueeze(3)
        gamma = gamma.expand( n, -1, -1,-1)    #n, class_num, 1
        gamma = self.conv_clsUp( gamma )
        beta = 1 - gamma
        #============================================================
        fea_hup = fea_h.unsqueeze(3)
        fea_wup = fea_w.unsqueeze(2)
        fea_hup = F.interpolate( fea_hup, (h,w), mode='bilinear', align_corners= True ) #n,c,h,w
        fea_wup = F.interpolate( fea_wup, (h,w), mode='bilinear', align_corners= True ) #n,c,h,w  
        fea_hw = beta*beta*fea_wup + gamma*gamma*fea_hup #????
        fea = fea*fea_hw+fea
        fea = torch.cat([fea,fea_hw],dim=1)
        fea = self.cmbFea( fea )       
        return fea, fea_hp, fea_wp    

def flow_warp( input, flow, size):
        out_h, out_w = size
        n, c, h, w = input.size()
        # n, c, h, w
        # n, 2, h, w

        norm = torch.tensor([[[[out_w, out_h]]]]).type_as(input).to(input.device)
       
        # s = 2.0
        # norm = torch.tensor([[[[(out_w-1)/s, (out_h-1)/s]]]]).type_as(input).to(input.device) # not [h/s, w/s]
 
        h = torch.linspace(-1.0, 1.0, out_h).view(-1, 1).repeat(1, out_w)
        w = torch.linspace(-1.0, 1.0, out_w).repeat(out_h, 1)
        grid = torch.cat((w.unsqueeze(2), h.unsqueeze(2)), 2)
        grid = grid.repeat(n, 1, 1, 1).type_as(input).to(input.device)
        grid = grid + flow.permute(0, 2, 3, 1) / norm

        output = F.grid_sample(input, grid, align_corners=True)
        return output

class AlignedModule(nn.Module):

    def __init__(self, inplane, outplane, kernel_size=3):
        super(AlignedModule, self).__init__()        
        self.down_h = conv2d(inplane, outplane, 1)   
        self.down_l = conv2d(inplane, outplane, 1)                                          
        self.delta_gen1 =  nn.Sequential(
            nn.Dropout2d(0.1), 
            nn.Conv2d(inplane*2, 2, kernel_size=3, padding=1, bias=False),     
        )
        self.delta_gen2 = nn.Sequential(
            nn.Dropout2d(0.1), 
            nn.Conv2d(inplane*2, 2, kernel_size=3, padding=1, bias=False),     
        )        
        self.weight_conv = nn.Sequential(
            conv2d( outplane, outplane,3),
            nn.Dropout2d(0.1),   
            nn.Conv2d( outplane, 1, kernel_size=3,padding=1,dilation=1,bias=False),     
            nn.Sigmoid(),                     
        )  
    def forward(self, low_feature, h_feature):
        h_feature_orign = h_feature
        l_feature_orign = low_feature
        h, w = low_feature.size()[2:]
        size = (h, w)        
        low_feature = self.down_l(low_feature)
        h_feature = self.down_h(h_feature)       
        h_feature = F.interpolate(h_feature, size=size, mode="bilinear", align_corners=True)   
        feat_cat = torch.cat([h_feature,low_feature],dim=1)
        flow_hi = self.delta_gen1( feat_cat )  
        flow_low = self.delta_gen2( feat_cat )       
        h_feature = flow_warp( h_feature_orign, flow_hi, size=size)            
        l_feature = flow_warp( l_feature_orign, flow_low, size=size) 
        diff = h_feature - l_feature
        fea_wei = self.weight_conv(diff)        
        h_feature = fea_wei * h_feature + h_feature  
        l_feature = (1-fea_wei) * l_feature + l_feature
        feature_add = h_feature + l_feature
        return feature_add    

class PSPAlign(nn.Module):
    def __init__(self, features, in_features, class_num):
        super(PSPAlign, self).__init__()
        self.gamma = Parameter(torch.ones(1))
        self.beta = Parameter(torch.ones(1))
        self.pool = nn.Sequential(
                        nn.AdaptiveAvgPool2d(output_size=(3, 3)),                        
                        conv2d( features, features,1 ),
                        conv2d(features, in_features,3), )                      

        self.adapt = nn.Sequential(
                        conv2d( features, features,1),
                        conv2d(features, in_features, 3),) 
        self.delta_gen1 = nn.Sequential( 
                        conv2d(in_features*2, in_features,3), 
                        nn.Dropout2d(0.1),         
                        nn.Conv2d(in_features, 2, kernel_size=3, padding=1, bias=False)  
                        )        
        #========================================================
        self.avg_pool1 = nn.Sequential( 
            nn.AdaptiveAvgPool2d(1),
            # nn.AdaptiveAvgPool2d(output_size=(3, 3)),  
            # nn.Conv2d( in_features, in_features, kernel_size=3, padding=0, bias=False),
            # InPlaceABNSync(in_features),
        )
        self.avg_pool2 = nn.Sequential( 
            nn.AdaptiveAvgPool2d(1),
            # nn.AdaptiveAvgPool2d(output_size=(3, 3)),  
            # nn.Conv2d( in_features, in_features, kernel_size=3, padding=0, bias=False),
            # InPlaceABNSync(in_features),
        )
        self.glpool_h = nn.Sequential(           
            nn.Linear(in_features, in_features, bias=False),
            nn.LeakyReLU(inplace=True),                          
            # nn.Softmax( dim = 1 ),                          
        )
        self.glpool_l = nn.Sequential(
            nn.Linear(in_features, in_features, bias=False),
            nn.LeakyReLU(inplace=True),                              
            # nn.Softmax( dim = 1 ),                          
        )  
        self.conv_pool = nn.Sequential(
            # nn.Dropout2d(0.1),
            nn.Linear(in_features, in_features//4, bias=False),
            # nn.Linear(in_features, class_num, bias=False),
            nn.LeakyReLU(inplace=True),
            # nn.Linear(class_num, in_features, bias=False), 
            nn.Linear(in_features//4,  in_features, bias=False), 
            nn.Sigmoid(),                                                          
            # nn.Softmax(1),                         
        )        
        self.conv2d_h = conv2d( in_features, in_features, 1 )
        self.conv2d_l = conv2d( in_features, in_features, 1 )
        #========================================================   
        self.convWeight = nn.Sequential(
            conv2d( in_features, in_features,3),
            nn.Dropout2d(0.1),       
            nn.Conv2d(in_features, 1, kernel_size=3, padding=1, bias=False),          
            nn.Sigmoid(),                  
        )
    def forward(self, low_stage):        
        high_stage = self.pool( low_stage )
        low_stage = self.adapt( low_stage)
        b, c, h, w = low_stage.shape
        #=============================================
        avgPool_h = self.avg_pool2(high_stage)
        avgPool_l = self.avg_pool1(low_stage)
        avgPool_h = self.glpool_h(avgPool_h.view(b, c))
        avgPool_l = self.glpool_l(avgPool_l.view(b, c))
        avg_diff = (avgPool_h - avgPool_l)
        chl_diff = self.conv_pool(avg_diff)
        avgPool_l =  (self.gamma*chl_diff * avgPool_l + avgPool_l ).view(b, c, 1, 1)               
        avgPool_h =  (self.beta*(1 - chl_diff) * avgPool_h + avgPool_h ).view(b, c, 1, 1)        #
        high_stage = avgPool_h * high_stage + high_stage 
        high_stage = self.conv2d_h( high_stage ) 
        low_stage = avgPool_l * low_stage + low_stage  
        low_stage = self.conv2d_l( low_stage )
        #=============================================      
        high_stage_up = F.interpolate(input=high_stage, size=(h, w), mode='bilinear', align_corners=True)
        concat_fea = torch.cat((low_stage, high_stage_up), 1)
        delta1 = self.delta_gen1(concat_fea)        
        high_stage = flow_warp(high_stage, delta1, (h, w))  
        #=========================================================
        diff = high_stage - low_stage
        fea_wei = self.convWeight( diff )
        low_stage = ( 1 - fea_wei ) * low_stage + low_stage
        high_stage = fea_wei * high_stage + high_stage
        #=========================================================
        stage_add = low_stage + high_stage
        return stage_add