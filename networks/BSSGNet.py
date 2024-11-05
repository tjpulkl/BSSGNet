import torch.nn as nn
from torch.nn import functional as F
import math
import torch.utils.model_zoo as model_zoo
import torch
import numpy as np
from torch.autograd import Variable
affine_par = True
import functools

import sys, os
from utils.attention import InPlaceABNSync, HeightWidthAttention, conv2d, AlignedModule, PSPAlign
from torch.nn import BatchNorm2d as BatchNorm2d
from torch.nn import BatchNorm1d as BatchNorm1d

def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None, fist_dilation=1, multi_grid=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=dilation*multi_grid, dilation=dilation*multi_grid, bias=False)
        self.bn2 = BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=False)
        self.relu_inplace = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.dilation = dilation
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out = out + residual      
        out = self.relu_inplace(out)

        return out

class ASPPModule(nn.Module):
    """
    Reference: 
        Chen, Liang-Chieh, et al. *"Rethinking Atrous Convolution for Semantic Image Segmentation."*
    """
    def __init__(self, features, inner_features=256, out_features=512, dilations=(12, 24, 36)):
        super(ASPPModule, self).__init__()

        self.conv1 = nn.Sequential(nn.AdaptiveAvgPool2d((1,1)),
                                   nn.Conv2d(features, inner_features, kernel_size=1, padding=0, dilation=1, bias=False),
                                   InPlaceABNSync(inner_features))
        self.conv2 = nn.Sequential(nn.Conv2d(features, inner_features, kernel_size=1, padding=0, dilation=1, bias=False),
                                   InPlaceABNSync(inner_features))
        self.conv3 = nn.Sequential(nn.Conv2d(features, inner_features, kernel_size=3, padding=dilations[0], dilation=dilations[0], bias=False),
                                   InPlaceABNSync(inner_features))
        self.conv4 = nn.Sequential(nn.Conv2d(features, inner_features, kernel_size=3, padding=dilations[1], dilation=dilations[1], bias=False),
                                   InPlaceABNSync(inner_features))
        self.conv5 = nn.Sequential(nn.Conv2d(features, inner_features, kernel_size=3, padding=dilations[2], dilation=dilations[2], bias=False),
                                   InPlaceABNSync(inner_features))

        self.bottleneck = nn.Sequential(
            nn.Conv2d(inner_features * 5, out_features, kernel_size=1, padding=0, dilation=1, bias=False),
            InPlaceABNSync(out_features),
            nn.Dropout2d(0.1)
            )
        
    def forward(self, x):

        _, _, h, w = x.size()

        feat1 = F.interpolate(self.conv1(x), size=(h, w), mode='bilinear', align_corners=True)

        feat2 = self.conv2(x)
        feat3 = self.conv3(x)
        feat4 = self.conv4(x)
        feat5 = self.conv5(x)
        out = torch.cat((feat1, feat2, feat3, feat4, feat5), 1)

        bottle = self.bottleneck(out)
        return bottle

class Edge(nn.Module):
    def __init__(self):
        super(Edge, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d( 256, 64, kernel_size=3, padding=1, dilation=1, bias=False),
            InPlaceABNSync(64),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d( 320, 256, kernel_size=3, padding=1, dilation=1, bias=False),
            InPlaceABNSync(256),
        )
        self.conv_edge = nn.Sequential(
            nn.Dropout2d(0.1),
            nn.Conv2d( 256, 2, kernel_size=1, padding=0, dilation=1, bias=True),          
        )
        self.sigmoid = nn.Sigmoid()
    def forward(self, feat, x2 ):
        x2 = self.conv1(x2)
        feat = torch.cat([feat,x2],dim=1)
        feat_edge = self.conv2( feat )
        edge = self.conv_edge( feat_edge )
        edge_sig = torch.argmax(edge, dim=1).unsqueeze(1)
        feat_edge = feat_edge * edge_sig + feat_edge        
        return feat_edge, edge     

class PSPModule(nn.Module):
    """
    Reference: 
        Zhao, Hengshuang, et al. *"Pyramid scene parsing network."*
    """
    def __init__(self, features, out_features=512, sizes=(1, 2, 3, 6)):
        super(PSPModule, self).__init__()

        self.stages = []
        self.stages = nn.ModuleList([self._make_stage(features, out_features, size) for size in sizes])
        self.bottleneck = nn.Sequential(
            nn.Conv2d(features+len(sizes)*out_features, out_features, kernel_size=3, padding=1, dilation=1, bias=False),
            InPlaceABNSync(out_features),
            )        
    def _make_stage(self, features, out_features, size):
        prior = nn.AdaptiveAvgPool2d(output_size=(size, size))
        conv = nn.Conv2d(features, out_features, kernel_size=1, bias=False)
        bn = InPlaceABNSync(out_features)
        return nn.Sequential(prior, conv, bn)

    def forward(self, feats):
        h, w = feats.size(2), feats.size(3)        
        priors = [ F.interpolate(input=stage(feats), size=(h, w), mode='bilinear', align_corners=True) for stage in self.stages] + [feats]
        bottle = self.bottleneck(torch.cat(priors, 1))
        return bottle

class AttentionModule(nn.Module):
    def __init__(self,num_classes):
        super(AttentionModule, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(256, 48, kernel_size=3, padding=1, dilation=1, bias=False),
            InPlaceABNSync(48),
         )
        self.conv2 = nn.Sequential(
            nn.Conv2d(1024, 256, kernel_size=1, padding=0, dilation=1, bias=False),
            InPlaceABNSync(256),
            nn.Conv2d(256, 256, kernel_size=3, padding=1, dilation=1, bias=False),
            InPlaceABNSync(256),
          )
        self.conv3 = nn.Sequential(
            nn.Conv2d(304, 256, kernel_size=3, padding=1, dilation=1, bias=False),
            InPlaceABNSync(256),
            )
        self.conv4 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1, dilation=1, bias=False),
            InPlaceABNSync(256),
            )
        self.seg = nn.Sequential(
            nn.Dropout2d(0.1),
            nn.Conv2d(256, num_classes, kernel_size=1, padding=0, dilation=1, bias=True)
            )
        self.addCAM = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1, dilation=1, bias=False),
            InPlaceABNSync(256),            
            ) 
    def PCM(self, cam, f):
        n,c,h,w = f.size() 
        f = f.view(n,-1,h*w)                                #n,-1,h*w
        cam = cam.view(n,-1,h*w)
        aff = torch.matmul(f.transpose(1,2), f)
        aff = ( c ** -0.5 ) * aff
        aff = F.softmax( aff, dim = 1 )                   
        cam_rv = torch.matmul(cam, aff).view(n,-1,h,w)        
        return cam_rv
    def forward(self, fea, x2, x4 ):
        _,_,h,w=x4.size()
        x2 = F.interpolate(self.conv1(x2), size=(h,w), mode='bilinear', align_corners=True)  
        x4 = self.conv2( x4 )
        xcmb = torch.cat([x4,x2],dim=1)
        xcmb = self.conv3( xcmb )   
        fea = self.conv4( fea )
        with torch.no_grad():
            xM = F.relu( fea.detach() )  
        xM = F.interpolate(xM, size=(h,w), mode='bilinear', align_corners=True)   
        xPCM = F.interpolate( self.PCM( xM, xcmb ),  size=fea.size()[2:], mode='bilinear', align_corners=True )
        x = torch.cat( [fea, xPCM], dim = 1 )
        x = self.addCAM( x )
        seg = self.seg( x )
        return seg, x 

class SPFlt(nn.Module):           #SPFlt
    def __init__(self, feat_in1, feat_in2, feat_mid=256, feat_out=256, class_num=20 ):
        super(SPFlt, self).__init__()
        self.gamma = nn.Parameter(torch.ones(1))       
        self.beta = nn.Parameter(torch.ones(1))
        self.AlignModule = AlignedModule( 256, 256 )
        self.feat1_conv = conv2d(feat_in1, feat_mid,1) 
        self.feat2_conv = conv2d(feat_in2, feat_mid,1)   
        #========================================================
        self.avg_pool = nn.Sequential( 
             nn.AdaptiveAvgPool2d(1),
        )                                                  
        self.glpool_h = nn.Sequential(           
            nn.Linear(feat_mid, feat_mid, bias=False),
            nn.LeakyReLU(inplace=True),                      
            # nn.Softmax( dim = 1 ),                         
        )
        self.glpool_l = nn.Sequential(
            nn.Linear(feat_mid, feat_mid, bias=False),
            nn.LeakyReLU(inplace=True),                        
            # nn.Softmax( dim = 1 ),                          
        )  
        self.conv_pool = nn.Sequential(
            # nn.Dropout2d(0.1),
            nn.Linear(feat_mid, feat_mid//4, bias=False),          
            nn.LeakyReLU(inplace=True),           
            nn.Linear(feat_mid//4, feat_mid, bias=False),
            nn.Sigmoid(),                                                 
            # nn.Softmax( dim = 1 ),                         
        )       
        self.conv2d_h = conv2d( feat_mid, feat_mid, 1 )
        self.conv2d_l = conv2d( feat_mid, feat_mid, 1 )
        #========================================================
        self.feat_cmb = conv2d(feat_mid*2, feat_out,3)        
    def forward(self, feat1, feat2 ):
        feat1 = self.feat1_conv( feat1 )
        feat2 = self.feat2_conv( feat2 )         
        b,c,_,_ = feat1.shape
        avgPool_h = self.avg_pool(feat2)
        avgPool_l = self.avg_pool(feat1)
        avgPool_h = self.glpool_h(avgPool_h.view(b, c))
        avgPool_l = self.glpool_l(avgPool_l.view(b, c))
        avg_diff = (avgPool_h - avgPool_l)
        chl_diff = self.conv_pool(avg_diff)
        avgPool_l =  (self.gamma * chl_diff * avgPool_l + avgPool_l ).view(b, c, 1, 1)               
        avgPool_h =  (self.beta * (1 - chl_diff) * avgPool_h + avgPool_h ).view(b, c, 1, 1)
        feat2 = feat2 * avgPool_h + feat2
        feat2 = self.conv2d_h( feat2 )
        feat1_ = feat1 * avgPool_l + feat1
        feat1_ = self.conv2d_l( feat1_ )        
        #=======================================              
        feat_cmb = self.AlignModule( feat1_,feat2 )      
        feat_cmb = torch.cat([feat_cmb,feat1],dim=1)     
        feats = self.feat_cmb( feat_cmb )
        return feats

class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes, img_width, img_height):
        self.inplanes = 128
        super(ResNet, self).__init__()
        self.conv1 = conv3x3(3, 64, stride=2)
        self.bn1 = BatchNorm2d(64)
        self.relu1 = nn.ReLU(inplace=False)
        self.conv2 = conv3x3(64, 64)
        self.bn2 = BatchNorm2d(64)
        self.relu2 = nn.ReLU(inplace=False)
        self.conv3 = conv3x3(64, 128)
        self.bn3 = BatchNorm2d(128)
        self.relu3 = nn.ReLU(inplace=False)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=1, dilation=2, multi_grid=(1,1,1))        
        self.PSPAlign = PSPAlign(2048, 512, num_classes )     
        self.CDGNet = HeightWidthAttention(512,512, num_classes,[img_width//4, img_height//4] )     
 
        self.seg1 = nn.Sequential(
            nn.Conv2d(1024, 256, kernel_size=1, padding=0, dilation=1, bias=False),
            InPlaceABNSync(256),
            nn.Dropout2d(0.1),
            nn.Conv2d(256, num_classes, kernel_size=1, padding=0, dilation=1, bias=True)
            )         
        self.seg3 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=1, padding=0, dilation=1, bias=False),
            InPlaceABNSync(256),
            nn.Dropout2d(0.1),                                                    #7-6
            nn.Conv2d(256, num_classes, kernel_size=1, padding=0, dilation=1, bias=True)
        )
        self.edge = Edge()
        self.SPFlt1 = SPFlt( 2048, 512, 256, 256, num_classes )
        self.SPFlt2 = SPFlt( 1024, 256, 256, 256, num_classes )
        self.SPFlt3 = SPFlt( 512, 256, 256, 256, num_classes )
        self.SPFlt4 = SPFlt( 256, 256, 256, 256, num_classes )
        self.Attention = AttentionModule( num_classes )
        self.cmbEdge = nn.Sequential(
            nn.Conv2d( 512, 256, kernel_size=3, padding=1, bias = False ),
            InPlaceABNSync(256),
        )        
        #=============================================================================
        for m in self.modules():
            if isinstance(m, (nn.Conv2d,nn.Conv1d) ):
                nn.init.kaiming_normal_(m.weight.data)
            elif isinstance(m, (BatchNorm2d,BatchNorm1d)):
                m.weight.data.fill_(1)
                m.bias.data.zero_()      
    def _make_layer(self, block, planes, blocks, stride=1, dilation=1, multi_grid=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                BatchNorm2d(planes * block.expansion))

        layers = []
        generate_multi_grid = lambda index, grids: grids[index%len(grids)] if isinstance(grids, tuple) else 1
        layers.append(block(self.inplanes, planes, stride,dilation=dilation, downsample=downsample, multi_grid=generate_multi_grid(0, multi_grid)))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=dilation, multi_grid=generate_multi_grid(i, multi_grid)))

        return nn.Sequential(*layers)   

    def forward(self, x):
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.relu3(self.bn3(self.conv3(x)))    
        x = self.maxpool(x)
        x2 = self.layer1(x)                        
        x3 = self.layer2( x2 )                      #1/8 512
        x4 = self.layer3( x3 )                      #1/16 1024
        seg1 = self.seg1( x4 )
        x5 = self.layer4( x4 )                      #1/16 2048
        # x = self.layer5(x5)                       #1/16 512
        x = self.PSPAlign( x5 ) 
        x, hpred, wpred = self.CDGNet(x)               
        feat = self.SPFlt1( x5, x )
        feat = self.SPFlt2( x4, feat )        
        feat = self.SPFlt3( x3, feat )        
        feat = self.SPFlt4( x2, feat )  
      
        seg2, feat = self.Attention( feat, x2, x4 )               #attention       
        feat_edge, edge = self.edge( feat, x2 )                   #edge prediction 
        feat = torch.cat( [feat, feat_edge], dim=1 )              #change to spflt
        feat = self.cmbEdge( feat )           
        seg3 = self.seg3( feat ) 
      
        return [[seg1, seg2,seg3], [edge],[hpred,wpred]]   

def Res_Deeplab(num_classes=21, img_width=384, img_height=384):
    model = ResNet(Bottleneck,[3, 4, 23, 3], num_classes, img_width, img_height )
    return model