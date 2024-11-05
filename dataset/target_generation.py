import os
import sys
import numpy as np
import random
import cv2
import torch
from torch.nn import functional as F
def generate_hw_gt( target, class_num = 20, isADE2K=False ):
    if( isADE2K == True):
        class_num = class_num + 1       #only for ADE2K
    h,w = target.shape   
    target = torch.from_numpy(target)
    target_c = target.clone()
    if isADE2K == False:
        target_c[target_c==255]=0
    else:
        target_c[target_c==255]=class_num-1    
    target_c = target_c.long()
    target_c = target_c.view(h*w)
    target_c = target_c.unsqueeze(1)
    target_onehot = torch.zeros(h*w,class_num)
    target_onehot.scatter_( 1, target_c, 1 )      #h*w,class_num
    target_onehot = target_onehot.transpose(0,1)
    target_onehot = target_onehot.view(class_num,h,w)
    # h distribution ground truth
    hgt = torch.zeros((class_num,h))
    hgt=( torch.sum( target_onehot, dim=2 ) ).float()
    if isADE2K == False:
        hgt[0,:]=0
    else:
        hgt = hgt[:-1]
    max = torch.max(hgt,dim=1)[0]         #c,1
    max = max.unsqueeze(1)  
    hgt = hgt / ( max + 1e-5 )     
    # w distribution gound truth
    wgt = torch.zeros((class_num,w))
    wgt=( torch.sum(target_onehot, dim=1 ) ).float()
    if isADE2K == False:
        wgt[0,:]=0  
    else:
        wgt= wgt[:-1] 
    max = torch.max(wgt,dim=1)[0]         #c,1
    max = max.unsqueeze(1)    
    wgt = wgt / ( max + 1e-5 )
    #=====================================   
    # cgt = torch.sum( target_onehot, dim=[1,2]).float()
    # cExs = ( cgt > 0 )
    # cgt = cgt / ( h*w )   
    #====================================================================
    return hgt, wgt#, cgt, cExs ,cch, ccw  gt_hw

def generate_hw_gt2( target, class_num = 20, isADE2K=False ):
    if( isADE2K == True):
        class_num = class_num + 1       #only for ADE2K
    b,h,w = target.size()  
    # print( "lable size:", target.size() )     #4,473,473
    # target = torch.from_numpy(target)
    target_c = target.clone()
    if isADE2K == False:
        target_c[target_c==255]=0
    else:
        target_c[target_c==255]=class_num-1    
    target_c = target_c.long()
    target_c = target_c.view(b,h*w)
    target_c = target_c.unsqueeze(2)
    target_onehot = torch.zeros(b, h*w,class_num)
    target_onehot = target_onehot.cuda(non_blocking=True)
    target_onehot.scatter_( 2, target_c, 1 )      #b, h*w,class_num
    target_onehot = target_onehot.transpose(1,2)
    target_onehot = target_onehot.view(b,class_num,h,w)
    # h distribution ground truth
    hgt = torch.zeros((b,class_num,h))
    hgt = hgt.cuda(non_blocking=True)
    hgt=( torch.sum( target_onehot, dim=3 ) ).float()
    if isADE2K == False:
        hgt[:,0,:]=0
    else:
        hgt = hgt[:-1]
    max = torch.max(hgt,dim=2)[0]         #b,c,1
    max = max.unsqueeze(2)  
    hgt = hgt / ( max + 1e-5 )     
    # w distribution gound truth
    wgt = torch.zeros((b,class_num,w))
    wgt = wgt.cuda(non_blocking=True)
    wgt=( torch.sum(target_onehot, dim=2 ) ).float()
    if isADE2K == False:
        wgt[:,0,:]=0  
    else:
        wgt= wgt[:-1] 
    max = torch.max(wgt,dim=2)[0]         #c,1
    max = max.unsqueeze(2)    
    wgt = wgt / ( max + 1e-5 )     
    #====================================================================
    return hgt, wgt

def generate_edge(label, edge_width=3):
    label = label.type(torch.cuda.FloatTensor)
    if len(label.shape) == 2:
        label = label.unsqueeze(0)
    n, h, w = label.shape
    edge = torch.zeros(label.shape, dtype=torch.float).cuda()
    # right
    edge_right = edge[:, 1:h, :]
    edge_right[(label[:, 1:h, :] != label[:, :h - 1, :]) & (label[:, 1:h, :] != 255)
               & (label[:, :h - 1, :] != 255)] = 1

    # up
    edge_up = edge[:, :, :w - 1]
    edge_up[(label[:, :, :w - 1] != label[:, :, 1:w])
            & (label[:, :, :w - 1] != 255)
            & (label[:, :, 1:w] != 255)] = 1

    # upright
    edge_upright = edge[:, :h - 1, :w - 1]
    edge_upright[(label[:, :h - 1, :w - 1] != label[:, 1:h, 1:w])
                 & (label[:, :h - 1, :w - 1] != 255)
                 & (label[:, 1:h, 1:w] != 255)] = 1

    # bottomright
    edge_bottomright = edge[:, :h - 1, 1:w]
    edge_bottomright[(label[:, :h - 1, 1:w] != label[:, 1:h, :w - 1])
                     & (label[:, :h - 1, 1:w] != 255)
                     & (label[:, 1:h, :w - 1] != 255)] = 1

    kernel = torch.ones((1, 1, edge_width, edge_width), dtype=torch.float).cuda()
    with torch.no_grad():
        edge = edge.unsqueeze(1)
        edge = F.conv2d(edge, kernel, stride=1, padding=1)
    edge[edge!=0] = 1
    edge = edge.squeeze()
    return edge



#     h, w = label.shape
#     edge = np.zeros(label.shape)

#     # right
#     edge_right = edge[1:h, :]
#     edge_right[(label[1:h, :] != label[:h - 1, :]) & (label[1:h, :] != 255)
#                & (label[:h - 1, :] != 255)] = 1

#     # up
#     edge_up = edge[:, :w - 1]
#     edge_up[(label[:, :w - 1] != label[:, 1:w])
#             & (label[:, :w - 1] != 255)
#             & (label[:, 1:w] != 255)] = 1

#     # upright
#     edge_upright = edge[:h - 1, :w - 1]
#     edge_upright[(label[:h - 1, :w - 1] != label[1:h, 1:w])
#                  & (label[:h - 1, :w - 1] != 255)
#                  & (label[1:h, 1:w] != 255)] = 1

#     # bottomright
#     edge_bottomright = edge[:h - 1, 1:w]
#     edge_bottomright[(label[:h - 1, 1:w] != label[1:h, :w - 1])
#                      & (label[:h - 1, 1:w] != 255)
#                      & (label[1:h, :w - 1] != 255)] = 1

#     kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (edge_width, edge_width))
#     edge = cv2.dilate(edge, kernel)
#     return edge
  
# def _box2cs(box, aspect_ratio, pixel_std):
#     x, y, w, h = box[:4]
#     return _xywh2cs(x, y, w, h, aspect_ratio, pixel_std)


# def _xywh2cs(x, y, w, h, aspect_ratio, pixel_std):
#     center = np.zeros((2), dtype=np.float32)
#     center[0] = x + w * 0.5
#     center[1] = y + h * 0.5

#     if w > aspect_ratio * h:
#         h = w * 1.0 / aspect_ratio
#     elif w < aspect_ratio * h:
#         w = h * aspect_ratio
#     scale = np.array(
#         [w * 1.0 / pixel_std, h * 1.0 / pixel_std], dtype=np.float32)
#     # if center[0] != -1:
#     #    scale = scale * 1.25

#     return center, scale
