import torch.nn as nn
from torch.nn import functional as F
from torch.nn import init
import math
import torch
import numpy as np
from torch.autograd import Variable
affine_par = True
import functools
from torchvision import models
import sys, os
torch.backends.cudnn.enabled = False
# from inplace_abn import InPlaceABN, InPlaceABNSync
BatchNorm = nn.BatchNorm2d

def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, inplanes, planes, stride=1, abn=BatchNorm, dilation=1, downsample=None, fist_dilation=1, multi_grid=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = abn(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=dilation*multi_grid, dilation=dilation*multi_grid, bias=False)
        self.bn2 = abn(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = abn(planes * 4)
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

class Edge_Module(nn.Module):                    ######## contour preservation module

    def __init__(self, abn=BatchNorm, in_fea=[64,128,256], mid_fea=64, out_fea=2):
        super(Edge_Module, self).__init__()
        
        self.conv1 =  nn.Sequential(
            nn.Conv2d(in_fea[0], mid_fea, kernel_size=1, padding=0, dilation=1, bias=False),
            abn(mid_fea)
            ) 
        self.conv2 =  nn.Sequential(
            nn.Conv2d(in_fea[1], mid_fea, kernel_size=1, padding=0, dilation=1, bias=False),
            abn(mid_fea)
            )  
        self.conv3 =  nn.Sequential(
            nn.Conv2d(in_fea[2], mid_fea, kernel_size=1, padding=0, dilation=1, bias=False),
            abn(mid_fea)
        )
        self.conv4 = nn.Conv2d(mid_fea,out_fea, kernel_size=3, padding=1, dilation=1, bias=True)
        self.conv5 = nn.Conv2d(out_fea*3,out_fea, kernel_size=1, padding=0, dilation=1, bias=True)
            

    def forward(self, x1, x2, x3):
        _, _, h, w = x1.size()
        
        edge1_fea = self.conv1(x1)
        edge1 = self.conv4(edge1_fea)
        edge2_fea = self.conv2(x2)
        edge2 = self.conv4(edge2_fea)
        edge3_fea = self.conv3(x3)
        edge3 = self.conv4(edge3_fea)        
        
        edge2_fea =  F.interpolate(edge2_fea, size=(h, w), mode='bilinear',align_corners=True) 
        edge3_fea =  F.interpolate(edge3_fea, size=(h, w), mode='bilinear',align_corners=True) 
        edge2 =  F.interpolate(edge2, size=(h, w), mode='bilinear',align_corners=True)
        edge3 =  F.interpolate(edge3, size=(h, w), mode='bilinear',align_corners=True) 
 
        edge = torch.cat([edge1, edge2, edge3], dim=1)
        edge_fea = torch.cat([edge1_fea, edge2_fea, edge3_fea], dim=1)
        edge = self.conv5(edge)
         
        return edge, edge_fea

class PSPModule(nn.Module):
    def __init__(self, features, out_features=512, abn = BatchNorm, sizes=(1, 2, 3, 6)):
        super(PSPModule, self).__init__()

        self.stages = []
        self.abn = abn
        self.stages = nn.ModuleList([self._make_stage(features, out_features, size) for size in sizes])
        self.bottleneck = nn.Sequential(
            nn.Conv2d(features+len(sizes)*out_features, out_features, kernel_size=3, padding=1, dilation=1, bias=False),
            abn(out_features),
            )

    def _make_stage(self, features, out_features, size):
        prior = nn.AdaptiveAvgPool2d(output_size=(size, size))
        conv = nn.Conv2d(features, out_features, kernel_size=1, bias=False)
        bn = self.abn(out_features)
        return nn.Sequential(prior, conv, bn)

    def forward(self, feats):
        h, w = feats.size(2), feats.size(3)
        priors = [ F.interpolate(input=stage(feats), size=(h, w), mode='bilinear', align_corners=True) for stage in self.stages] + [feats]
        bottle = self.bottleneck(torch.cat(priors, 1))
        return bottle

class Decoder_Module(nn.Module):
    def __init__(self, in_plane1, in_plane2, num_classes, abn=BatchNorm):
        super(Decoder_Module, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_plane1, 256, kernel_size=1, padding=0, dilation=1, bias=False),
            abn(256)
            )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_plane2, 48, kernel_size=1, stride=1, padding=0, dilation=1, bias=False),
            abn(48)
            )
        self.conv3 = nn.Sequential(
            nn.Conv2d(304, 256, kernel_size=1, padding=0, dilation=1, bias=False),
            abn(256),
            nn.Conv2d(256, 256, kernel_size=1, padding=0, dilation=1, bias=False),
            abn(256)
            )
        self.conv4 = nn.Conv2d(256, num_classes, kernel_size=1, padding=0, dilation=1, bias=True)

    def forward(self, xt, xl):
        _, _, h, w = xl.size()

        xt = F.interpolate(self.conv1(xt), size=(h, w), mode='bilinear', align_corners=True)
        xl = self.conv2(xl)
        x = torch.cat([xt, xl], dim=1)
        x = self.conv3(x)
        seg = self.conv4(x)
        return seg,x      

class GCN(nn.Module):
    def __init__(self, num_state, num_node, bias=False):
        super(GCN, self).__init__()
        self.conv1 = nn.Conv1d(num_node, num_node, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(num_state, num_state, kernel_size=1, bias=bias)

    def forward(self, x):
        h = self.conv1(x.permute(0, 2, 1)).permute(0, 2, 1)
        h = h - x
        h = self.relu(self.conv2(h))
        return h


class CGRModule(nn.Module):
    def __init__(self, num_in, plane_mid, mids, abn=BatchNorm, normalize=False):
        super(CGRModule, self).__init__()
        
        self.normalize = normalize
        self.num_s = int(plane_mid)
        self.num_n = (mids) * (mids)
        self.priors = nn.AdaptiveAvgPool2d(output_size=(mids + 2, mids + 2))

        self.conv_state = nn.Conv2d(num_in, self.num_s, kernel_size=1)
        self.conv_proj = nn.Conv2d(num_in, self.num_s, kernel_size=1)
        self.gcn = GCN(num_state=self.num_s, num_node=self.num_n)
        self.conv_extend = nn.Conv2d(self.num_s, num_in, kernel_size=1, bias=False)

        self.blocker = abn(num_in)


    def forward(self, x, edge):
        edge = F.upsample(edge, (x.size()[-2], x.size()[-1]))

        n, c, h, w = x.size()
        edge = torch.nn.functional.softmax(edge, dim=1)[:, 1, :, :].unsqueeze(1)


        # Construct projection matrix
        x_state_reshaped = self.conv_state(x).view(n, self.num_s, -1)
        x_proj = self.conv_proj(x)
        x_mask = x_proj * edge
        x_anchor = self.priors(x_mask)[:, :, 1:-1, 1:-1].reshape(n, self.num_s, -1)
        x_proj_reshaped = torch.matmul(x_anchor.permute(0, 2, 1), x_proj.reshape(n, self.num_s, -1))
        x_proj_reshaped = torch.nn.functional.softmax(x_proj_reshaped, dim=1)
        x_rproj_reshaped = x_proj_reshaped

        # print(x_rproj_reshaped.shape)
      

        # Project and graph reason
        x_n_state = torch.matmul(x_state_reshaped, x_proj_reshaped.permute(0, 2, 1))
        if self.normalize:
            x_n_state = x_n_state * (1. / x_state_reshaped.size(2))
        x_n_rel = self.gcn(x_n_state)

        # Reproject
        x_state_reshaped = torch.matmul(x_n_rel, x_rproj_reshaped)       #x_n_rel   ###没有gcn
        x_state = x_state_reshaped.view(n, self.num_s, *x.size()[2:])
        out = x + self.blocker(self.conv_extend(x_state))
        # print(out.shape)
        return out








class CGRNet(nn.Module):
    def __init__(self,n_channels, n_classes, abn=BatchNorm):
        super(CGRNet, self).__init__()
        ################################vgg16#######################################
        # feats = list(models.vgg16_bn(pretrained=True).features.children())
        # # print(nn.Sequential(*feats[:]))
        # feats[0] = nn.Conv2d(n_channels, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))   #适应任务
        # self.conv1 = nn.Sequential(*feats[:6])
        # # print(self.conv1)
        # self.conv2 = nn.Sequential(*feats[6:13])
        # # print(self.conv2)
        # self.conv3 = nn.Sequential(*feats[13:23])
        # self.conv4 = nn.Sequential(*feats[23:33])
        # self.conv5 = nn.Sequential(*feats[33:43])   #####增强细节
        # print(self.conv5)
       
        ################################Gate#######################################
        resnet = models.resnet18(pretrained=True)
        # print(resnet)
        resnet.conv1 = nn.Conv2d(n_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3))  
        self.firstconv = resnet.conv1
        
        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu
        self.firstmaxpool = resnet.maxpool
        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4
        # print(self.encoder4)






        self.edge_layer = Edge_Module(abn)
        self.block1 = CGRModule(512, 128, 4, abn)
        self.block2 = CGRModule(64, 64, 4, abn)
        self.layer5 = PSPModule(512,512,abn)
        self.layer6 = Decoder_Module(512, 64, n_classes, abn)    #128 原始  2
        self.out = nn.Conv2d(2, 1, 1)
        self.out1 = nn.Conv2d(258, 1, 1)

    def forward(self, x):
        _, _, h, w = x.size()
        x = self.firstconv(x)
        x = self.firstbn(x)
        x = self.firstrelu(x)
        x1 = self.firstmaxpool(x)     ##low-level
        # print(x1.shape)
        x2 = self.encoder1(x1)
        x3 = self.encoder2(x2)
        x4 = self.encoder3(x3)
        x5 = self.encoder4(x4)

        ###########
        x6 = self.layer5(x5)    ##high-level
        # print(x.shape)
        # print(x6.shape)
       
        edge,edge_fea = self.edge_layer(x2,x3,x4)

        x11 = self.block1(x6, edge.detach())               #.detach()
      
        x2 = self.block2(x2, edge.detach())

        seg, x = self.layer6(x11, x2)
        # print(seg.shape)
        # print(x.shape)
    
        
        ##fusion module #####
        seg =torch.cat([x, edge], dim=1)
        seg =self.out1(seg)
        ###############


        edge_out = self.out(edge)
        edge_out = F.interpolate(edge_out, size=(h, w), mode='bilinear', align_corners=True)
        seg = F.interpolate(seg, size=(h, w), mode='bilinear', align_corners=True)
        # fg = torch.sigmoid(seg)
        # p = fg - .5
        # cg = .5 - torch.abs(p)
        # print(cg.shape) 
        return  edge_out, seg









if __name__ == '__main__':
    ras =CGRNet(n_channels=3, n_classes=1).cuda()
    input_tensor = torch.randn(4, 3, 352, 352).cuda()

    out,out1 = ras(input_tensor)
    print(out.shape)
   
