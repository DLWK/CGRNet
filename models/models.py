import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import numpy as np
from lib.nn import SynchronizedBatchNorm2d
from smoothgrad import generate_smooth_grad
from guided_backprop import GuidedBackprop
from vanilla_backprop import VanillaBackprop
from misc_functions import (get_example_params,
                            convert_to_grayscale,
                            save_gradient_images,
                            get_positive_negative_saliency)
import math
from .attention_blocks import DualAttBlock
from .resnet import BasicBlock as ResBlock
from . import GSConv as gsc
import cv2
from .norm import Norm2d
from .multi_scale import ASPP,GPM,PPM,FoldConv_aspp,PAFEM
from .wassp import *
from functools import partial
nonlinearity = partial(F.relu, inplace=True)
class SegmentationModuleBase(nn.Module):
    def __init__(self):
        super(SegmentationModuleBase, self).__init__()

    def intersectionAndUnion(self, imPred, imLab, numClass):
        imPred = np.asarray(imPred.cpu()).copy()
        imLab = np.asarray(imLab.cpu()).copy()

        imPred += 1
        imLab += 1
        # Remove classes from unlabeled pixels in gt image.
        # We should not penalize detections in unlabeled portions of the image.
        imPred = imPred * (imLab > 0)

        # Compute area intersection:
        intersection = imPred * (imPred == imLab)
        (area_intersection, _) = np.histogram(
        intersection, bins=numClass, range=(1, numClass))

        # Compute area union:
        (area_pred, _) = np.histogram(imPred, bins=numClass, range=(1, numClass))
        (area_lab, _) = np.histogram(imLab, bins=numClass, range=(1, numClass))
        area_union = area_pred + area_lab - area_intersection

        jaccard = area_intersection/area_union
        #print("I: " + str(area_intersection))
        #print("U: " + str(area_union))
        jaccard = (jaccard[1]+jaccard[2])/2
        return jaccard if jaccard <= 1 else 0

    def pixel_acc(self, pred, label, num_class):
        _, preds = torch.max(pred, dim=1)
        valid = (label >= 1).long()
        acc_sum = torch.sum(valid * (preds == label).long())
        pixel_sum = torch.sum(valid)

        acc = acc_sum.float() / (pixel_sum.float() + 1e-10) #When you +falsePos, acc == Jaccard.
        
        jaccard = []
         
        # calculate jaccard for classes indexed 1 to num_class-1.
        for i in range(1, num_class):
            v = (label == i).long()
            pred = (preds == i).long()
            anb = torch.sum(v * pred)
            try:
                j = anb.float() / (torch.sum(v).float() + torch.sum(pred).float() - anb.float() + 1e-10)
            except:
                j = 0

            j = j if j <= 1 else 0
            jaccard.append(j)

        return acc, jaccard

    def jaccard(self, pred, label):
        AnB = torch.sum(pred.long() & label)
        return AnB/(pred.view(-1).sum().float() + label.view(-1).sum().float() - AnB)

class SegmentationModule(SegmentationModuleBase):
    def __init__(self, crit, unet, num_class):
        super(SegmentationModule, self).__init__()
        self.crit = crit
        self.unet = unet
        self.num_class = num_class

    def forward(self, feed_dict, epoch, *, segSize=None):
        #training
        if segSize is None:
            p = self.unet(feed_dict['image'])
            loss = self.crit(p, feed_dict['mask'], epoch=epoch)
            acc = self.pixel_acc(torch.round(nn.functional.softmax(p[0], dim=1)).long(), feed_dict['mask'][0].long().cuda(), self.num_class)
            return loss, acc

        #test
        if segSize == True:
            p = self.unet(feed_dict['image'])[0]
            pred = nn.functional.softmax(p, dim=1)
            return pred

        #inference
        else:
            p = self.unet(feed_dict['image'])
            loss = self.crit((p[0], p[1]), (feed_dict['mask'][0].long().unsqueeze(0), feed_dict['mask'][1]))
            pred = nn.functional.softmax(p[0], dim=1)
            return pred, loss

    def SRP(self, model, pred, seg):
        output = pred #actual output
        _, pred = torch.max(nn.functional.softmax(pred, dim=1), dim=1) #class index prediction

        tmp = []
        for i in range(output.shape[1]):
            tmp.append((pred == i).long().unsqueeze(1))
        T_model = torch.cat(tmp, dim=1).float()

        LRP = self.layer_relevance_prop(model, output * T_model)

        return LRP

    def layer_relevance_prop(self, model, R):
        for l in range(len(model.layers), 0, -1):
            print(model.layers[l-1])
            R  = model.layers[l-1].relprop(R)
        return R


def conv3x3(in_planes, out_planes, stride=1, has_bias=False):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=has_bias)


def conv3x3_bn_relu(in_planes, out_planes, stride=1):
    return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(out_planes),
            nn.ReLU(inplace=True)
            )

def conv1x1_bn_relu(in_planes, out_planes, stride=1):
    return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, padding=0),
            nn.BatchNorm2d(out_planes),
            nn.ReLU(inplace=True)
            )

class ConvRelu(nn.Module):
    def __init__(self, in_, out):
        super(ConvRelu, self).__init__()
        self.conv = conv3x3(in_, out)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.activation(x)
        return x

class ModelBuilder():
    # custom weights initialization
    def weights_init(self, m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.kaiming_normal_(m.weight.data)
        elif classname.find('BatchNorm') != -1:
            m.weight.data.fill_(1.)
            m.bias.data.fill_(1e-4)

    def build_unet(self, num_class=1, arch='albunet', weights=''):
        arch = arch.lower()

        if arch == 'saunet':
            unet = SAUNet(num_classes=num_class)
        else:
            raise Exception('Architecture undefined!')

        if len(weights) > 0:
            unet.load_state_dict(
                    torch.load(weights, map_location=lambda storage, loc: storage), strict=False)
            print("Loaded pretrained UNet weights.")
        print('Loaded weights for unet')
        return unet

class CenterBlock(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels, is_deconv=True):
        super(CenterBlock, self).__init__()
        self.in_channels = in_channels
        self.is_deconv=is_deconv

        if self.is_deconv: #Dense Block
            self.conv1 = conv3x3_bn_relu(in_channels, middle_channels)
            self.conv2 = conv3x3_bn_relu(in_channels+middle_channels, middle_channels)
            self.convUp = nn.Sequential(
                                nn.ConvTranspose2d(in_channels+2*middle_channels, out_channels,
                                                kernel_size=4, stride=2, padding=1),
                                nn.ReLU(inplace=True)
            )

        else:
            self.convUp = nn.Unsample(scale_factor=2, mode='bilinear', align_corners=True),
            self.conv1 = conv3x3_bn_relu(in_channels, middle_channels),
            self.conv2 = conv3x3_bn_relu(in_channels+middle_channels, out_channels)

    def forward(self, x):
        tmp = []
        if self.is_deconv == False:
            convUp = self.convUp(x); tmp.append(convUp)
            conv1 = self.conv1(convUp); tmp.append(conv1)
            conv2 = self.conv2(torch.cat(tmp, 1))
            return conv2

        else:
            tmp.append(x)
            conv1 = self.conv1(x); tmp.append(conv1)
            conv2 = self.conv2(torch.cat(tmp, 1)); tmp.append(conv2)
            convUp = self.convUp(torch.cat(tmp, 1))
            return convUp

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels, is_deconv=True):
        super(DecoderBlock, self).__init__()
        self.in_channels = in_channels

        if is_deconv:
            self.block = nn.Sequential(
                conv3x3_bn_relu(in_channels, middle_channels),
                nn.ConvTranspose2d(middle_channels, out_channels, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )
        else:
            self.block = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                conv3x3_bn_relu(in_channels, middle_channels),
                conv3x3_bn_relu(middle_channels, out_channels),
            )

        ### initialize
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, SynchronizedBatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def forward(self, x):
        return self.block(x)

class SkipConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        self.block = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True))

        ## initialize
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, SynchronizedBatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def forward(self, x):
        return self.block(x)


class SAUNet(nn.Module): #SAUNet
    def __init__(self, num_classes=4, num_filters=32, pretrained=True, is_deconv=True):
        super(SAUNet, self).__init__()

        self.num_classes = num_classes
        print("SAUNet w/ Shape Stream")
        self.pool = nn.MaxPool2d(2,2)
        self.encoder = torchvision.models.densenet121(pretrained=False)

        #for n, p in self.encoder.named_parameters():
        #    print(n)

        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

        #Shape Stream
        self.c3 = nn.Conv2d(256, 1, kernel_size=1)
        self.c4 = nn.Conv2d(512, 1, kernel_size=1)
        self.c5 = nn.Conv2d(1024, 1, kernel_size=1)

        self.d0 = nn.Conv2d(128, 64, kernel_size=1)
        self.res1 = ResBlock(64, 64)
        self.d1 = nn.Conv2d(64, 32, kernel_size=1)
        self.res2 = ResBlock(32, 32)
        self.d2 = nn.Conv2d(32, 16, kernel_size=1)
        self.res3 = ResBlock(16, 16)
        self.d3 = nn.Conv2d(16, 8, kernel_size=1)
        self.fuse = nn.Conv2d(8, 1, kernel_size=1, padding=0, bias=False)

        self.cw = nn.Conv2d(2, 1, kernel_size=1, padding=0, bias=False)

        self.gate1 = gsc.GatedSpatialConv2d(32, 32)
        self.gate2 = gsc.GatedSpatialConv2d(16, 16)
        self.gate3 = gsc.GatedSpatialConv2d(8, 8)

        self.expand = nn.Sequential(nn.Conv2d(1, num_filters, kernel_size=1),
                                    Norm2d(num_filters),
                                    nn.ReLU(inplace=True))

        #Encoder
        self.encoder.features.conv0 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.conv1 = nn.Sequential(self.encoder.features.conv0,
                                   self.encoder.features.norm0)
        self.conv2 = self.encoder.features.denseblock1
        self.conv2t = self.encoder.features.transition1
        self.conv3 = self.encoder.features.denseblock2
        self.conv3t = self.encoder.features.transition2
        self.conv4 = self.encoder.features.denseblock3
        self.conv4t = self.encoder.features.transition3
        self.conv5 = nn.Sequential(self.encoder.features.denseblock4,
                                   self.encoder.features.norm5)

        #Decoder
        self.center = conv3x3_bn_relu(1024, num_filters * 8 * 2)
        self.dec5 = DualAttBlock(inchannels=[512, 1024], outchannels=512)
        self.dec4 = DualAttBlock(inchannels=[512, 512], outchannels=256)
        self.dec3 = DualAttBlock(inchannels=[256, 256], outchannels=128)
        self.dec2 = DualAttBlock(inchannels=[128,  128], outchannels=64)
        self.dec1 = DecoderBlock(64, 48, num_filters, is_deconv)
        self.dec0 = conv3x3_bn_relu(num_filters*2, num_filters)

        self.final = nn.Conv2d(num_filters, self.num_classes, kernel_size=1)

    def forward(self, x):
        x_size = x.size()

        #Encoder
        conv1 = self.conv1(x)
        conv2 = self.conv2t(self.conv2(conv1))
        conv3 = self.conv3t(self.conv3(conv2))
        conv4 = self.conv4t(self.conv4(conv3))
        conv5 = self.conv5(conv4)

        #Shape Stream
        ss = F.interpolate(self.d0(conv2), x_size[2:],
                            mode='bilinear', align_corners=True)
        ss = self.res1(ss)
        c3 = F.interpolate(self.c3(conv3), x_size[2:],
                            mode='bilinear', align_corners=True)
        ss = self.d1(ss)
        ss = self.gate1(ss, c3)
        ss = self.res2(ss)
        ss = self.d2(ss)
        c4 = F.interpolate(self.c4(conv4), x_size[2:],
                            mode='bilinear', align_corners=True)
        ss = self.gate2(ss, c4)
        ss = self.res3(ss)
        ss = self.d3(ss)
        c5 = F.interpolate(self.c5(conv5), x_size[2:],
                            mode='bilinear', align_corners=True)
        ss = self.gate3(ss, c5)
        ss = self.fuse(ss)
        ss = F.interpolate(ss, x_size[2:], mode='bilinear', align_corners=True)
        edge_out = self.sigmoid(ss)

        ### Canny Edge
        im_arr = np.mean(x.cpu().numpy(), axis=1).astype(np.uint8)
        canny = np.zeros((x_size[0], 1, x_size[2], x_size[3]))
        for i in range(x_size[0]):
            canny[i] = cv2.Canny(im_arr[i], 10, 100)
        canny = torch.from_numpy(canny).cuda().float()
        ### End Canny Edge

        cat = torch.cat([edge_out, canny], dim=1)
        acts = self.cw(cat)
        acts = self.sigmoid(acts)
       
        edge = self.expand(acts)

        #Decoder
        conv2 = F.interpolate(conv2, scale_factor=2, mode='bilinear', align_corners=True)
        conv3 = F.interpolate(conv3, scale_factor=2, mode='bilinear', align_corners=True)
        conv4 = F.interpolate(conv4, scale_factor=2, mode='bilinear', align_corners=True)

        center = self.center(self.pool(conv5))
        dec5, _ = self.dec5([center, conv5])
        dec4, _ = self.dec4([dec5, conv4])
        dec3, att = self.dec3([dec4, conv3])
        dec2, _ = self.dec2([dec3, conv2])
        dec1 = self.dec1(dec2)
        dec0 = self.dec0(torch.cat([dec1, edge], dim=1))

        x_out = self.final(dec0)

        att = F.interpolate(att, scale_factor=4, mode='bilinear', align_corners=True)

        return x_out, ss#, att

    def pad(self, x, y):
        diffX = y.shape[3] - x.shape[3]
        diffY = y.shape[2] - x.shape[2]

        return nn.functional.pad(x, (diffX // 2, diffX - diffX//2,
                                        diffY // 2, diffY - diffY //2))
################ours##############

class sSE(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.Conv1x1 = nn.Conv2d(in_channels, 1, kernel_size=1, bias=False)
        # self.Conv1x1 = nn.Conv2d(in_channels=channels, out_channels=1,kernel_size=1,stride=stride,padding=1,groups=1,bias=False)
        self.norm = nn.Sigmoid()

    def forward(self, U):
        q = self.Conv1x1(U)  # U:[bs,c,h,w] to q:[bs,1,h,w]
        q = self.norm(q)
        return U * q  # 广播机制

class cSE(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.Conv_Squeeze = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1, bias=False)   
        self.Conv_Excitation = nn.Conv2d(in_channels//8, in_channels, kernel_size=1, bias=False)
        self.norm = nn.Sigmoid()

    def forward(self, U):
        z = self.avgpool(U)# shape: [bs, c, h, w] to [bs, c, 1, 1]
        z = self.Conv_Squeeze(z) # shape: [bs, c/8]
        z = self.Conv_Excitation(z) # shape: [bs, c]
        z = self.norm(z)
        return U * z.expand_as(U)

class scSE(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.cSE = cSE(in_channels)
        self.sSE = sSE(in_channels)

    def forward(self, U):
        U_sse = self.sSE(U)
        U_cse = self.cSE(U)
        return U_cse+U_sse

class ContextBlock(nn.Module):
    def __init__(self,inplanes,ratio,pooling_type='att',
                 fusion_types=('channel_add', )):
        super(ContextBlock, self).__init__()
        valid_fusion_types = ['channel_add', 'channel_mul']
 
        assert pooling_type in ['avg', 'att']
        assert isinstance(fusion_types, (list, tuple))
        assert all([f in valid_fusion_types for f in fusion_types])
        assert len(fusion_types) > 0, 'at least one fusion should be used'
 
        self.inplanes = inplanes
        self.ratio = ratio
        self.planes = int(inplanes * ratio)
        self.pooling_type = pooling_type
        self.fusion_types = fusion_types
 
        if pooling_type == 'att':
            self.conv_mask = nn.Conv2d(inplanes, 1, kernel_size=1)
            self.softmax = nn.Softmax(dim=2)
        else:
            self.avg_pool = nn.AdaptiveAvgPool2d(1)
        if 'channel_add' in fusion_types:
            self.channel_add_conv = nn.Sequential(
                nn.Conv2d(self.inplanes, self.planes, kernel_size=1),
                nn.LayerNorm([self.planes, 1, 1]),
                nn.ReLU(inplace=True),  # yapf: disable
                nn.Conv2d(self.planes, self.inplanes, kernel_size=1))
        else:
            self.channel_add_conv = None
        if 'channel_mul' in fusion_types:
            self.channel_mul_conv = nn.Sequential(
                nn.Conv2d(self.inplanes, self.planes, kernel_size=1),
                nn.LayerNorm([self.planes, 1, 1]),
                nn.ReLU(inplace=True),  # yapf: disable
                nn.Conv2d(self.planes, self.inplanes, kernel_size=1))
        else:
            self.channel_mul_conv = None
 
 
    def spatial_pool(self, x):
        batch, channel, height, width = x.size()
        if self.pooling_type == 'att':
            input_x = x
            # [N, C, H * W]
            input_x = input_x.view(batch, channel, height * width)
            # [N, 1, C, H * W]
            input_x = input_x.unsqueeze(1)
            # [N, 1, H, W]
            context_mask = self.conv_mask(x)
            # [N, 1, H * W]
            context_mask = context_mask.view(batch, 1, height * width)
            # [N, 1, H * W]
            context_mask = self.softmax(context_mask)
            # [N, 1, H * W, 1]
            context_mask = context_mask.unsqueeze(-1)
            # [N, 1, C, 1]
            context = torch.matmul(input_x, context_mask)
            # [N, C, 1, 1]
            context = context.view(batch, channel, 1, 1)
        else:
            # [N, C, 1, 1]
            context = self.avg_pool(x)
        return context
 
    def forward(self, x):
        # [N, C, 1, 1]
        context = self.spatial_pool(x)
        out = x
        if self.channel_mul_conv is not None:
            # [N, C, 1, 1]
            channel_mul_term = torch.sigmoid(self.channel_mul_conv(context))
            out = out * channel_mul_term
        if self.channel_add_conv is not None:
            # [N, C, 1, 1]
            channel_add_term = self.channel_add_conv(context)
            out = out + channel_add_term






class DilatedBlock(nn.Module):
	
	def __init__(self, in_ch, out_ch, stride=1, dilation = 1):
		super(DilatedBlock, self).__init__()
		self.conv = nn.Sequential(
			# GCT(in_ch),
			nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False),
			nn.BatchNorm2d(out_ch),
			nn.ReLU(inplace=True),
		    # GCT(out_ch),
			nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=stride, padding=(dilation*(3-1)) // 2, bias=False, dilation=dilation),
			nn.BatchNorm2d(out_ch),
			nn.ReLU(inplace=True),
			# GCT(out_ch),
			nn.Conv2d(out_ch, out_ch, kernel_size=1, bias=False),
			nn.BatchNorm2d(out_ch)
		)
		
		self.relu = nn.ReLU(inplace=True)
		self.se = scSE(out_ch)
		
		self.channel_conv = nn.Sequential(
			# GCT(in_ch),
			nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1, bias=False),
			nn.BatchNorm2d(out_ch)
		)
	
	def forward(self, x):

		residual = x
		out = self.conv(x)
		out = self.se(out)
		if residual.shape[1] != out.shape[1]:
			residual = self.channel_conv(residual)
		out += residual
		out = self.relu(out)
		
		return out
	
class DilatedBottleneck(nn.Module):
	
	def __init__(self, in_ch, out_ch, stride=1, dilations = [1, 2, 4, 8]):
		super(DilatedBottleneck, self).__init__()                                                 #1 
		self.block_num = len(dilations)
		self.dilatedblock1 = DilatedBlock(in_ch, out_ch, stride=stride, dilation=dilations[0])
		self.dilatedblock2 = DilatedBlock(in_ch, out_ch, stride=stride, dilation=dilations[1])
		self.dilatedblock3 = DilatedBlock(in_ch, out_ch, stride=stride, dilation=dilations[2])
		self.dilatedblock4 = DilatedBlock(in_ch, out_ch, stride=stride, dilation=dilations[3])
	
	def forward(self, x):
		x1 = self.dilatedblock1(x)
		x2 = self.dilatedblock2(x1)
		x3 = self.dilatedblock3(x2)
		x4 = self.dilatedblock4(x3)
		
		x_out = x1 + x2 + x3 + x4
		# print(x1.size())
		# print(x2.size())
		# print(x3.size())
		# print(x4.size())
		# print(x_out.size())
	
		return x_out

class SPPblock(nn.Module):
    def __init__(self, in_channels):
        super(SPPblock, self).__init__()
        self.pool1 = nn.MaxPool2d(kernel_size=[2, 2], stride=2)
        self.pool2 = nn.MaxPool2d(kernel_size=[3, 3], stride=3)
        self.pool3 = nn.MaxPool2d(kernel_size=[5, 5], stride=5)
        self.pool4 = nn.MaxPool2d(kernel_size=[6, 6], stride=6)

        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=1, kernel_size=1, padding=0)

    def forward(self, x):
        self.in_channels, h, w = x.size(1), x.size(2), x.size(3)
        self.layer1 = F.upsample(self.conv(self.pool1(x)), size=(h, w), mode='bilinear')
        self.layer2 = F.upsample(self.conv(self.pool2(x)), size=(h, w), mode='bilinear')
        self.layer3 = F.upsample(self.conv(self.pool3(x)), size=(h, w), mode='bilinear')
        self.layer4 = F.upsample(self.conv(self.pool4(x)), size=(h, w), mode='bilinear')

        out = torch.cat([self.layer1, self.layer2, self.layer3, self.layer4, x], 1)

        return out

def conv3x3_bn_relu(in_planes, out_planes, stride=1):
    return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(out_planes),
            nn.ReLU(inplace=True)
            )

def conv1x1_bn_relu(in_planes, out_planes, stride=1):
    return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, padding=0),
            nn.BatchNorm2d(out_planes),
            nn.ReLU(inplace=True)
            )


# class sSE(nn.Module):
#     def __init__(self, in_channels):
#         super().__init__()
#         self.Conv1x1 = nn.Conv2d(in_channels, 1, kernel_size=1, bias=False)
#         # self.Conv1x1 = nn.Conv2d(in_channels=channels, out_channels=1,kernel_size=1,stride=stride,padding=1,groups=1,bias=False)
#         self.norm = nn.Sigmoid()

#     def forward(self, U):
#         q = self.Conv1x1(U)  # U:[bs,c,h,w] to q:[bs,1,h,w]
#         q = self.norm(q)
#         return U * q  # 广播机制





class WASP(nn.Module):
    def __init__(self, channel, ):
        super(WASP, self).__init__()
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.bn=nn.ModuleList([nn.BatchNorm2d(channel),nn.BatchNorm2d(channel),nn.BatchNorm2d(channel),nn.BatchNorm2d(channel)]) 
        self.convs1 =nn.Conv2d(channel, channel, kernel_size=1, padding=0)
        self.conv3x3=nn.Conv2d(channel, channel,dilation=2,kernel_size=3, padding=2)
    
        self.convs0 =nn.Conv2d(channel, channel, kernel_size=1, padding=0)
        self.Conv1x1 = nn.Conv2d(channel * 5, 1, kernel_size=1, bias=False)
        self.fuse = nn.Sequential(
        nn.Conv2d(channel * 5,   channel , kernel_size=1, bias=False),
        nn.BatchNorm2d(channel),
        nn.ReLU(inplace=True)
            )

        self.gamma = nn.Parameter(torch.zeros(1))
        self.relu=nn.ReLU(inplace=True)
        self.norm = nn.Sigmoid()
       
    def forward(self, x):
        size = x.shape[2:]
        # print(x.shape)
        avg = self.pool(x)
        avg = F.upsample(avg, size=size, mode='bilinear')
        # print(avg.shape)
         
        _res6 = self.convs1(x)
        _res6 = self.bn[0](_res6)
        _res6 = self.relu(_res6)
        _res6 = self.conv3x3(_res6)
        _res6 = self.bn[0](_res6)
        _res6 = self.relu(_res6)
        res6 =self.convs0(_res6)
        res6 = self.bn[0](res6)
        res6 = self.relu(res6)
        
        _res12 = F.conv2d(x,self.convs1.weight,stride=1, padding=0)
        _res12 = self.bn[1](_res12)
        _res12 = self.relu(_res12)
        _res12 = F.conv2d(x,self.conv3x3.weight,padding=4,dilation=4)#share weight
        _res12 = self.bn[1](_res12)
        _res12 = self.relu(_res12)
        res12 = F.conv2d(_res12, self.convs0.weight, stride=1, padding=0)
        res12 = self.bn[1](res12)
        res12 = self.relu(res12)


        _res18 = F.conv2d(x,self.convs1.weight,stride=1, padding=0)          
        _res18 = self.bn[2](_res18)
        _res18 = self.relu(_res18)
        _res18 = F.conv2d(x,self.conv3x3.weight,padding=6,dilation=6)#share weight
        _res18 = self.bn[2](_res18)
        _res18 = self.relu(_res18)
        res18 = F.conv2d(_res18, self.convs0.weight, stride=1, padding=0)
        res18 = self.bn[2](res18)
        res18 = self.relu(res18)

        _res24 = F.conv2d(x,self.convs1.weight,stride=1, padding=0)
        _res24 = self.bn[3](_res24)
        _res24 = self.relu(_res24)
        _res24 = F.conv2d(x,self.conv3x3.weight,padding=8,dilation=8)#share weight
        _res24 = self.bn[3](_res24)
        _res24 = self.relu(_res24)
        res24 = F.conv2d(_res24, self.convs0.weight, stride=1, padding=0)
        res24 = self.bn[3](res24)
        res24 = self.relu(res24)


        U = torch.cat((avg, res6, res12, res18, res24), 1)
        out = self.Conv1x1(U)
        out =self.norm(out)
        out = U*out
        out = self.gamma*U+(1-self.gamma)*out
        out=self.fuse(out)

        # out=self.gamma*out+ (1-self.gamma)*x
        # out = out+ x
        return out
     










class double_conv(nn.Module):
    '''(conv => BN => ReLU) * 2'''

    def __init__(self, in_ch, out_ch, reduction=16):
        super(double_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU()
        )
        

        self.channel_conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(out_ch)
        )

    def forward(self, x):
        residual = x
        x = self.conv(x)
        # x = self.se(x)

        if residual.shape[1] != x.shape[1]:
            residual = self.channel_conv(residual)
        x += residual
        return x

class inconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(inconv, self).__init__()
        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x):
        x = self.conv(x)
        return x


# class down(nn.Module):
#     def __init__(self, in_ch, out_ch):
#         super(down, self).__init__()
#         self.mpconv = nn.Sequential(
#             nn.MaxPool2d(2),
#             double_conv(in_ch, out_ch)
#         )

#     def forward(self, x):
#         x = self.mpconv(x)
#         return x


class up(nn.Module):
    def __init__(self, in_ch, out_ch, bilinear=True):
        super(up, self).__init__()

        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_ch // 2, in_ch // 2, 2, stride=2)

        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)

        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, (diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2))

        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x


class outconv(nn.Module):
    def __init__(self, in_ch, out_ch, dropout=False, rate=0.1):
        super(outconv, self).__init__()
        self.dropout = dropout
        if dropout:
            print('dropout', rate)
            self.dp = nn.Dropout2d(rate)
        self.conv = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x):
        if self.dropout:
            x = self.dp(x)
        x = self.conv(x)
        return x
# class DACblock(nn.Module):
#     def __init__(self, channel):
#         super(DACblock, self).__init__()
#         self.dilate1 = nn.Conv2d(channel, channel, kernel_size=3, dilation=1, padding=1)
#         self.dilate2 = nn.Conv2d(channel, channel, kernel_size=3, dilation=3, padding=3)
#         self.dilate3 = nn.Conv2d(channel, channel, kernel_size=3, dilation=5, padding=5)
#         self.conv1x1 = nn.Conv2d(channel, channel, kernel_size=1, dilation=1, padding=0)
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
#                 if m.bias is not None:
#                     m.bias.data.zero_()

#     def forward(self, x):
#         dilate1_out = nonlinearity(self.dilate1(x))
#         dilate2_out = nonlinearity(self.conv1x1(self.dilate2(x)))
#         dilate3_out = nonlinearity(self.conv1x1(self.dilate2(self.dilate1(x))))
#         dilate4_out = nonlinearity(self.conv1x1(self.dilate3(self.dilate2(self.dilate1(x)))))
#         out = x + dilate1_out + dilate2_out + dilate3_out + dilate4_out
#         return out
        
class DecoderBlock(nn.Module):
    def __init__(self, in_channels, n_filters):
        super(DecoderBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, in_channels // 4, 1)
        self.norm1 = nn.BatchNorm2d(in_channels // 4)
        self.relu1 = nonlinearity

        self.deconv2 = nn.ConvTranspose2d(in_channels // 4, in_channels // 4, 3, stride=2, padding=1, output_padding=1)
        self.norm2 = nn.BatchNorm2d(in_channels // 4)
        self.relu2 = nonlinearity

        self.conv3 = nn.Conv2d(in_channels // 4, n_filters, 1)
        self.norm3 = nn.BatchNorm2d(n_filters)
        self.relu3 = nonlinearity

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)
        x = self.deconv2(x)
        x = self.norm2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.norm3(x)
        x = self.relu3(x)
        return x




class VGG19UNet(nn.Module):
    def __init__(self, n_channels, n_classes, deep_supervision=False,num_filters=64, dropout=False, rate=0.1, bn=False):
        super(VGG19UNet, self).__init__()
        self.deep_supervision = deep_supervision
        self.vgg_features = torchvision.models.vgg19(pretrained=True).features
        self.vgg_features[0]=nn.Conv2d(n_channels, 64, kernel_size=3, padding=1)
        # print(self.vgg_features)
        
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()
        self.inc = self.vgg_features[:4]
        # ##########
        # self.vgg_features[0]=nn.Conv2d(1, 64, kernel_size=3, padding=1)
        # #######
        # self.inc1= self.vgg_features[:4]
      
        self.down1 = self.vgg_features[4:9]
        self.down2 = self.vgg_features[9:18]
        self.down3 = self.vgg_features[18:27]
        self.down4 = self.vgg_features[27:36]
        # self.wassp = WASP(512)
        # self.wassp = ASPP(512,512)
        self.wassp=SAPP(512)      #lastest
        self.up1 = up(1024, 256)
        self.up2 = up(512, 128)
        self.up3 = up(256, 64)
        self.up4 = up(128, 64)
        self.outc = outconv(64+64, n_classes, dropout, rate)
        self.dsoutc4 = outconv(256+256, n_classes)
        self.dsoutc3 = outconv(128+128, n_classes)
        self.dsoutc2 = outconv(64+64, n_classes)
        self.dsoutc1 = outconv(64, n_classes)
        self.dsoutc5 = outconv(512+512, n_classes)

        # self.fuout =outconv(5, n_classes)
        
        
        #boundray stream

        self.cw = nn.Conv2d(2, 1, kernel_size=1, padding=0, bias=False)   
        self.expand = nn.Sequential(nn.Conv2d(1, num_filters, kernel_size=1),
                                    Norm2d(num_filters),
                                    nn.ReLU(inplace=True))

        self.expand1 = nn.Sequential(nn.Conv2d(1, 64, kernel_size=1),
                                    Norm2d(64),
                                    nn.ReLU(inplace=True))
        self.expand2 = nn.Sequential(nn.Conv2d(1, 128, kernel_size=1),
                                    Norm2d(128),
                                    nn.ReLU(inplace=True)) 
        self.expand3 = nn.Sequential(nn.Conv2d(1, 256, kernel_size=1),
                                    Norm2d(256),
                                    nn.ReLU(inplace=True))
        self.expand4 = nn.Sequential(nn.Conv2d(1, 512, kernel_size=1),
                                    Norm2d(512),
                                    nn.ReLU(inplace=True))                                                                                   


        self.gate1 = gsc.GatedSpatialConv2d(32, 32)
        self.gate2 = gsc.GatedSpatialConv2d(16, 16)
        self.gate3 = gsc.GatedSpatialConv2d(8, 8)

        self.c3 = nn.Conv2d(128, 1, kernel_size=1)
        self.c4 = nn.Conv2d(256, 1, kernel_size=1)
        self.c5 = nn.Conv2d(512, 1, kernel_size=1)

        self.d0 = nn.Conv2d(64, 64, kernel_size=1)
        self.res1 = ResBlock(64, 64)
        self.d1 = nn.Conv2d(64, 32, kernel_size=1)
        self.res2 = ResBlock(32, 32)
        self.d2 = nn.Conv2d(32, 16, kernel_size=1)
        self.res3 = ResBlock(16, 16)
        self.d3 = nn.Conv2d(16, 8, kernel_size=1)
        self.fuse = nn.Conv2d(8, 1, kernel_size=1, padding=0, bias=False)

    def forward(self, x):
        x_size = x.size()
        x1 = self.inc(x)
        x2 = self.down1(x1)
        # print(x2.size())
        x3 = self.down2(x2)
        # print(x3.size())
        x4 = self.down3(x3)
        # print(x4.size())
        x5 = self.down4(x4)
        # print(x5.size())
        x55=self.wassp(x5)
       
        # x55=self.spp(x55)
        # x55= self.DilatedBottleneck(x5)
      ###shape stream

        ss = F.interpolate(self.d0(x1), x_size[2:],
                            mode='bilinear', align_corners=True)
        ss = self.res1(ss)
        c3 = F.interpolate(self.c3(x2), x_size[2:],
                            mode='bilinear', align_corners=True)
        ss = self.d1(ss)
        ss1 = self.gate1(ss, c3)
        # print("***********")
        # print(ss1.shape)
        ss = self.res2(ss1)
        ss = self.d2(ss)
        c4 = F.interpolate(self.c4(x3), x_size[2:],
                            mode='bilinear', align_corners=True)
        ss2 = self.gate2(ss, c4)
        ss = self.res3(ss2)
        ss = self.d3(ss)
        c5 = F.interpolate(self.c5(x4), x_size[2:],
                            mode='bilinear', align_corners=True)
        ss3 = self.gate3(ss, c5)
        ss = self.fuse(ss3)
        ss = F.interpolate(ss, x_size[2:], mode='bilinear', align_corners=True)
        edge_out = self.sigmoid(ss)
        

        ### Canny Edge
        im_arr = np.mean(x.cpu().numpy(), axis=1).astype(np.uint8)
        canny = np.zeros((x_size[0], 1 , x_size[2], x_size[3]))     
        for i in range(x_size[0]):
            canny[i] = cv2.Canny(im_arr[i], 10, 100)
        canny = torch.from_numpy(canny).cuda().float()   #注意.cuda()
        ### End Canny Edge

        cat = torch.cat([edge_out, canny], dim=1)
        acts = self.cw(cat)
        acts = self.sigmoid(acts)
       
        edge = self.expand(acts)

        edge1=self.expand1(acts)
        edge2=self.expand2(acts)
        edge3=self.expand3(acts)
        edge4=self.expand4(acts)
        
        x44 = self.up1(x55, x4)
        x33 = self.up2(x44, x3)      
        x22 = self.up3(x33, x2)
        x11 = self.up4(x22, x1)
       
        x11_con = torch.cat([x11, edge], dim=1)  #fusion
        # print(x11_con.shape)
        x0 = self.outc(x11_con)


        # print(x1.shape)
        # print(x2.shape)
        # print(x3.shape)
        # print(x4.shape)
        # print(x5.shape)
        # print(x55.shape)
        # print(x44.shape)
        # print(x33.shape)
        # print(x22.shape)
        # print(x11.shape)
        # print(x0.shape)
        

        
        # x0 = self.sigmoid(x0)
        # #self_deep supvision
        # x_11=self.dsoutc2(x22)   # x2
        # x_12 =self.inc(x0)
        # x_13 =self.down1(x_12)
        # x_14=self.dsoutc3(x_13)   #x2
        # # 48
        # x_22=self.dsoutc3(x33)
        # x_23=self.inc(x_14)
        # x_24=self.down1(x_23)
        # x_25=self.dsoutc3(x_24)
        # # 24
        # x_33=self.dsoutc4(x44)
        # x_34=self.inc(x_25)
        # x_35=self.down1(x_34)
        # x_36=self.dsoutc3(x_35)
        # # 12
        # x_44=self.dsoutc5(x55)
        # x_45=self.inc(x_36)
        # x_46=self.down1(x_45)
        # x_47=self.dsoutc3(x_46)
# 闭环设计 closed-loop
         #branch2
        edge1 = F.interpolate(edge1, scale_factor=1/2, mode='bilinear')
        crop_1 = F.interpolate(x0, scale_factor=0.5, mode='bilinear')
        x_s= -1*(torch.sigmoid(crop_1)) + 1  #反转
        x22= x_s.expand(-1, 64, -1, -1).mul(x22)
        x_cat1=torch.cat([x22, edge1], dim=1)
        x_11=self.dsoutc2(x_cat1)
        x_11 = x_11+crop_1
    
        x_14 = F.interpolate(x_11, scale_factor=2, mode='bilinear')      
        # print(x_14.shape)
        #branch3
        edge2 = F.interpolate(edge2, scale_factor=1/4, mode='bilinear')
      
   
        crop_2 = F.interpolate(x_11, scale_factor=0.5, mode='bilinear')
        x= -1*(torch.sigmoid(crop_2)) + 1  #反转
        x33= x.expand(-1, 128, -1, -1).mul(x33)
        x_cat2=torch.cat([x33, edge2], dim=1)
        x_22=self.dsoutc3(x_cat2)

        x_12 = x_22 +crop_2
        x_25 = F.interpolate(x_12, scale_factor=4, mode='bilinear')
       
        #branch4
        edge3 = F.interpolate(edge3, scale_factor=1/8, mode='bilinear')
        
        crop_3 = F.interpolate(x_12, scale_factor=0.5, mode='bilinear')
        x= -1*(torch.sigmoid(crop_3)) + 1  #反转
        x44= x.expand(-1, 256, -1, -1).mul(x44)
        x_cat3=torch.cat([x44, edge3], dim=1)
        x_33=self.dsoutc4(x_cat3)
        x_13 = x_33+crop_3
        x_36 = F.interpolate(x_13, scale_factor=8, mode='bilinear')
    
        #branch5
        edge4 = F.interpolate(edge4, scale_factor=1/16, mode='bilinear')
       
        crop_4 = F.interpolate(x_13, scale_factor=0.5, mode='bilinear')
        x= -1*(torch.sigmoid(crop_4)) + 1  #反转
        x55= x.expand(-1, 512, -1, -1).mul(x55)   #x55
        x_cat3=torch.cat([x55, edge4], dim=1)  
        x_44=self.dsoutc5(x_cat3)
        x_15 = x_44 + crop_4
        x_47 = F.interpolate(x_15, scale_factor=16, mode='bilinear')
        print(x_47.shape)


        # x_fu = torch.cat([x0, x_14, x_25, x_36, x_47], dim=1)
        # x_fu = self.fuout(x_fu)


    
    # # #前馈设计I

    #     #branch5
    #     edge4 = F.interpolate(edge4, scale_factor=1/16, mode='bilinear')
    #     x_cat3=torch.cat([x55, edge4], dim=1)  
    #     x_44=self.dsoutc5(x_cat3)
    #     x_47 = F.interpolate(x_44, scale_factor=16, mode='bilinear')
    #     #branch4
    #     edge3 = F.interpolate(edge3, scale_factor=1/8, mode='bilinear')
    #     x_cat3=torch.cat([x44, edge3], dim=1)
    #     x_33=self.dsoutc4(x_cat3)
    #     x_36 = F.interpolate(x_33, scale_factor=8, mode='bilinear')
    #     #branch3
    #     edge2 = F.interpolate(edge2, scale_factor=1/4, mode='bilinear')
    #     x_cat2=torch.cat([x33, edge2], dim=1)
    #     x_22=self.dsoutc3(x_cat2)
    #     crop_2 = F.interpolate(x_33, scale_factor=2, mode='bilinear')
    #     x_12 = x_22 +crop_2
    #     x_25 = F.interpolate(x_12, scale_factor=4, mode='bilinear')
    #     #branch2
    #     edge1 = F.interpolate(edge1, scale_factor=1/2, mode='bilinear') 
    #     x_cat1=torch.cat([x22, edge1], dim=1)
    #     x_11=self.dsoutc2(x_cat1)
    #     crop_1 = F.interpolate(x_12, scale_factor=2, mode='bilinear')
    #     x_11 = x_11+crop_1
    #     x_14 = F.interpolate(x_11, scale_factor=2, mode='bilinear')
    #     #branch1
    #     x11_con = torch.cat([x11, edge], dim=1)  #fusion
    #     # print(x11_con.shape)
    #     x0 = self.outc(x11_con)
    #     crop_0 = F.interpolate(x_11, scale_factor=2, mode='bilinear')
    #     x_fu = x0+crop_0

        if self.deep_supervision and self.training:
            x11 = F.interpolate(self.dsoutc1(x11), x0.shape[2:], mode='bilinear')
            x22 = F.interpolate(self.dsoutc2(x22), x0.shape[2:], mode='bilinear')
            x33 = F.interpolate(self.dsoutc3(x33), x0.shape[2:], mode='bilinear')
            x44 = F.interpolate(self.dsoutc4(x44), x0.shape[2:], mode='bilinear')

            return x0, x11, x22, x33, x44
        else:

            return  x0, x_14, x_25, x_36, x_47, ss


class VGG19UNet2(nn.Module):
    def __init__(self, n_channels, n_classes, deep_supervision=False,num_filters=64, dropout=False, rate=0.1, bn=False):
        super(VGG19UNet2, self).__init__()
        self.deep_supervision = deep_supervision
        self.vgg_features = torchvision.models.vgg19(pretrained=True).features
        self.vgg_features[0]=nn.Conv2d(n_channels, 64, kernel_size=3, padding=1)
        
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()
        self.inc = self.vgg_features[:4]
        # ##########
        # self.vgg_features[0]=nn.Conv2d(1, 64, kernel_size=3, padding=1)
        # #######
        # self.inc1= self.vgg_features[:4]
      
        self.down1 = self.vgg_features[4:9]
        self.down2 = self.vgg_features[9:18]
        self.down3 = self.vgg_features[18:27]
        self.down4 = self.vgg_features[27:36]
        self.wassp = ASPP(512,512)
        # self.wassp = WASP(512)
        self.up1 = up(1024, 256)
        self.up2 = up(512, 128)
        self.up3 = up(256, 64)
        self.up4 = up(128, 64)
        self.outc = outconv(64+64, n_classes, dropout, rate)
        self.dsoutc4 = outconv(256+256, n_classes)
        self.dsoutc3 = outconv(128+128, n_classes)
        self.dsoutc2 = outconv(64+64, n_classes)
        self.dsoutc1 = outconv(64, n_classes)
        self.dsoutc5 = outconv(512+512, n_classes)
        
        
        #boundray stream

        self.cw = nn.Conv2d(2, 1, kernel_size=1, padding=0, bias=False)     
        self.expand = nn.Sequential(nn.Conv2d(1, num_filters, kernel_size=1),
                                    Norm2d(num_filters),
                                    nn.ReLU(inplace=True))

        self.expand1 = nn.Sequential(nn.Conv2d(1, 64, kernel_size=1),
                                    Norm2d(64),
                                    nn.ReLU(inplace=True))
        self.expand2 = nn.Sequential(nn.Conv2d(1, 128, kernel_size=1),
                                    Norm2d(128),
                                    nn.ReLU(inplace=True)) 
        self.expand3 = nn.Sequential(nn.Conv2d(1, 256, kernel_size=1),
                                    Norm2d(256),
                                    nn.ReLU(inplace=True))
        self.expand4 = nn.Sequential(nn.Conv2d(1, 512, kernel_size=1),
                                    Norm2d(512),
                                    nn.ReLU(inplace=True))                                                                                   


        self.gate1 = gsc.GatedSpatialConv2d(32, 32)
        self.gate2 = gsc.GatedSpatialConv2d(16, 16)
        self.gate3 = gsc.GatedSpatialConv2d(8, 8)

        self.c3 = nn.Conv2d(128, 1, kernel_size=1)
        self.c4 = nn.Conv2d(256, 1, kernel_size=1)
        self.c5 = nn.Conv2d(512, 1, kernel_size=1)

        self.d0 = nn.Conv2d(64, 64, kernel_size=1)
        self.res1 = ResBlock(64, 64)
        self.d1 = nn.Conv2d(64, 32, kernel_size=1)
        self.res2 = ResBlock(32, 32)
        self.d2 = nn.Conv2d(32, 16, kernel_size=1)
        self.res3 = ResBlock(16, 16)
        self.d3 = nn.Conv2d(16, 8, kernel_size=1)
        self.fuse = nn.Conv2d(8, 1, kernel_size=1, padding=0, bias=False)

    def forward(self, x):
        x_size = x.size()
        x1 = self.inc(x)
        x2 = self.down1(x1)
        # print(x2.size())
        x3 = self.down2(x2)
        # print(x3.size())
        x4 = self.down3(x3)
        # print(x4.size())
        x5 = self.down4(x4)
        # print(x5.size())
        x55=self.wassp(x5)
       
        # x55=self.spp(x55)
        # x55= self.DilatedBottleneck(x5)
      ###shape stream

        ss = F.interpolate(self.d0(x1), x_size[2:],
                            mode='bilinear', align_corners=True)
        ss = self.res1(ss)
        c3 = F.interpolate(self.c3(x2), x_size[2:],
                            mode='bilinear', align_corners=True)
        ss = self.d1(ss)
        ss1 = self.gate1(ss, c3)
        ss = self.res2(ss1)
        ss = self.d2(ss)
        c4 = F.interpolate(self.c4(x3), x_size[2:],
                            mode='bilinear', align_corners=True)
        ss2 = self.gate2(ss, c4)
        ss = self.res3(ss2)
        ss = self.d3(ss)
        c5 = F.interpolate(self.c5(x4), x_size[2:],
                            mode='bilinear', align_corners=True)
        ss3 = self.gate3(ss, c5)
        ss = self.fuse(ss3)
        ss = F.interpolate(ss, x_size[2:], mode='bilinear', align_corners=True)
        edge_out = self.sigmoid(ss)
        

        ### Canny Edge
        im_arr = np.mean(x.cpu().numpy(), axis=1).astype(np.uint8)
        canny = np.zeros((x_size[0], 1, x_size[2], x_size[3]))
        for i in range(x_size[0]):
            canny[i] = cv2.Canny(im_arr[i], 10, 100)
        canny = torch.from_numpy(canny).cuda().float()   #注意cuda()
        ### End Canny Edge

        cat = torch.cat([edge_out, canny], dim=1)
        acts = self.cw(cat)
        acts = self.sigmoid(acts)
       
        edge = self.expand(acts)

        edge1=self.expand1(acts)
        edge2=self.expand2(acts)
        edge3=self.expand3(acts)
        edge4=self.expand4(acts)

     

    
        
        x44 = self.up1(x5, x4)
        x33 = self.up2(x44, x3)      
        x22 = self.up3(x33, x2)
        x11 = self.up4(x22, x1)
       
        x11_con = torch.cat([x11, edge], dim=1)  #fusion
        # print(x11_con.shape)
        x0 = self.outc(x11_con)


        # print(x1.shape)
        # print(x2.shape)
        # print(x3.shape)
        # print(x4.shape)
        # print(x5.shape)
        # print(x55.shape)
        # print(x44.shape)
        # print(x33.shape)
        # print(x22.shape)
        # print(x11.shape)
        # print(x0.shape)
        

        
        # x0 = self.sigmoid(x0)
        # #self_deep supvision
        # x_11=self.dsoutc2(x22)   # x2
        # x_12 =self.inc(x0)
        # x_13 =self.down1(x_12)
        # x_14=self.dsoutc3(x_13)   #x2
        # # 48
        # x_22=self.dsoutc3(x33)
        # x_23=self.inc(x_14)
        # x_24=self.down1(x_23)
        # x_25=self.dsoutc3(x_24)
        # # 24
        # x_33=self.dsoutc4(x44)
        # x_34=self.inc(x_25)
        # x_35=self.down1(x_34)
        # x_36=self.dsoutc3(x_35)
        # # 12
        # x_44=self.dsoutc5(x55)
        # x_45=self.inc(x_36)
        # x_46=self.down1(x_45)
        # x_47=self.dsoutc3(x_46)
# 闭环设计
         #branch2
        edge1 = F.interpolate(edge1, scale_factor=1/2, mode='bilinear')
       
        
        x_cat1=torch.cat([x22, edge1], dim=1)
        
        x_11=self.dsoutc2(x_cat1)
   
        crop_1 = F.interpolate(x0, scale_factor=0.5, mode='bilinear')
      
        x_11 = x_11+crop_1
    
        x_14 = F.interpolate(x_11, scale_factor=2, mode='bilinear')
        # print(x_14.shape)
        #branch3
        edge2 = F.interpolate(edge2, scale_factor=1/4, mode='bilinear')
        x_cat2=torch.cat([x33, edge2], dim=1)
        x_22=self.dsoutc3(x_cat2)
   
        crop_2 = F.interpolate(x_11, scale_factor=0.5, mode='bilinear')
       
        x_12 = x_22 +crop_2
        x_25 = F.interpolate(x_12, scale_factor=4, mode='bilinear')
       
        #branch4
        edge3 = F.interpolate(edge3, scale_factor=1/8, mode='bilinear')
        x_cat3=torch.cat([x44, edge3], dim=1)
        x_33=self.dsoutc4(x_cat3)
        crop_3 = F.interpolate(x_12, scale_factor=0.5, mode='bilinear')
        x_13 = x_33+crop_3
        x_36 = F.interpolate(x_13, scale_factor=8, mode='bilinear')
    
        #branch5
        edge4 = F.interpolate(edge4, scale_factor=1/16, mode='bilinear')
        x_cat3=torch.cat([x5, edge4], dim=1)  
        x_44=self.dsoutc5(x_cat3)
        crop_3 = F.interpolate(x_13, scale_factor=0.5, mode='bilinear')
        x_15 = x_44 + crop_3
        x_47 = F.interpolate(x_15, scale_factor=16, mode='bilinear')

        #加另一个分支
        



        # x_fu = (x0+x_14 + x_25 + x_36 + x_47)
    
    # #前馈设计I

    #     #branch5
    #     edge4 = F.interpolate(edge4, scale_factor=1/16, mode='bilinear')
    #     x_cat3=torch.cat([x55, edge4], dim=1)  
    #     x_44=self.dsoutc5(x_cat3)
    #     x_47 = F.interpolate(x_44, scale_factor=16, mode='bilinear')
        # #branch4
        # edge3 = F.interpolate(edge3, scale_factor=1/8, mode='bilinear')
        # x_cat3=torch.cat([x44, edge3], dim=1)
        # x_33=self.dsoutc4(x_cat3)
        # x_36 = F.interpolate(x_33, scale_factor=8, mode='bilinear')
        # #branch3
        # edge2 = F.interpolate(edge2, scale_factor=1/4, mode='bilinear')
        # x_cat2=torch.cat([x33, edge2], dim=1)
        # x_22=self.dsoutc3(x_cat2)
        # crop_2 = F.interpolate(x_33, scale_factor=2, mode='bilinear')
        # x_12 = x_22 +crop_2
        # x_25 = F.interpolate(x_12, scale_factor=4, mode='bilinear')
        # #branch2
        # edge1 = F.interpolate(edge1, scale_factor=1/2, mode='bilinear') 
        # x_cat1=torch.cat([x22, edge1], dim=1)
        # x_11=self.dsoutc2(x_cat1)
        # crop_1 = F.interpolate(x_12, scale_factor=2, mode='bilinear')
        # x_11 = x_11+crop_1
        # x_14 = F.interpolate(x_11, scale_factor=2, mode='bilinear')
        # #branch1
        # x11_con = torch.cat([x11, edge], dim=1)  #fusion
        # # print(x11_con.shape)
        # x0 = self.outc(x11_con)
        # crop_0 = F.interpolate(x_11, scale_factor=2, mode='bilinear')
        # x_fu = x0+crop_0

        

      
    
  
    
    
    


       








        if self.deep_supervision and self.training:
            x11 = F.interpolate(self.dsoutc1(x11), x0.shape[2:], mode='bilinear')
            x22 = F.interpolate(self.dsoutc2(x22), x0.shape[2:], mode='bilinear')
            x33 = F.interpolate(self.dsoutc3(x33), x0.shape[2:], mode='bilinear')
            x44 = F.interpolate(self.dsoutc4(x44), x0.shape[2:], mode='bilinear')

            return x0, x11, x22, x33, x44
        else:
           
            # x0 = self.sigmoid(x0)
            # ss = self.sigmoid(ss)
           
            # x_14 = self.sigmoid(x_14)
            
            # x_25 = self.sigmoid(x_25)
            
            # x_36 = self.sigmoid(x_36)
           
            # x_47 = self.sigmoid(x_47)

            # # x11 = self.sigmoid(x11)
            # # x22 = self.sigmoid(x22)
            # # x33 = self.sigmoid(x33)
            # # x44 = self.sigmoid(x44) 
            # ss1=self.sigmoid(ss1)
            # ss2=self.sigmoid(ss2)
            # ss3=self.sigmoid(ss2)
            # edge =self.sigmoid(edge)

            return  x0, x_14, x_25, x_36, x_47, ss





class VGG19UNet_without_boudary(nn.Module):
    def __init__(self, n_channels, n_classes, deep_supervision=False,num_filters=32, dropout=False, rate=0.1, bn=False, pretrain=True):
        super(VGG19UNet_without_boudary, self).__init__()
        self.deep_supervision = deep_supervision
        self.vgg_features = torchvision.models.vgg19(pretrained=True).features
        self.vgg_features[0]=nn.Conv2d(n_channels, 64, kernel_size=3, padding=1)
        
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()
        self.inc = self.vgg_features[:4]
 
        self.down1 = self.vgg_features[4:9]
        self.down2 = self.vgg_features[9:18]
        self.down3 = self.vgg_features[18:27]
        self.down4 = self.vgg_features[27:36]
      
        self.wassp=SAPP(512) 
        self.up1 = up(1024, 256)
        self.up2 = up(512, 128)
        self.up3 = up(256, 64)
        self.up4 = up(128, 64)
        self.outc = outconv(64, n_classes, dropout, rate)
        self.dsoutc4 = outconv(256, n_classes)
        self.dsoutc3 = outconv(128, n_classes)
        self.dsoutc2 = outconv(64, n_classes)
        self.dsoutc1 = outconv(64, n_classes)
        self.dsoutc5 = outconv(512, n_classes)
       
    def forward(self, x):
        x_size = x.size()
        x1 = self.inc(x)
        x2 = self.down1(x1)
        # print(x2.size())
        x3 = self.down2(x2)
        # print(x3.size())
        x4 = self.down3(x3)
        # print(x4.size())
        x5 = self.down4(x4)
        # print(x5.size())
        x55=self.wassp(x5)
        x44 = self.up1(x55, x4)
        x33 = self.up2(x44, x3)      
        x22 = self.up3(x33, x2)
        x11 = self.up4(x22, x1)
        # print(x11_con.shape)
        x0 = self.outc(x11)
        if self.deep_supervision and self.training:
            x11 = F.interpolate(self.dsoutc1(x11), x0.shape[2:], mode='bilinear')
            x22 = F.interpolate(self.dsoutc2(x22), x0.shape[2:], mode='bilinear')
            x33 = F.interpolate(self.dsoutc3(x33), x0.shape[2:], mode='bilinear')
            x44 = F.interpolate(self.dsoutc4(x44), x0.shape[2:], mode='bilinear')

            return x0, x11, x22, x33, x44
        else:
            return x0


class VGGUNet(nn.Module):
    def __init__(self, n_channels, n_classes, deep_supervision=False,num_filters=32, dropout=False, rate=0.1, bn=False, pretrain=True):
        super(VGGUNet, self).__init__()
        self.deep_supervision = deep_supervision
        self.vgg_features = torchvision.models.vgg19(pretrained=False).features
        self.vgg_features[0]=nn.Conv2d( n_channels, 64, kernel_size=3, padding=1)
        
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()
        self.inc = self.vgg_features[:4]
 
        self.down1 = self.vgg_features[4:9]
        self.down2 = self.vgg_features[9:18]
        self.down3 = self.vgg_features[18:27]
        self.down4 = self.vgg_features[27:36]
        self.wassp = SAPP(512)
        self.up1 = up(1024, 256)
        self.up2 = up(512, 128)
        self.up3 = up(256, 64)
        self.up4 = up(128, 64)
        self.outc = outconv(64, n_classes, dropout, rate)
        self.dsoutc4 = outconv(256, n_classes)
        self.dsoutc3 = outconv(128, n_classes)
        self.dsoutc2 = outconv(64, n_classes)
        self.dsoutc1 = outconv(64, n_classes)
        self.dsoutc5 = outconv(512, n_classes)
       
    def forward(self, x):
        x_size = x.size()
        x1 = self.inc(x)
        x2 = self.down1(x1)
        # print(x2.size())
        x3 = self.down2(x2)
        # print(x3.size())
        x4 = self.down3(x3)
        # print(x4.size())
        x5 = self.down4(x4)
        # print(x5.size())
        x55=self.wassp(x5)
        x44 = self.up1(x55, x4)
        x33 = self.up2(x44, x3)      
        x22 = self.up3(x33, x2)
        x11 = self.up4(x22, x1)
        # print(x11_con.shape)
        x0 = self.outc(x11)
        # x0 = self.sigmoid(x0)
        # #self_deep supvision
        if self.deep_supervision and self.training:
            x11 = F.interpolate(self.dsoutc1(x11), x0.shape[2:], mode='bilinear')
            x22 = F.interpolate(self.dsoutc2(x22), x0.shape[2:], mode='bilinear')
            x33 = F.interpolate(self.dsoutc3(x33), x0.shape[2:], mode='bilinear')
            x44 = F.interpolate(self.dsoutc4(x44), x0.shape[2:], mode='bilinear')

            return x0, x11, x22, x33, x44
        else:
            return x0, x11, x22, x33, x44

            #  x_11, x_14, x_22, x_25, x_33, x_36, x_44, x_47