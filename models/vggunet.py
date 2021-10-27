import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings
import torchvision

warnings.filterwarnings(action='ignore')





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


class down(nn.Module):
	def __init__(self, in_ch, out_ch):
		super(down, self).__init__()
		self.mpconv = nn.Sequential(
			nn.MaxPool2d(2),
			double_conv(in_ch, out_ch)
		)

	def forward(self, x):
		x = self.mpconv(x)
		return x


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

class WASP(nn.Module):
    def __init__(self, channel, ):
        super(WASP, self).__init__()
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.conv3x3=nn.Conv2d(channel, channel,dilation=1,kernel_size=3, padding=1)
        # self.atrous_block6  = nn.Conv2d(channel, channel, kernel_size=3, padding=6, dilation=6)
        # self.atrous_block12 = nn.Conv2d(channel, channel, kernel_size=3, padding=12, dilation=12)
        # self.atrous_block18 = nn.Conv2d(channel, channel, kernel_size=3, padding=18, dilation=18)
        # self.atrous_block24 = nn.Conv2d(channel, channel, kernel_size=3, padding=24, dilation=24)
       
        # for i in range(4):
        #     convs.append(nn.Conv2d(channel, channel, kernel_size=1, padding=0))
        
        self.convs0 =nn.Conv2d(channel, channel, kernel_size=1, padding=0)
        self.fuse = nn.Conv2d(channel * 5, channel, kernel_size=1, padding=0)
       
    def forward(self, x):
        size = x.shape[2:]
        # print(x.shape)
        avg = self.pool(x)
        avg = F.upsample(avg, size=size, mode='bilinear')
        # print(avg.shape)

        _res6 = self.conv3x3(x)
        res6 = F.conv2d(_res6 , self.convs0.weight,stride=1, padding=0)

        _res12 = F.conv2d(x,self.conv3x3.weight,padding=2,dilation=2)#share weight
        res12 = F.conv2d(_res12, self.convs0.weight, stride=1, padding=0)

        _res18 = F.conv2d(x,self.conv3x3.weight,padding=4,dilation=4)#share weight
        res18 = F.conv2d(_res18, self.convs0.weight, stride=1, padding=0)

        _res24 = F.conv2d(x,self.conv3x3.weight,padding=8,dilation=8)#share weight
        res24 = F.conv2d(_res24, self.convs0.weight, stride=1, padding=0)
        # print(res24.shape)


        out = torch.cat((avg, res6, res12, res18, res24), 1)
        out = self.fuse(out)
        out=x+out
 
    
        return out
     






class VGGUNet(nn.Module):
    def __init__(self, n_channels, n_classes, deep_supervision=False,num_filters=32, dropout=False, rate=0.1, bn=False, pretrain=True):
        super(VGGUNet, self).__init__()
        self.deep_supervision = deep_supervision
        self.vgg_features = torchvision.models.vgg19(pretrained=False).features
        self.vgg_features[0]=nn.Conv2d(3, 64, kernel_size=3, padding=1)
        
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()
        self.inc = self.vgg_features[:4]
        ##########
        self.vgg_features[0]=nn.Conv2d(1, 64, kernel_size=3, padding=1)
        #######
        self.inc1= self.vgg_features[:4]
      
        self.down1 = self.vgg_features[4:9]
        self.down2 = self.vgg_features[9:18]
        self.down3 = self.vgg_features[18:27]
        self.down4 = self.vgg_features[27:36]
        self.wassp = WASP(512)
        self.up1 = up(1028, 256)
        self.up2 = up(512, 128)
        self.up3 = up(256, 64)
        self.up4 = up(128, 64)
        self.outc = outconv(64, n_classes, dropout, rate)
        self.dsoutc4 = outconv(256, n_classes)
        self.dsoutc3 = outconv(128, n_classes)
        self.dsoutc2 = outconv(64, n_classes)
        self.dsoutc1 = outconv(64, n_classes)
        self.dsoutc5 = outconv(516, n_classes)
        # self.Res2Net_Seg = Res2Net_Seg(512, stride=1, scale=4, basewidth=28)
        # self.sigmoid = torch.nn.Sigmoid()
       
        

        # self.DilatedBottleneck = DilatedBottleneck(512, 512, stride=1, dilations = [1, 2, 4, 8])
        self.spp = SPPblock(512)

    def forward(self, x):
        
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
        x55=self.spp(x55)
        # x55= self.DilatedBottleneck(x5)
      

        # x55=self.spp(x55)
        # x55 = self.Res2Net_Seg(x5)
        x44 = self.up1(x55, x4)
        x33 = self.up2(x44, x3)      
        x22 = self.up3(x33, x2)
        x11 = self.up4(x22, x1)
        #fusion
        # print(x11_con.shape)
        x0 = self.outc(x11)
        # x0 = self.sigmoid(x0)
        # #self_deep supvision
        # x_11=self.dsoutc2(x22)   # x2
        # x_12 =self.inc1(x0)
        # x_13 =self.down1(x_12)
        # x_14=self.dsoutc3(x_13)   #x2
        # # 48
        # x_22=self.dsoutc3(x33)
        # x_23=self.inc1(x_14)
        # x_24=self.down1(x_23)
        # x_25=self.dsoutc3(x_24)
        # # 24
        # x_33=self.dsoutc4(x44)
        # x_34=self.inc1(x_25)
        # x_35=self.down1(x_34)
        # x_36=self.dsoutc3(x_35)
        # # 12
        # x_44=self.dsoutc5(x55)
        # x_45=self.inc1(x_36)
        # x_46=self.down1(x_45)
        # x_47=self.dsoutc3(x_46)

        if self.deep_supervision and self.training:
            x11 = F.interpolate(self.dsoutc1(x11), x0.shape[2:], mode='bilinear')
            x22 = F.interpolate(self.dsoutc2(x22), x0.shape[2:], mode='bilinear')
            x33 = F.interpolate(self.dsoutc3(x33), x0.shape[2:], mode='bilinear')
            x44 = F.interpolate(self.dsoutc4(x44), x0.shape[2:], mode='bilinear')

            return x0, x11, x22, x33, x44
        else:
            return x0




if __name__ == '__main__':
	import os

	os.environ['CUDA_VISIBLE_DEVICES'] = '6'


	def weights_init(m):
		classname = m.__class__.__name__
		# print(classname)
		if classname.find('Conv') != -1:
			torch.nn.init.xavier_uniform_(m.weight.data)
			if m.bias is not None:
				torch.nn.init.constant_(m.bias.data, 0.0)


	# model = VGGUNet(3, 3, encoder='vgg11', deep_supervision=True).cuda()
	# model.apply(weights_init)

	# x = torch.randn((1, 3, 256, 256)).cuda()

	# for i in range(1000):
	# 	y0, y1, y2, y3, y4 = model(x)
	# 	print(y0.shape, y1.shape, y2.shape, y3.shape, y4.shape)
	# 	break
# print(model.weight)

# for name, param in model.named_parameters():
#     print(name, param.requires_grad)

# optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
# loss_func = torch.nn.CrossEntropyLoss()
#
# # model.apply(weights_init)
# # # model.load_state_dict(torch.load('./MODEL.pth'))
# model = model.cuda()
# print('load done')
# input()
#
# model.train()
# for i in range(100):
# 	x = torch.randn(1, 3, 256, 256).cuda()
# 	label = torch.rand(1, 256, 256).long().cuda()
# 	y = model(x)
# 	print(i)
#
# 	loss = loss_func(y, label)
# 	optimizer.zero_grad()
# 	loss.backward()
# 	optimizer.step()
#
# print('train done')
# input()
#
# with torch.no_grad():
# 	model.eval()
# 	for i in range(1000):
# 		x = torch.randn(1, 1, 256, 256).cuda()
# 		label = torch.rand(1, 256, 256).long().cuda()
# 		y = model(x)
# 		print(y.shape)
#
# 		# loss = loss_func(y, label)
# 		# optimizer.zero_grad()
# 		# loss.backward()
# 		# optimizer.step()

# input()

