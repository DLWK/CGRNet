import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings
warnings.filterwarnings(action='ignore')


class double_conv(nn.Module):
	'''(conv => BN => ReLU) * 2'''

	def __init__(self, in_ch, out_ch):
		super(double_conv, self).__init__()
		self.conv = nn.Sequential(
			nn.Conv2d(in_ch, out_ch, 3, padding=1),
			nn.BatchNorm2d(out_ch),
			nn.ReLU(inplace=True),
			nn.Conv2d(out_ch, out_ch, 3, padding=1),
			nn.BatchNorm2d(out_ch),
			nn.ReLU(inplace=True)
		)

	def forward(self, x):
		x = self.conv(x)
		return x

# class double_conv(nn.Module):
# 	'''(conv => BN => ReLU) * 2'''

# 	def __init__(self, in_ch, out_ch):
# 		super(double_conv, self).__init__()
# 		self.conv = nn.Sequential(
# 			DSConv3x3(in_ch, out_ch),
# 			# nn.BatchNorm2d(out_ch),
# 			# nn.ReLU(inplace=True),
# 			DSConv3x3(out_ch, out_ch)
# 			# nn.BatchNorm2d(out_ch),
# 			# nn.ReLU(inplace=True)
# 		)

# 	def forward(self, x):
# 		x = self.conv(x)
# 		return x


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

	def forward(self,x1, x2):
		x1 = self.up(x1)

		diffY = x2.size()[2] - x1.size()[2]
		diffX = x2.size()[3] - x1.size()[3]

		x1 = F.pad(x1, (diffX // 2, diffX - diffX // 2,
						diffY // 2, diffY - diffY // 2))

		x = torch.cat([x2, x1], dim=1)
		x = self.conv(x)
		return x


class outconv(nn.Module):
	def __init__(self, in_ch, out_ch):
		super(outconv, self).__init__()
		self.conv = nn.Conv2d(in_ch, out_ch, 1)

	def forward(self, x):
		x = self.conv(x)
		return x
#############################################################################################################
class convbnrelu(nn.Module):
    def __init__(self, in_channel, out_channel, k=3, s=1, p=1, g=1, d=1, bias=False, bn=True, relu=True):
        super(convbnrelu, self).__init__()
        conv = [nn.Conv2d(in_channel, out_channel, k, s, p, dilation=d, groups=g, bias=bias)]
        if bn:
            conv.append(nn.BatchNorm2d(out_channel))
        if relu:
            conv.append(nn.ReLU(inplace=True))
        self.conv = nn.Sequential(*conv)

    def forward(self, x):
        return self.conv(x)


class DSConv3x3(nn.Module):
    def __init__(self, in_channel, out_channel, stride=1, dilation=1, relu=True):
        super(DSConv3x3, self).__init__()
        self.conv = nn.Sequential(
                convbnrelu(in_channel, in_channel, k=3, s=stride, p=dilation, d=dilation, g=in_channel),
                convbnrelu(in_channel, out_channel, k=1, s=1, p=0, relu=relu)
                )

    def forward(self, x):
        return self.conv(x)


class DSConv5x5(nn.Module):
    def __init__(self, in_channel, out_channel, stride=1, dilation=1, relu=True):
        super(DSConv5x5, self).__init__()
        self.conv = nn.Sequential(
                convbnrelu(in_channel, in_channel, k=5, s=stride, p=2*dilation, d=dilation, g=in_channel),
                convbnrelu(in_channel, out_channel, k=1, s=1, p=0, relu=relu)
                )

    def forward(self, x):
        return self.conv(x)




class VAMM(nn.Module):
    def __init__(self, channel, dilation_level=[1,2,4,8], reduce_factor=4):
        super(VAMM, self).__init__()
        self.planes = channel
        self.dilation_level = dilation_level
        self.conv = DSConv3x3(channel, channel, stride=1)
        self.branches = nn.ModuleList([
                DSConv3x3(channel, channel, stride=1, dilation=d) for d in dilation_level
                ])
        ### ChannelGate
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc1 = convbnrelu(channel, channel, 1, 1, 0, bn=True, relu=True)
        self.fc2 = nn.Conv2d(channel, (len(self.dilation_level) + 1) * channel, 1, 1, 0, bias=False)
        self.fuse = convbnrelu(channel, channel, k=1, s=1, p=0, relu=False)
        ### SpatialGate
        self.convs = nn.Sequential(
                convbnrelu(channel, channel // reduce_factor, 1, 1, 0, bn=True, relu=True),
                DSConv3x3(channel // reduce_factor, channel // reduce_factor, stride=1, dilation=2),
                DSConv3x3(channel // reduce_factor, channel // reduce_factor, stride=1, dilation=4),
                nn.Conv2d(channel // reduce_factor, 1, 1, 1, 0, bias=False)
                )

    def forward(self, x):
        conv = self.conv(x)
        brs = [branch(conv) for branch in self.branches]
        brs.append(conv)
        gather = sum(brs)

        ### ChannelGate
        d = self.gap(gather)
        d = self.fc2(self.fc1(d))
        d = torch.unsqueeze(d, dim=1).view(-1, len(self.dilation_level) + 1, self.planes, 1, 1)

        ### SpatialGate
        s = self.convs(gather).unsqueeze(1)

        ### Fuse two gates
        f = d * s
        f = F.softmax(f, dim=1)

        return self.fuse(sum([brs[i] * f[:, i, ...] for i in range(len(self.dilation_level) + 1)]))	+ x

###############################################################################################################################





class LDUNet(nn.Module):
	def __init__(self, n_channels, n_classes, deep_supervision = False):
		super(LDUNet, self).__init__()
		self.deep_supervision = deep_supervision
		self.inc = inconv(n_channels, 64)
		self.down1 = down(64, 128)
		self.down2 = down(128, 256)
		self.down3 = down(256, 512)
		self.down4 = down(512, 512)
		self.up1 = up(1024, 256)
		self.up2 = up(512, 128)
		self.up3 = up(256, 64)
		self.up4 = up(128, 64)
		self.sap = VAMM(512,dilation_level=[1,2,4,8])

         #  body
		self.up1b = up(1024, 256)
		self.up2b = up(512, 128)
		self.up3b = up(256, 64)
		self.up4b = up(128, 64)

		 #  detail
		self.up1d = up(1024, 256)
		self.up2d = up(512, 128)
		self.up3d = up(256, 64)
		self.up4d = up(128, 64)
		
		self.outc = outconv(64*3, n_classes)
		self.outs = outconv(64, n_classes)


		self.dsc = DSConv3x3(64*3, 64)
		self.dsoutc4 = outconv(256, n_classes)
		self.dsoutc3 = outconv(128, n_classes)
		self.dsoutc2 = outconv(64, n_classes)
		self.dsoutc1 = outconv(64, n_classes)

	def forward(self, x):
		x1 = self.inc(x)
	
		x2 = self.down1(x1)
		x3 = self.down2(x2)
		x4 = self.down3(x3)
		x5 = self.down4(x4)
		x5=self.sap(x5)
        #Mask
		x44 = self.up1(x5, x4)
		x33 = self.up2(x44, x3)
		x22 = self.up3(x33, x2)
		x11 = self.up4(x22, x1)
        #Body
		x44b = self.up1b(x5, x4)
		x33b = self.up2b(x44, x3)
		x22b = self.up3b(x33, x2)
		x11b = self.up4b(x22, x1)

        #Detail
		x44d = self.up1d(x5, x4)
		x33d = self.up2d(x44, x3)
		x22d = self.up3d(x33, x2)
		x11d = self.up4d(x22, x1)

		
		x0 = self.outs(x11)
		xb=  self.outs(x11b)
		xd= self.outs(x11d)

		xf = torch.cat((x11, x11b, x11d), dim=1)

		xf = self.dsc(xf)
		xf = self.outs(xf)
		# print(xf.shape)
		
	

		# x0 = self.outc(x11)

		if self.deep_supervision:
			x11 = F.interpolate(self.dsoutc1(x11), x0.shape[2:], mode='bilinear')
			x22 = F.interpolate(self.dsoutc2(x22), x0.shape[2:], mode='bilinear')
			x33 = F.interpolate(self.dsoutc3(x33), x0.shape[2:], mode='bilinear')
			x44 = F.interpolate(self.dsoutc4(x44), x0.shape[2:], mode='bilinear')
			
			return x0, x11, x22, x33, x44
		else:
			return x0, xb, xd, xf


if __name__ == '__main__':
    ras =LDUNet(n_channels=1, n_classes=1).cuda()
    input_tensor = torch.randn(4, 1, 96, 96).cuda()
    out = ras(input_tensor)
    # print(out[0].shape)


