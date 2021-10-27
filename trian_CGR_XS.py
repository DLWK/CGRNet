from torch import optim
# from losses import *
from data.dataloader import XSDataset, XSDatatest
import torch.nn as nn
import torch

from models.unet import UNet

from CGRmodes.CGR import CGRNet
# from models.vggunet import  VGGUNet
# from utils.metric import *
from torchvision.transforms import transforms
from evaluation import *
# from utils.metric import *
import torch.nn.functional as F
# from models.newnet import  FastSal
import tqdm 

def iou_loss(pred, mask):
    weit  = 1+5*torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15)-mask)
    wbce  = F.binary_cross_entropy_with_logits(pred, mask, reduce='none')
    wbce  = (weit*wbce).sum(dim=(2,3))/weit.sum(dim=(2,3))

    pred  = torch.sigmoid(pred)
    inter = ((pred*mask)*weit).sum(dim=(2,3))
    union = ((pred+mask)*weit).sum(dim=(2,3))
    wiou  = 1-(inter+1)/(union-inter+1)
    loss_total= (wbce+wiou).mean()/wiou.size(0)
    return loss_total


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.3, gamma=2, logits=True, reduce=True):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.logits = logits
        self.reduce = reduce

    def forward(self, inputs, targets):
        if self.logits:
            BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduce=False)
        else:
            BCE_loss = F.binary_cross_entropy(inputs, targets, reduce=False)
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss

        if self.reduce:
            return torch.mean(F_loss)
        else:
            return F_loss   



def test(testLoader,fold, net, device):
    net.to(device)
    sig = torch.nn.Sigmoid()
    print('start test!')
    net.eval()
    with torch.no_grad():
         # when in test stage, no grad
        acc = 0.	# Accuracy
        SE = 0.		# Sensitivity (Recall)
        SP = 0.		# Specificity
        PC = 0. 	# Precision
        F1 = 0.		# F1 Score
        JS = 0.		# Jaccard Similarity
        DC = 0.		# Dice Coefficient
        count = 0
        for image, label, path in tqdm.tqdm(testLoader):
            image = image.to(device=device, dtype=torch.float32)
            label = label.to(device=device, dtype=torch.float32)
            # pred,p1,p2,p3,p4,e= net(image)
            e1,p2= net(image)
            pred = sig(p2)
            # print(pred.shape)
            acc += get_accuracy(pred,label)
            SE += get_sensitivity(pred,label) 
            SP += get_specificity(pred,label)
            PC += get_precision(pred,label)
            F1 += get_F1(pred,label)
            JS += get_JS(pred,label)
            DC += get_DC(pred,label)
            count+=1
        acc = acc/count
        SE = SE/count
        SP = SP/count
        PC = PC/count
        F1 = F1/count
        JS = JS/count
        DC = DC/count
        score = JS + DC
        print("\tacc: {:.4f}\tSE: {:.4f}\tSP: {:.4f}\tPC: {:.4f}\tF1: {:.4f} \tJS: {:.4f}".format(acc, SE, SP, PC, F1, JS))
        return  acc, SE, SP, PC, F1, JS, DC, score




def train_net(net, device, data_path,test_data_path, fold, epochs=40, batch_size=4, lr=0.00001):
    # 加载训练集
    isbi_dataset = XSDataset(data_path)
    train_loader = torch.utils.data.DataLoader(dataset=isbi_dataset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               drop_last=True)

    test_dataset = XSDatatest(test_data_path)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                               batch_size=1,
                                               shuffle=False)                                           
    # 定义RMSprop算法11
    optimizer = optim.RMSprop(net.parameters(), lr=lr, weight_decay=1e-8, momentum=0.9)
    # optimizer = optim.Adam(net.parameters(), lr=lr)

    # 定义Loss算法
    
    # criterion2 = nn.BCEWithLogitsLoss()
    # criterion3 = nn.BCELoss()
    criterion2 = FocalLoss()
    
    # best_loss统计，初始化为正无穷
    best_loss = float('inf')
    result = 0
    # 训练epochs次
    # f = open('./segmentation2/UNet.csv', 'w')
    # f.write('epoch,loss'+'\n')
    for epoch in range(epochs):
        # 训练模式
        net.train()
        # 按照batch_size开始训练
        for image, label, edge in train_loader:
          
            optimizer.zero_grad()
            # 将数据拷贝到device中
            image = image.to(device=device, dtype=torch.float32)
            label = label.to(device=device, dtype=torch.float32)
            edge = edge.to(device=device, dtype=torch.float32)
            # 使用网络参数，输出预测结果
            # pred, p1,p2,p3,p4,e= net(image)
            # # 计算loss
            e1,p2= net(image)
            loss = iou_loss(p2, label)+ criterion2(e1, edge)
            
            print('Train Epoch:{}'.format(epoch))
            print('Loss/train', loss.item())
            # print('Loss/edge', loss1.item())
           
            # 保存loss值最小的网络参数
            if loss < best_loss:
                best_loss = loss
                # torch.save(net.state_dict(), './LUNG/fff'+str(fold)+'.pth')
            # 更新参数
            loss.backward()
            optimizer.step()
        # f.write(str(epoch)+","+str(best_loss.item())+"\n")    
        if epoch>0:
            acc, SE, SP, PC, F1, JS, DC, score=test(test_loader,fold, net, device)
            if result < score:
                result = score
                # best_epoch = epoch
                torch.save(net.state_dict(), '/home/wangkun/BPGL/result/EPGNet/XS/EPNet_best_'+str(fold)+'.pth')
                with open("/home/wangkun/BPGL/result/EPGNet/XS/EPNet_"+str(fold)+".csv", "a") as w:
                    w.write("epoch="+str(epoch)+",acc="+str(acc)+", SE="+str(SE)+",SP="+str(SP)+",PC="+str(PC)+",F1="+str(F1)+",JS="+str(JS)+",DC="+str(DC)+",Score="+str(score)+"\n")


if __name__ == "__main__":
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    def weights_init(m):
        classname = m.__class__.__name__
        # print(classname)
        if classname.find('Conv') != -1:
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                torch.nn.init.constant_(m.bias.data, 0.0)
    seed=1234
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # 选择设备，有cuda用cuda，没有就用cpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 加载网络，图片单通道1，分类为1。
    # net = UNet(n_channels=1, n_classes=1)
    # net = MyMGNet(n_channels=1, n_classes=1)
    net = CGRNet(n_channels=3, n_classes=1)
    # net = VGG19UNet_without_boudary(n_channels=1, n_classes=1)
    # net = R2U_Net(img_ch=1, output_ch=1)
    # net = CE_Net(num_classes=1, num_channels=1)
    # net = VGG19UNet(n_channels=1, n_classes=1)
    # net = AttU_Net(img_ch=1, output_ch=1)
    # net =get_fcn8s(n_class=1)
    # net = UNet_2Plus(in_channels=1, n_classes=1)
    # net =  DenseUnet(in_ch=1, num_classes=1)
    # net = CPFNet()
    net.to(device=device)
    # 指定训练集地址，开始训练
    fold=1
    # data_path = "/home/wangkun/shape-attentive-unet/data/train_96"
    # data_path = "/home/wangkun/shape-attentive-unet/data/ISIC/train"
    # test_data_path = "/home/wangkun/shape-attentive-unet/data/ISIC/test"
    # data_path = "/home/wangkun/shape-attentive-unet/data/LUNG/train"
    # test_data_path = "/home/wangkun/shape-attentive-unet/data/LUNG/test"
    data_path = "/home/wangkun/data/dataset/XS/Train/"
    test_data_path = "/home/wangkun/data/dataset/XS/val/"

   


    train_net(net, device, data_path,test_data_path, fold)
        
   



    #by kun wang