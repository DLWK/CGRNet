import glob
import numpy as np
import torch
import os
import cv2
import csv
import tqdm
# from models import ModelBuilder, SegmentationModule, SAUNet, VGG19UNet,VGG19UNet_without_boudary
# from models.unet import UNet
# from models.fcn import get_fcn8s
# from models.AttU_Net_model  import AttU_Net
# from models.R2U_Net_model  import R2U_Net
# from models.denseunet_model import DenseUnet
# from models.cenet import CE_Net
# # from models.UNet_2Plus import  UNet_2Plus
# from models.BaseNet import CPFNet 
# # from models.vggunet import  VGGUNet

from CGRmodes.CGR import CGRNet
# from transunet_pytorch.utils.transunet import TransUNet

# from medpy import metric
from evaluation import *
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
from data.dataloader import XSDataset, XSDatatest
from medpy import metric

def calculate_metric_percase(pred, gt):
    pred[pred > 0.5] = 1
    pred[pred != 1] = 0
    
    gt[gt > 0] = 1

    try:
        dice = metric.binary.dc(pred, gt)
        jc = metric.binary.jc(pred, gt)
        hd95 = metric.binary.hd95(pred, gt)
        hd = metric.binary.hd(pred, gt) 
        return dice,hd95,jc,hd
    except: 
        return 0,0,0,0



if __name__ == "__main__":

    fold=1
    # 选择设备，有cuda用cuda，没有就用cpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 加载网络，图片单通道，分类为1。
    # net =Inf_Net()
    # net = UNet(n_channels=1, n_classes=1)
    # net =  UNet_3Plus(in_channels=1, n_classes=1)
    # net = SeResUNet(n_channels=1, n_classes=1)
    # net =  DilatedResUnet(n_channels=1, n_classes=1)
    # net =  SegNet(input_nbr=1,label_nbr=1)
    # net = UNet_2Plus(in_channels=1, n_classes=1)
    # net =  DenseUnet(in_ch=1, num_classes=1)
    # net = SceResUNet(n_channels=1, n_classes=1)
    # net = myChannelUnet(in_ch=1, out_ch=1)
    # net = ResUNet(n_channels=1, n_classes=1)
    # net =  DenseUnet(in_ch=1, num_classes=1)
    # net = UNet_2Plus(in_channels=1, n_classes=1)
    # net =get_fcn8s(n_class=1)
    # net = VGG19UNet_without_boudary(n_channels=1, n_classes=1)
    # net = R2U_Net(img_ch=1, output_ch=1)
    # net = VGGUNet(n_channels=1, n_classes=1)
    # net = CPFNet()
    # net = CE_Net(num_classes=1, num_channels=3)
    # net = AttU_Net(img_ch=1, output_ch=1)       # 加载网络..........***************
    # 将网络拷贝到deivce中
    net = CGRNet(n_channels=3, n_classes=1)
    # net= TransUNet(img_dim=352,
    #                       in_channels=3,
    #                       out_channels=128,              ########
    #                       head_num=4,
    #                       mlp_dim=512,
    #                       block_num=8,
    #                       patch_dim=16,
    #                       class_num=1)
    # net = VGG19UNet(n_channels=1, n_classes=1)
    # net.to(device=device)
    # 加载模型参数
    net.load_state_dict(torch.load('/home/wangkun/BPGL/result/EPGNet/XS/EPNet_best_'+str(fold)+'.pth', map_location=device))
    # 测试模式
    # net.eval()
    # 读取所有图片路径
    # tests_path = glob.glob('/home/wangkun/shape-attentive-unet/data/test_96/image/*.jpg')
    # mask_path = "/home/wangkun/shape-attentive-unet/data/test_96/label/"
    # save_path = "/home/wangkun/shape-attentive-unet/data/test_96/MyNet-baseline/"

    # 
    # image_path  = "/home/wangkun/data/XS/Test/CVC-ClinicDB/images/"       #Kvasir
    test_data_path = "/home/wangkun/data/dataset/XS/Test/Kvasir/"
   
    save_path = "/home/wangkun/data/dataset/XS/Test/Kvasir/EPNet/"

    if not os.path.exists(save_path):
        os.mkdir(save_path)

    
    
         
    # 遍历素有图片
    test_dataset = XSDatatest(test_data_path)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                               batch_size=1,
                                               shuffle=False) 


   
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
        DICE = 0
        hd95=0
        hd=0
        count = 0
        jc=0
        for image, label, image_path in tqdm.tqdm(test_loader):
            # print(image)
            for test_path in image_path:
                
                name = test_path.split('/')[-1][0:-4]
                save_path1 = save_path + name+ ".png"
                
            
            image = image.to(device=device, dtype=torch.float32)
            label = label.to(device=device, dtype=torch.float32)
            # pred,p1,p2,p3,p4,e= net(image)
            e1,p2 = net(image)
            # p1,p2,p3,p4,ss = net(image)
            pred = sig(p2)

            pred2 = np.array(pred.data.cpu()[0])[0]
            pred1 = np.array(pred.data.cpu()[0])[0]
            pred1[pred1 >= 0.5] = 255
            pred1[pred1 < 0.5] = 0
            img = pred1
            
            # print(label.shape)
            # print(pred.shape)
            test_mask = label.cpu().numpy()
            # print(test_mask.min())
            dice_s, hd95_s, jc_s,hd_s = calculate_metric_percase(pred2, test_mask)
            # print(hd_s)
            



        # 保存图片
            cv2.imwrite(save_path1, img)
            # print(pred.shape)
            acc += get_accuracy(pred,label)
            SE += get_sensitivity(pred,label)
            SP += get_specificity(pred,label)
            PC += get_precision(pred,label)
            F1 += get_F1(pred,label)
            JS += get_JS(pred,label)
            DC += get_DC(pred,label)
            DICE += dice_s
            hd95 += hd95_s
            hd += hd_s
            jc += jc_s
            count+=1
        acc = acc/count
        SE = SE/count
        SP = SP/count
        PC = PC/count
        F1 = F1/count
        JS = JS/count
        DC = DC/count
        DICE = DICE/count
        hd95= hd95/count
        hd= hd/count
    # dc =dc/count
        jc = jc/count
        
        
    print('ACC:%.4f' % acc)
    print('SE:%.4f' % SE)
    print('SP:%.4f' % SP)
    print('PC:%.4f' % PC)
    print('F1:%.4f' % F1)
    print('JS:%.4f' % JS)
    print('DC:%.4f' % DC)
    print('DICE:%.4f' %  DICE)
    print('hd95:%.4f' %  hd95)
    print('hd:%.4f' %  hd)
    print('jc:%.4f' %  jc)
 
    # f = open('./Ablation/vggu19+SAM.csv', 'w')
    # f.write('name,dice,iou,sen,pp'+'\n')
#     for test_path in tqdm.tqdm(tests_path):
        
#         name = test_path.split('/')[-1][0:-4]
       
    
#         mask = mask_path + name+".png"
       
#         image_path = image_path + name+".png"
        
#         # pred_name =  name+"_mask.png"
 
#         mask = cv2.imread(mask,0)

        
#         mask = torch.from_numpy(mask).cuda()
        


#         mask  = mask  / 255
     
       

#         # save_res_path = save_path+name + '_res.jpg'
#         save_mask_path = save_path+ name + '.png'

#         # 读取图片
#         img = cv2.imread(image_path ,1)
#         img = cv2.resize(img,(352,352))
#         mask = cv2.resize(mask,(352,352), interpolation=cv2.INTER_NEAREST)
#         # 转为灰度图
#         # img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
#         # 转为batch为1，通道为1，大小为96*96的数组
#         img = img.reshape(1, 3, img.shape[0], img.shape[1])
#         # 转为tensor
#         img_tensor = torch.from_numpy(img)
#         # 将tensor拷贝到device中，只用cpu就是拷贝到cpu中，用cuda就是拷贝到cuda中。
#         img_tensor = img_tensor.to(device=device, dtype=torch.float32)
#         # 预测
#         # pred = net(img_tensor)
#         e, pred = net(img_tensor)
#         sig = torch.nn.Sigmoid()
#         pred = sig(pred)
       

#         # 提取结果
#         pred1 = np.array(pred.data.cpu()[0])[0]
#         # # 处理结果
#         pred1[pred1 >= 0.5] = 255
#         pred1[pred1 < 0.5] = 0
#         img = pred1
#         # 保存图片
#         cv2.imwrite(save_mask_path, img)
        
#         # hd_s = metric.hd(mask, pred, voxelspacing= 0.3515625)
#         # f.write(name+","+str(dice_s)+","+str(iou_s)+","+str(sen_s)+","+str(ppv_s)+"\n")
#         acc += get_accuracy(pred,mask)
#         SE += get_sensitivity(pred,mask)
#         SP += get_specificity(pred,mask)
#         PC += get_precision(pred,mask)
#         F1 += get_F1(pred,mask)
#         JS += get_JS(pred,mask)
#         DC += get_DC(pred,mask)
#         count+=1
#     acc = acc/count
#     SE = SE/count
#     SP = SP/count
#     PC = PC/count
#     F1 = F1/count
#     JS = JS/count
#     DC = DC/count
# # hd_score = hd/count
    # print('ACC:%.4f' % acc)
    # print('SE:%.4f' % SE)
    # print('SP:%.4f' % SP)
    # print('PC:%.4f' % PC)
    # print('F1:%.4f' % F1)
    # print('JS:%.4f' % JS)
    # print('DC:%.4f' % DC)
   





# python<predict.py>sce2.txts
#by kun wang