import torch
import cv2
import os
import glob
from torch.utils.data import Dataset
import random
import torch.nn.functional as F


###lung nodule
class FJJ_Loader(Dataset):
    def __init__(self, data_path):
        # 初始化函数，读取所有data_path下的图片
        self.data_path = data_path
        self.imgs_path = glob.glob(os.path.join(data_path, 'image/*.png'))
     
    def augment(self, image, flipCode):
        # 使用cv2.flip进行数据增强，filpCode为1水平翻转，0垂直翻转，-1水平+垂直翻转
        flip = cv2.flip(image, flipCode)
        return flip
        
    def __getitem__(self, index):
        # 根据index读取图片
        image_path = self.imgs_path[index]
        # 根据image_path生成label_path
        label_path = image_path.replace('image', 'label').split('.')[0]+"_mask.png"
        label_path1 = image_path.replace('image', 'body-origin').split('.')[0]+".png"
        label_path2 = image_path.replace('image', 'detail-origin').split('.')[0]+".png"
        # edge_path = image_path.replace('image', 'label').split('.')[0]+"_edge.png"
        # 读取训练图片和标签图片
        
        image = cv2.imread(image_path, 0)
        label = cv2.imread(label_path, 0)
        label1 = cv2.imread(label_path1, 0)
        label2 = cv2.imread(label_path2, 0)
        # edge = cv2.imread(edge_path, 0)
      
        # 将数据转为单通道的图片
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # label = cv2.cvtColor(label, cv2.COLOR_BGR2GRAY)
        image = image.reshape(1, image.shape[0], image.shape[1])
        label = label.reshape(1, label.shape[0], label.shape[1])
        label1 = label1.reshape(1, label1.shape[0], label1.shape[1])
        label2 = label2.reshape(1, label2.shape[0], label2.shape[1])
        # edge = edge.reshape(1, edge.shape[0], edge.shape[1])


        # 处理标签，将像素值为255的改为1
        if label.max() > 1:
            label = label / 255
            label1 = label1 / 255
            label2 = label2 / 255
        # if edge.max() > 1:
        #     edge = edge / 255
        # 随机进行数据增强，为2时不做处理
        flipCode = random.choice([-1, 0, 1, 2])
        if flipCode != 2:
            image = self.augment(image, flipCode)
            label = self.augment(label, flipCode)
            label1 = self.augment(label1, flipCode)
            label2 = self.augment(label2, flipCode)
            # edge = self.augment(edge, flipCode)
    
        return image, label, label1, label2

    def __len__(self):
        # 返回训练集大小
        return len(self.imgs_path)



class FJJ_Loadertest(Dataset):
    def __init__(self, data_path):
        # 初始化函数，读取所有data_path下的图片
        self.data_path = data_path
        self.imgs_path = glob.glob(os.path.join(data_path, 'image/*.png'))
       
    def augment(self, image, flipCode):
        # 使用cv2.flip进行数据增强，filpCode为1水平翻转，0垂直翻转，-1水平+垂直翻转
        flip = cv2.flip(image, flipCode)
        return flip
        
    def __getitem__(self, index):
        # 根据index读取图片
        image_path = self.imgs_path[index]
        # 根据image_path生成label_path
        label_path = image_path.replace('image', 'label').split('.')[0]+"_mask.png"
        
        # 读取训练图片和标签图片
        
        image = cv2.imread(image_path, 0)
        label = cv2.imread(label_path, 0)
       
      
        # 将数据转为单通道的图片
      
        image = image.reshape(1, image.shape[0], image.shape[1])
        label = label.reshape(1, label.shape[0], label.shape[1])
       

        # 处理标签，将像素值为255的改为1
        if label.max() > 1:
            label = label / 255
       
    
        return image, label

    def __len__(self):
        # 返回训练集大小
        return len(self.imgs_path)




###COVD-19
class COVD19_Loader(Dataset):
    def __init__(self, data_path):
        # 初始化函数，读取所有data_path下的图片
        self.data_path = data_path
        self.imgs_path = glob.glob(os.path.join(data_path, 'image/*.png'))
     
    def augment(self, image, flipCode):
        # 使用cv2.flip进行数据增强，filpCode为1水平翻转，0垂直翻转，-1水平+垂直翻转
        flip = cv2.flip(image, flipCode)
        return flip
        
    def __getitem__(self, index):
        # 根据index读取图片
        image_path = self.imgs_path[index]
        
        
        # 根据image_path生成label_path
        name= image_path.split('/')[-1].split('_') [-1]
        # print(name)

        label_path = image_path.replace('image', 'label')
        label_path= os.path.dirname(label_path)+'/'+"tr_mask_"+name
        edge_path = os.path.dirname(label_path)+'/'+"tr_mask_"+name[0:-4]+"_edge.png"
        # print(edge_path)
     
        
        
        label_path1 = image_path.replace('image', 'body-origin')
       
        label_path1= os.path.dirname(label_path1)+'/'+"tr_mask_"+name
        
        label_path2 = image_path.replace('image', 'detail-origin')
        label_path2= os.path.dirname(label_path2)+'/'+"tr_mask_"+name
        # edge_path = image_path.replace('image', 'label').split('.')[0]+"_edge.png"
        # 读取训练图片和标签图片
        
        image = cv2.imread(image_path, 0)
        label = cv2.imread(label_path, 0)
        edge =  cv2.imread(edge_path, 0)
        label1 = cv2.imread(label_path1, 0)
        label2 = cv2.imread(label_path2, 0)
        # edge = cv2.imread(edge_path, 0)
      
        # 将数据转为单通道的图片
     
        image = image.reshape(1, image.shape[0], image.shape[1])
        label = label.reshape(1, label.shape[0], label.shape[1])
        edge = edge.reshape(1, edge.shape[0], edge.shape[1])
        label1 = label1.reshape(1, label1.shape[0], label1.shape[1])
        label2 = label2.reshape(1, label2.shape[0], label2.shape[1])
        # edge = edge.reshape(1, edge.shape[0], edge.shape[1])


        # 处理标签，将像素值为255的改为1
        # if label.max() > 1:
        label = label / 255
        label1 = label1 / 255
        label2 = label2 / 255
        # if edge.max() > 1:
        #     edge = edge / 255
        # 随机进行数据增强，为2时不做处理
        flipCode = random.choice([-1, 0, 1, 2])
        if flipCode != 2:
            image = self.augment(image, flipCode)
            label = self.augment(label, flipCode)
            label1 = self.augment(label1, flipCode)
            label2 = self.augment(label2, flipCode)
            # edge = self.augment(edge, flipCode)
    
        return image, label,  label1, label2

    def __len__(self):
        # 返回训练集大小
        return len(self.imgs_path)



class COVD19_Loadertest(Dataset):
    def __init__(self, data_path):
        # 初始化函数，读取所有data_path下的图片
        self.data_path = data_path
        self.imgs_path = glob.glob(os.path.join(data_path, 'image/*.png'))
       
    def augment(self, image, flipCode):
        # 使用cv2.flip进行数据增强，filpCode为1水平翻转，0垂直翻转，-1水平+垂直翻转
        flip = cv2.flip(image, flipCode)
        return flip
        
    def __getitem__(self, index):
        # 根据index读取图片
        image_path = self.imgs_path[index]
        name= image_path.split('/')[-1].split('_') [-1]
        # 根据image_path生成label_path
        label_path = image_path.replace('image', 'label')
        label_path= os.path.dirname(label_path)+'/'+"tr_mask_"+name
        # 读取训练图片和标签图片
        
        image = cv2.imread(image_path, 0)
        label = cv2.imread(label_path, 0)
       
      
        # 将数据转为单通道的图片
      
        image = image.reshape(1, image.shape[0], image.shape[1])
        label = label.reshape(1, label.shape[0], label.shape[1])
       

        # 处理标签，将像素值为255的改为1
    
        label = label / 255
       
    
        return image, label

    def __len__(self):
        # 返回训练集大小
        return len(self.imgs_path)


# cv2.resize(label,(320,224))

###COVD-19-e
class COVD19_Loader2(Dataset):
    def __init__(self, data_path):
        # 初始化函数，读取所有data_path下的图片
        self.data_path = data_path
        self.imgs_path = glob.glob(os.path.join(data_path, 'image/*.png'))
     
    def augment(self, image, flipCode):
        # 使用cv2.flip进行数据增强，filpCode为1水平翻转，0垂直翻转，-1水平+垂直翻转
        flip = cv2.flip(image, flipCode)
        return flip
        
    def __getitem__(self, index):
        # 根据index读取图片
        image_path = self.imgs_path[index]
      
        
        
        # 根据image_path生成label_path
        name= image_path.split('/')[-1].split('_') [-1]
        # print(name)

        label_path = image_path.replace('image', 'label')
        label_path= os.path.dirname(label_path)+'/'+"tr_mask_"+name
       
        edge_path = os.path.dirname(label_path)+'/'+"tr_mask_"+name[0:-4]+"_edge.png"
      
     
        
        
        # label_path1 = image_path.replace('image', 'body-origin')
       
        # label_path1= os.path.dirname(label_path1)+'/'+"tr_mask_"+name
        
        # label_path2 = image_path.replace('image', 'detail-origin')
        # label_path2= os.path.dirname(label_path2)+'/'+"tr_mask_"+name
        # edge_path = image_path.replace('image', 'label').split('.')[0]+"_edge.png"
        # 读取训练图片和标签图片
        
        image = cv2.imread(image_path, 0)
        label = cv2.imread(label_path, 0)
        edge =  cv2.imread(edge_path, 0)
        # label1 = cv2.imread(label_path1, 0)
        # label2 = cv2.imread(label_path2, 0)
        # edge = cv2.imread(edge_path, 0)
      
        # 将数据转为单通道的图片
     
        image = image.reshape(1, image.shape[0], image.shape[1])
        label = label.reshape(1, label.shape[0], label.shape[1])
        edge = edge.reshape(1, edge.shape[0], edge.shape[1])
        # label1 = label1.reshape(1, label1.shape[0], label1.shape[1])
        # label2 = label2.reshape(1, label2.shape[0], label2.shape[1])
        # edge = edge.reshape(1, edge.shape[0], edge.shape[1])


        # 处理标签，将像素值为255的改为1
        # if label.max() > 1:
        label = label / 255
        # label1 = label1 / 255
        # label2 = label2 / 255
        # if edge.max() > 1:
        edge = edge / 255
        # 随机进行数据增强，为2时不做处理
        flipCode = random.choice([-1, 0, 1, 2])
        if flipCode != 2:
            image = self.augment(image, flipCode)
            label = self.augment(label, flipCode)
            # label1 = self.augment(label1, flipCode)
            # label2 = self.augment(label2, flipCode)
            edge = self.augment(edge, flipCode)
    
        return image, label, edge

    def __len__(self):
        # 返回训练集大小
        return len(self.imgs_path)



class COVD19_Loadertest2(Dataset):
    def __init__(self, data_path):
        # 初始化函数，读取所有data_path下的图片
        self.data_path = data_path
        self.imgs_path = glob.glob(os.path.join(data_path, 'image/*.png'))
       
    def augment(self, image, flipCode):
        # 使用cv2.flip进行数据增强，filpCode为1水平翻转，0垂直翻转，-1水平+垂直翻转
        flip = cv2.flip(image, flipCode)
        return flip
        
    def __getitem__(self, index):
        # 根据index读取图片
        image_path = self.imgs_path[index]
        name= image_path.split('/')[-1].split('_') [-1]
        # 根据image_path生成label_path
        label_path = image_path.replace('image', 'label')
        label_path= os.path.dirname(label_path)+'/'+"tr_mask_"+name
        # 读取训练图片和标签图片
        
        image = cv2.imread(image_path, 0)
        label = cv2.imread(label_path, 0)
       
      
        # 将数据转为单通道的图片
      
        image = image.reshape(1, image.shape[0], image.shape[1])
        label = label.reshape(1, label.shape[0], label.shape[1])
       

        # 处理标签，将像素值为255的改为1
        # if label.max() > 1:

        label = label / 255
       
    
        return image, label

    def __len__(self):
        # 返回训练集大小
        return len(self.imgs_path)


# cv2.resize(label,(320,224))







###skin 2017
class ISIC_Loader(Dataset):
    def __init__(self, data_path):
        # 初始化函数，读取所有data_path下的图片
        self.data_path = data_path
        self.imgs_path = glob.glob(os.path.join(data_path, 'image/*.jpg'))
     
    def augment(self, image, flipCode):
        # 使用cv2.flip进行数据增强，filpCode为1水平翻转，0垂直翻转，-1水平+垂直翻转
        flip = cv2.flip(image, flipCode)
        return flip
        
    def __getitem__(self, index):
        # 根据index读取图片
        image_path = self.imgs_path[index]
     
        # 根据image_path生成label_path
        label_path = image_path.replace('image', 'label').split('.')[0]+"_segmentation.png"
        
        
        # edge_path = image_path.replace('image', 'label').split('.')[0]+"_edge.png"
        # 读取训练图片和标签图片


        image = cv2.imread(image_path, 1)
        label = cv2.imread(label_path, 0)
        image = cv2.resize(image,(256,192))
        label = cv2.resize(label,(256,192))
        
        # edge = cv2.imread(edge_path, 0)
      
        # 将数据转为3通道的图片
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # label = cv2.cvtColor(label, cv2.COLOR_BGR2GRAY)
        image = image.reshape(3, image.shape[0], image.shape[1])
        label = label.reshape(1, label.shape[0], label.shape[1])
        


        # 处理标签，将像素值为255的改为1
        if label.max() > 1:
            label = label / 255
            
        # if edge.max() > 1:
        #     edge = edge / 255
        # 随机进行数据增强，为2时不做处理
        flipCode = random.choice([-1, 0, 1, 2])
        if flipCode != 2:
            image = self.augment(image, flipCode)
            label = self.augment(label, flipCode)
           
    
        return image, label

    def __len__(self):
        # 返回训练集大小
        return len(self.imgs_path)



class ISIC_Loadertest(Dataset):
    def __init__(self, data_path):
        # 初始化函数，读取所有data_path下的图片
        self.data_path = data_path
        self.imgs_path = glob.glob(os.path.join(data_path, 'image/*.jpg'))
       
    def augment(self, image, flipCode):
        # 使用cv2.flip进行数据增强，filpCode为1水平翻转，0垂直翻转，-1水平+垂直翻转
        flip = cv2.flip(image, flipCode)
        return flip
        
    def __getitem__(self, index):
        # 根据index读取图片
        image_path = self.imgs_path[index]
        # 根据image_path生成label_path
        label_path = image_path.replace('image', 'label').split('.')[0]+"_segmentation.png"
        
        # 读取训练图片和标签图片

        image = cv2.imread(image_path, 1)
        label = cv2.imread(label_path, 0)
        image = cv2.resize(image,(256,192))
        label = cv2.resize(label,(256,192))
       
      
        # 将数据转为单通道的图片
      
        image = image.reshape(3, image.shape[0], image.shape[1])
        label = label.reshape(1, label.shape[0], label.shape[1])
       

        # 处理标签，将像素值为255的改为1
        if label.max() > 1:
            label = label / 255
       
    
        return image, label

    def __len__(self):
        # 返回训练集大小
        return len(self.imgs_path)




import os
import numpy as np
from glob import glob
from PIL import Image
import torch
from torchvision.transforms import Compose, CenterCrop, Normalize, ToTensor
from transform import ReLabel, ToLabel, Scale, HorizontalFlip, VerticalFlip, ColorJitter
import random
import cv2
import glob
def makedirs(path):
    if not os.path.exists(path):
        os.makedirs(path)


class BasicDataset(torch.utils.data.Dataset):

    def __init__(self, data):
        # self.size = (180,135)
        self.size = (256, 192)
        # self.root = root
        self.data = data
        # self.label = label
        # if not os.path.exists(self.root):
        #     raise Exception("[!] {} not exists.".format(root))
        self.img_resize = Compose([
            Scale(self.size, Image.BILINEAR),
            # We can do some colorjitter augmentation here
            # ColorJitter(brightness=0, contrast=0, saturation=0, hue=0),
        ])
        self.label_resize = Compose([
            Scale(self.size, Image.NEAREST),
        ])
        self.img_transform = Compose([
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]),
            # Normalize(mean=[0.448749], std=[0.399953])
        ])
        self.hsv_transform = Compose([
            ToTensor(),
        ])
        self.label_transform = Compose([
            ToLabel(),
            ReLabel(127, 1),
        ])
        # sort file names
        # self.input_paths = sorted(glob(os.path.join(self.root, '{}/*.jpg'.format("train_data"))))
        # self.label_paths = sorted(glob(os.path.join(self.root, '{}/*.jpg'.format("train_groundtruth"))))
        self.input_paths = glob.glob(os.path.join(data, 'image/*.jpg'))
        # self.label_paths = self.label
    
        # self.name = os.path.basename(root)
        # if len(self.input_paths) == 0 or len(self.label_paths) == 0:
        #     raise Exception("No images/labels are found in {}".format(self.root))
    def augment(self, image, flipCode):
        flip = cv2.flip(image, flipCode)
        return flip
    def __getitem__(self, index):
        # image = Image.open(self.input_paths[index]).convert('RGB')
        # image = Image.open(self.input_paths[index])  # 0-255
        # # image_hsv = Image.open(self.input_paths[index]).convert('HSV')
        # label = Image.open(self.label_paths[index]).convert('P')  # 8位像素，使用调色板映射到任何其他模式

        

        image_path = self.input_paths[index]
        
        label_path = image_path.replace('image', 'label').split('.')[0]+"_segmentation.png"
        label_path1 = image_path.replace('image', 'body-origin').split('.')[0]+"_segmentation.png"
        label_path2 = image_path.replace('image', 'detail-origin').split('.')[0]+"_segmentation.png"
        
        image = Image.open(image_path).convert('RGB')
        label = Image.open(label_path).convert('L')
        label1 = Image.open(label_path1).convert('L')
        label2 = Image.open(label_path2).convert('L')
        image = self.img_resize(image)
        label = self.label_resize(label)
        label1 =self.label_resize(label1)
        label2 = self.label_resize(label2)
        # image_hsv = self.img_resize(image_hsv)
        # image = image.reshape(1, image.shape[0], image.shape[1])
        # label = label.reshape(1, label.shape[0], label.shape[1])
        # randomly flip images
        
        # if random.random() > 0.5:
        #     image = HorizontalFlip()(image)
        #     # image_hsv = HorizontalFlip()(image_hsv)
        #     label = HorizontalFlip()(label)
        # if random.random() > 0.5:
        #     image = VerticalFlip()(image)
        #     label = VerticalFlip()(label)
        # flipCode = random.choice([-1, 0, 1, 2])
        # if flipCode != 2:
        #     image = self.augment(image, flipCode)
        #     label = self.augment(label, flipCode)

        # randomly crop image to size 128*128
        # print(image.size)
        w, h = image.size
        # th, tw = (128,128)
        th, tw = (256, 192)
        # x1 = random.randint(0, w - tw)
        # y1 = random.randint(0, h - th)
        # if w == tw and h == th:
        image = image
        

        # image_hsv = image_hsv
        label = label
        
        
        image = self.img_transform(image)
        # image_hsv = self.hsv_transform(image_hsv)
        # image = torch.cat([image,image_hsv],0)
        
        label = self.label_transform(label)
        label1 = self.label_transform(label1)
        label2 = self.label_transform(label2)
        # label_d1 = self.label_transform(label_d1)
        # label_d2 = self.label_transform(label_d2)
        # label_d3 = self.label_transform(label_d3)
        # label_d4 = self.label_transform(label_d4)


        # print(image.max())
        # print(image.min())
        # print(label.max())
        # print(label.min())
       
        # return image, label, label_d1, label_d2, label_d3, label_d4
        # image = image.reshape(1, image.shape[0], image.shape[1])
        # label = label.reshape(1, label.shape[0], label.shape[1])
        return image, label, label1, label2 
    def __len__(self):

        return len(self.input_paths) 

class ISICtest(torch.utils.data.Dataset):

    def __init__(self, data):
        # self.size = (180,135)
        self.size = (256, 192)
        # self.root = root
        self.data = data
        # self.label = label
        # if not os.path.exists(self.root):
        #     raise Exception("[!] {} not exists.".format(root))
        self.img_resize = Compose([
            Scale(self.size, Image.BILINEAR),
            # We can do some colorjitter augmentation here
            # ColorJitter(brightness=0, contrast=0, saturation=0, hue=0),
        ])
        self.label_resize = Compose([
            Scale(self.size, Image.NEAREST),
        ])
        self.img_transform = Compose([
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]),
            # Normalize(mean=[0.448749], std=[0.399953])
        ])
        self.hsv_transform = Compose([
            ToTensor(),
        ])
        self.label_transform = Compose([
            ToLabel(),
            ReLabel(127, 1),
        ])
        # sort file names
        # self.input_paths = sorted(glob(os.path.join(self.root, '{}/*.jpg'.format("train_data"))))
        # self.label_paths = sorted(glob(os.path.join(self.root, '{}/*.jpg'.format("train_groundtruth"))))
        self.input_paths = glob.glob(os.path.join(data, 'image/*.jpg'))
        # self.label_paths = self.label
    
        # self.name = os.path.basename(root)
        # if len(self.input_paths) == 0 or len(self.label_paths) == 0:
        #     raise Exception("No images/labels are found in {}".format(self.root))
    def augment(self, image, flipCode):
        flip = cv2.flip(image, flipCode)
        return flip
    def __getitem__(self, index):
        # image = Image.open(self.input_paths[index]).convert('RGB')
        # image = Image.open(self.input_paths[index])  # 0-255
        # # image_hsv = Image.open(self.input_paths[index]).convert('HSV')
        # label = Image.open(self.label_paths[index]).convert('P')  # 8位像素，使用调色板映射到任何其他模式

        

        image_path = self.input_paths[index]
        
        label_path = image_path.replace('image', 'label').split('.')[0]+"_segmentation.png"
        
        image = Image.open(image_path).convert('RGB')
        label = Image.open(label_path).convert('L')
       
        image = self.img_resize(image)
        label = self.label_resize(label)
       
        # image_hsv = self.img_resize(image_hsv)
        # image = image.reshape(1, image.shape[0], image.shape[1])
        # label = label.reshape(1, label.shape[0], label.shape[1])
        # randomly flip images
        
        # if random.random() > 0.5:
        #     image = HorizontalFlip()(image)
        #     # image_hsv = HorizontalFlip()(image_hsv)
        #     label = HorizontalFlip()(label)
        # if random.random() > 0.5:
        #     image = VerticalFlip()(image)
        #     label = VerticalFlip()(label)
        # flipCode = random.choice([-1, 0, 1, 2])
        # if flipCode != 2:
        #     image = self.augment(image, flipCode)
        #     label = self.augment(label, flipCode)

        # randomly crop image to size 128*128
        # print(image.size)
        w, h = image.size
        # th, tw = (128,128)
        th, tw = (256, 192)
        # x1 = random.randint(0, w - tw)
        # y1 = random.randint(0, h - th)
     
        
        
        image = self.img_transform(image)
        # image_hsv = self.hsv_transform(image_hsv)
        # image = torch.cat([image,image_hsv],0)
        
        label = self.label_transform(label)
       
        
        return image, label
    def __len__(self):

        return len(self.input_paths) 














class ISICDataset(torch.utils.data.Dataset):

    def __init__(self, data):
        # self.size = (180,135)
        self.size = (256, 192)
        # self.root = root
        self.data = data
        # self.label = label
        # if not os.path.exists(self.root):
        #     raise Exception("[!] {} not exists.".format(root))
        self.img_resize = Compose([
            Scale(self.size, Image.BILINEAR),
            # We can do some colorjitter augmentation here
            # ColorJitter(brightness=0, contrast=0, saturation=0, hue=0),
        ])
        self.label_resize = Compose([
            Scale(self.size, Image.NEAREST),
        ])
        self.img_transform = Compose([
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]),
            # Normalize(mean=[0.448749], std=[0.399953])
        ])
        self.hsv_transform = Compose([
            ToTensor(),
        ])
        self.label_transform = Compose([
            ToLabel(),
            ReLabel(127, 1),
        ])
        # sort file names
        # self.input_paths = sorted(glob(os.path.join(self.root, '{}/*.jpg'.format("train_data"))))
        # self.label_paths = sorted(glob(os.path.join(self.root, '{}/*.jpg'.format("train_groundtruth"))))
        self.input_paths = glob.glob(os.path.join(data, 'image/*.jpg'))
        # self.label_paths = self.label
    
        # self.name = os.path.basename(root)
        # if len(self.input_paths) == 0 or len(self.label_paths) == 0:
        #     raise Exception("No images/labels are found in {}".format(self.root))
    def augment(self, image, flipCode):
        flip = cv2.flip(image, flipCode)
        return flip
    def __getitem__(self, index):
        # image = Image.open(self.input_paths[index]).convert('RGB')
        # image = Image.open(self.input_paths[index])  # 0-255
        # # image_hsv = Image.open(self.input_paths[index]).convert('HSV')
        # label = Image.open(self.label_paths[index]).convert('P')  # 8位像素，使用调色板映射到任何其他模式

        

        image_path = self.input_paths[index]
        # print(image_path)
        label_path = image_path.replace('image', 'label').split('.')[0]+"_Segmentation.png"
        # print(label_path)
        edge_path  = image_path.replace('image', 'label').split('.')[0]+"_Segmentation_edge.png"
        image = Image.open(image_path).convert('RGB')
        label = Image.open(label_path)
        edge = Image.open(edge_path)
        image = self.img_resize(image)
        label = self.img_resize(label)
        edge  = self.img_resize(edge)
        # image_hsv = self.img_resize(image_hsv)
        # image = image.reshape(1, image.shape[0], image.shape[1])
        # label = label.reshape(1, label.shape[0], label.shape[1])
        # randomly flip images
        
        # if random.random() > 0.5:
        #     image = HorizontalFlip()(image)
        #     # image_hsv = HorizontalFlip()(image_hsv)
        #     label = HorizontalFlip()(label)
        # if random.random() > 0.5:
        #     image = VerticalFlip()(image)
        #     label = VerticalFlip()(label)
        # flipCode = random.choice([-1, 0, 1, 2])
        # if flipCode != 2:
        #     image = self.augment(image, flipCode)
        #     label = self.augment(label, flipCode)

        # randomly crop image to size 128*128
        # print(image.size)
        w, h = image.size
        # th, tw = (128,128)
        th, tw = (256, 192)
        # x1 = random.randint(0, w - tw)
        # y1 = random.randint(0, h - th)
        # if w == tw and h == th:
        image = image
        

        # image_hsv = image_hsv
        label = label
        # label_d1 = label.resize((48, 48), Image.NEAREST)
        # label_d2 = label.resize((24, 24), Image.NEAREST)
        # label_d3 = label.resize((12, 12), Image.NEAREST)
        # label_d4 = label.resize((6, 6), Image.NEAREST)

        # else:
        #     if random.random() > 0.5:
        #         image = image.resize((256, 192), Image.BILINEAR)
        #         # image_hsv = image_hsv.resize((128,128),Image.BILINEAR)
        #         label = label.resize((256, 192), Image.NEAREST)
        #     else:
        #         pass

                # image = image.crop((x1, y1, x1 + tw, y1 + th))
                # # image_hsv = image_hsv.crop((x1, y1, x1 + tw, y1 + th))
                # label = label.crop((x1, y1, x1 + tw, y1 + th))
        # angle = random.randint(-20, 20)
        # image = image.rotate(angle, resample=Image.BILINEAR)
        # image_hsv = image_hsv.rotate(angle, resample=Image.BILINEAR)
        # label = label.rotate(angle, resample=Image.NEAREST)
        
        image = self.img_transform(image)
        # image_hsv = self.hsv_transform(image_hsv)
        # image = torch.cat([image,image_hsv],0)
        
        label = self.label_transform(label)
        edge =self.label_transform(edge)
        # label_d1 = self.label_transform(label_d1)
        # label_d2 = self.label_transform(label_d2)
        # label_d3 = self.label_transform(label_d3)
        # label_d4 = self.label_transform(label_d4)


        # print(image.max())
        # print(image.min())
        # print(label.max())
        # print(label.min())
       
        # return image, label, label_d1, label_d2, label_d3, label_d4
        # image = image.reshape(1, image.shape[0], image.shape[1])
        # label = label.reshape(1, label.shape[0], label.shape[1])
        return image, label, edge
    def __len__(self):

        return len(self.input_paths)  




class XSDataset(torch.utils.data.Dataset):

    def __init__(self, data):
        # self.size = (180,135)
        self.size = (352, 352)
        # self.root = root
        self.data = data
        # self.label = label
        # if not os.path.exists(self.root):
        #     raise Exception("[!] {} not exists.".format(root))
        self.img_resize = Compose([
            Scale(self.size, Image.BILINEAR),
            # We can do some colorjitter augmentation here
            # ColorJitter(brightness=0, contrast=0, saturation=0, hue=0),
        ])
        self.label_resize = Compose([
            Scale(self.size, Image.NEAREST),
        ])
        self.img_transform = Compose([
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]),
            # Normalize(mean=[0.448749], std=[0.399953])
        ])
        self.hsv_transform = Compose([
            ToTensor(),
        ])
        self.label_transform = Compose([
            ToLabel(),
            ReLabel(127, 1),
        ])
    
        self.input_paths = glob.glob(os.path.join(data, 'image/*.png'))
    
    def augment(self, image, flipCode):
        flip = cv2.flip(image, flipCode)
        return flip
    def __getitem__(self, index):
        
        

        image_path = self.input_paths[index]
        
        label_path = image_path.replace('image', 'mask').split('.')[0]+".png"
        
        label_path1 = image_path.replace('image', 'body-origin').split('.')[0]+".png"
        label_path2 = image_path.replace('image', 'detail-origin').split('.')[0]+".png"
        
        image = cv2.imread(image_path, 1)
        image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        label = cv2.imread(label_path, 0)
        label = Image.fromarray(label)
        label1 = cv2.imread(label_path1, 0)
        label1 = Image.fromarray(label1)
        label2 = cv2.imread(label_path2, 0)
        label2 = Image.fromarray(label2)



        # image = Image.open(image_path).convert('RGB')
        # label = Image.open(label_path).convert('L')
        # label1 = Image.open(label_path1).convert('L')
        # label2 = Image.open(label_path2).convert('L')
        image = self.img_resize(image)
        label = self.label_resize(label)
        label1 = self.label_resize(label1)
        label2 = self.label_resize(label2)
        #
        
        image = self.img_transform(image)
        # image_hsv = self.hsv_transform(image_hsv)
        # image = torch.cat([image,image_hsv],0)
        
        label = self.label_transform(label)
        label1 = self.label_transform(label1)
        label2 = self.label_transform(label2)
        
        return image, label, label1, label2 
    def __len__(self):

        return len(self.input_paths) 


class XSDatatest(torch.utils.data.Dataset):


    def __init__(self, data):
        # self.size = (180,135)
        self.size = (352, 352)
        # self.root = root
        self.data = data
        # self.label = label
        # if not os.path.exists(self.root):
        #     raise Exception("[!] {} not exists.".format(root))
        self.img_resize = Compose([
            Scale(self.size, Image.BILINEAR),
            # We can do some colorjitter augmentation here
            # ColorJitter(brightness=0, contrast=0, saturation=0, hue=0),
        ])
        self.label_resize = Compose([
            Scale(self.size, Image.NEAREST),
        ])
        self.img_transform = Compose([
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]),
            # Normalize(mean=[0.448749], std=[0.399953])
        ])
        self.hsv_transform = Compose([
            ToTensor(),
        ])
        self.label_transform = Compose([
            ToLabel(),
            ReLabel(127, 1),
        ])
      
        self.input_paths = glob.glob(os.path.join(data, 'images/*.png'))
        # self.label_paths = self.label
    
        # self.name = os.path.basename(root)
        # if len(self.input_paths) == 0 or len(self.label_paths) == 0:
        #     raise Exception("No images/labels are found in {}".format(self.root))
    def augment(self, image, flipCode):
        flip = cv2.flip(image, flipCode)
        return flip
    def __getitem__(self, index):
      
        

        image_path = self.input_paths[index]
        # print(image_path)
        label_path = image_path.replace('images', 'masks').split('.')[0]+".png"
        # print(label_path)
        
        image = cv2.imread(image_path, 1)
        image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        label = cv2.imread(label_path, 0)
        label = Image.fromarray(label)
        # image = Image.open(image_path).convert('RGB')
        # label = Image.open(label_path).convert('L')
      
        image = self.img_resize(image)
        label = self.label_resize(label)
     
       
        
        image = self.img_transform(image)
        # image_hsv = self.hsv_transform(image_hsv)
        # image = torch.cat([image,image_hsv],0)
        
        label = self.label_transform(label)
        
        return image, label
    def __len__(self):

        return len(self.input_paths)         