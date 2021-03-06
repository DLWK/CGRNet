import torch
import cv2
import os
import glob
from torch.utils.data import Dataset
import random
import torch.nn.functional as F
# class ISBI_Loader(Dataset):
#     def __init__(self, data_path,transform):
#         # 初始化函数，读取所有data_path下的图片
#         self.data_path = data_path
#         self.imgs_path = glob.glob(os.path.join(data_path, 'image/*.jpg'))
#         self.transform=transform
#     def augment(self, image, flipCode):
#         # 使用cv2.flip进行数据增强，filpCode为1水平翻转，0垂直翻转，-1水平+垂直翻转
#         flip = cv2.flip(image, flipCode)
#         return flip
        
#     def __getitem__(self, index):
#         # 根据index读取图片
#         image_path = self.imgs_path[index]
#         # 根据image_path生成label_path
#         label_path = image_path.replace('image', 'label').split('.')[0]+"_mask.png"
#         edge_path = image_path.replace('image', 'label').split('.')[0]+"_edge.jpg"
#         # 读取训练图片和标签图片
        
#         image = cv2.imread(image_path, 0)
#         label = cv2.imread(label_path, 0)
#         edge = cv2.imread(edge_path, 0)
      
#         # 将数据转为单通道的图片
#         # image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#         # label = cv2.cvtColor(label, cv2.COLOR_BGR2GRAY)
#         image = image.reshape(1, image.shape[0], image.shape[1])
#         label = label.reshape(1, label.shape[0], label.shape[1])
#         edge = edge.reshape(1, edge.shape[0], edge.shape[1])
#         # 处理标签，将像素值为255的改为1
#         if label.max() > 1:
#             label = label / 255
#         if edge.max() > 1:
#             edge = edge / 255
#         # 随机进行数据增强，为2时不做处理
#         flipCode = random.choice([-1, 0, 1, 2])
#         if flipCode != 2:
#             image = self.augment(image, flipCode)
#             label = self.augment(label, flipCode)
#             edge = self.augment(edge, flipCode)
#         return image, label, edge

#     def __len__(self):
#         # 返回训练集大小
#         return len(self.imgs_path)


class ISBI_Loader(Dataset):
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
        edge_path = image_path.replace('image', 'label').split('.')[0]+"_edge.png"
        # 读取训练图片和标签图片
        
        image = cv2.imread(image_path, 0)
        label = cv2.imread(label_path, 0)
        edge = cv2.imread(edge_path, 0)
      
        # 将数据转为单通道的图片
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # label = cv2.cvtColor(label, cv2.COLOR_BGR2GRAY)
        image = image.reshape(1, image.shape[0], image.shape[1])
        label = label.reshape(1, label.shape[0], label.shape[1])
        edge = edge.reshape(1, edge.shape[0], edge.shape[1])


        # 处理标签，将像素值为255的改为1
        if label.max() > 1:
            label = label / 255
        if edge.max() > 1:
            edge = edge / 255
        # 随机进行数据增强，为2时不做处理
        flipCode = random.choice([-1, 0, 1, 2])
        if flipCode != 2:
            image = self.augment(image, flipCode)
            label = self.augment(label, flipCode)
            edge = self.augment(edge, flipCode)
    
        return image, label, edge

    def __len__(self):
        # 返回训练集大小
        return len(self.imgs_path)



class ISBI_Loadertest(Dataset):
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
        # edge_path = image_path.replace('image', 'label').split('.')[0]+"_edge.png"
        # 读取训练图片和标签图片
        
        image = cv2.imread(image_path, 0)
        label = cv2.imread(label_path, 0)
        # edge = cv2.imread(edge_path, 0)
      
        # 将数据转为单通道的图片
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # label = cv2.cvtColor(label, cv2.COLOR_BGR2GRAY)
        image = image.reshape(1, image.shape[0], image.shape[1])
        label = label.reshape(1, label.shape[0], label.shape[1])
        # edge = edge.reshape(1, edge.shape[0], edge.shape[1])

        # 处理标签，将像素值为255的改为1
        if label.max() > 1:
            label = label / 255
        # if edge.max() > 1:
        #     edge = edge / 255
        # 随机进行数据增强，为2时不做处理
        # flipCode = random.choice([-1, 0, 1, 2])
        # if flipCode != 2:
        #     image = self.augment(image, flipCode)
        #     label = self.augment(label, flipCode)
        #     edge = self.augment(edge, flipCode)
        return image, label

    def __len__(self):
        # 返回训练集大小
        return len(self.imgs_path)






class liver_Loader(Dataset):
    def __init__(self, data_path):
        # 初始化函数，读取所有data_path下的图片
        self.data_path = data_path
        self.imgs_path = glob.glob(os.path.join(data_path, '*.png'))
        
    def augment(self, image, flipCode):
        # 使用cv2.flip进行数据增强，filpCode为1水平翻转，0垂直翻转，-1水平+垂直翻转
        flip = cv2.flip(image, flipCode)
        return flip
        
    def __getitem__(self, index):
        # 根据index读取图片
        image_path = self.imgs_path[index]
        path = os.path.dirname(image_path)
       
        name =image_path.split('/')[-1][0:3]
       
        image_path = path +'/'+ name +'.png'
       
     
        # 根据image_path生成label_path
        label_path =path +'/'+ name+"_mask.png"
      
        edge_path = path +'/'+name+"_edge.png"
        # 读取训练图片和标签图片
        
        image = cv2.imread(image_path, 0)
        label = cv2.imread(label_path, 0)
        edge = cv2.imread(edge_path, 0)
      
        # 将数据转为单通道的图片
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # label = cv2.cvtColor(label, cv2.COLOR_BGR2GRAY)
        image = image.reshape(1, image.shape[0], image.shape[1])
        label = label.reshape(1, label.shape[0], label.shape[1])
        edge = edge.reshape(1, edge.shape[0], edge.shape[1])


        # 处理标签，将像素值为255的改为1
        if label.max() > 1:
            label = label / 255
        if edge.max() > 1:
            edge = edge / 255
        # 随机进行数据增强，为2时不做处理
        flipCode = random.choice([-1, 0, 1, 2])
        if flipCode != 2:
            image = self.augment(image, flipCode)
            label = self.augment(label, flipCode)
            edge = self.augment(edge, flipCode)
    
        return image, label, edge

    def __len__(self):
        # 返回训练集大小
        return len(self.imgs_path)



class liver_Loadertest(Dataset):
    def __init__(self, data_path):
        # 初始化函数，读取所有data_path下的图片
        self.data_path = data_path
        self.imgs_path = glob.glob(os.path.join(data_path, '*.png'))
       
    
    def augment(self, image, flipCode):
        # 使用cv2.flip进行数据增强，filpCode为1水平翻转，0垂直翻转，-1水平+垂直翻转
        flip = cv2.flip(image, flipCode)
        return flip
        
    def __getitem__(self, index):
        image_path = self.imgs_path[index]
        path = os.path.dirname(image_path)
       
        name =image_path.split('/')[-1][0:3]
       
        image_path = path +'/'+ name +'.png'
       
     
        # 根据image_path生成label_path
        label_path =path +'/'+ name+"_mask.png"
    
        # 读取训练图片和标签图片
        
        image = cv2.imread(image_path, 0)
        label = cv2.imread(label_path, 0)
      
        # 将数据转为单通道的图片
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # label = cv2.cvtColor(label, cv2.COLOR_BGR2GRAY)
        image = image.reshape(1, image.shape[0], image.shape[1])
        label = label.reshape(1, label.shape[0], label.shape[1])
       

        # 处理标签，将像素值为255的改为1
        if label.max() > 1:
            label = label / 255
       
        # 随机进行数据增强，为2时不做处理
        # flipCode = random.choice([-1, 0, 1, 2])
        # if flipCode != 2:
        #     image = self.augment(image, flipCode)
        #     label = self.augment(label, flipCode)
        #     edge = self.augment(edge, flipCode)
        return image, label

    def __len__(self):
        # 返回训练集大小
        return len(self.imgs_path)












class ISIC_Loader(Dataset):
    def __init__(self, data_path):
        self.data_path = data_path
        self.imgs_path = glob.glob(os.path.join(data_path, 'image/*.jpg'))

    def augment(self, image, flipCode):
        flip = cv2.flip(image, flipCode)
        return flip
        
    def __getitem__(self, index):
        image_path = self.imgs_path[index]
        label_path = image_path.replace('image', 'label').split('.')[0]+"_Segmentation.png"
        edge_path = image_path.replace('image', 'label').split('.')[0]+"_Segmentation_edge.png"

        image = cv2.imread(image_path)
        label = cv2.imread(label_path, 0)
        edge = cv2.imread(edge_path, 0)
      
        image = image.reshape(3, image.shape[0], image.shape[1])
        label = label.reshape(1, label.shape[0], label.shape[1])
        edge = edge.reshape(1, edge.shape[0], edge.shape[1])
        label = label / 255
        edge = edge / 255
        flipCode = random.choice([-1, 0, 1, 2])
        if flipCode != 2:
            image = self.augment(image, flipCode)
            label = self.augment(label, flipCode)
            edge = self.augment(edge, flipCode)
        return image, label, edge

    def __len__(self):
        return len(self.imgs_path)


class ISIC_Loadertest(Dataset):
    def __init__(self, data_path):
        self.data_path = data_path
        self.imgs_path = glob.glob(os.path.join(data_path, 'image/*.jpg'))

    def augment(self, image, flipCode):
        flip = cv2.flip(image, flipCode)
        return flip
        
    def __getitem__(self, index):
        image_path = self.imgs_path[index]
        label_path = image_path.replace('image', 'label').split('.')[0]+"_Segmentation.png"
        edge_path = image_path.replace('image', 'label').split('.')[0]+"_Segmentation_edge.png"

        image = cv2.imread(image_path)
        label = cv2.imread(label_path, 0)
        edge = cv2.imread(edge_path, 0)
      
        image = image.reshape(3, image.shape[0], image.shape[1])
        label = label.reshape(1, label.shape[0], label.shape[1])
       
        label = label / 255
       
        
        return image, label

    def __len__(self):
        return len(self.imgs_path)


class Lung_Loader(Dataset):
    def __init__(self, data_path):
        self.data_path = data_path
        self.imgs_path = glob.glob(os.path.join(data_path, 'images/*.tif'))

    def augment(self, image, flipCode):
        flip = cv2.flip(image, flipCode)
        return flip
        
    def __getitem__(self, index):
        image_path = self.imgs_path[index]
        label_path = image_path.replace('images', 'masks').split('.')[0]+".tif"
        edge_path = image_path.replace('images', 'edge').split('.')[0]+"_edge.png"

        image = cv2.imread(image_path,0)
        label = cv2.imread(label_path, 0)
        edge = cv2.imread(edge_path, 0)
      
        image = image.reshape(1, image.shape[0], image.shape[1])
        label = label.reshape(1, label.shape[0], label.shape[1])
        edge = edge.reshape(1, edge.shape[0], edge.shape[1])
        # 处理标签，将像素值为255的改为1
        if label.max() > 1:
            label = label / 255
        if edge.max() > 1:
            edge = edge / 255
        # 随机进行数据增强，为2时不做处理
        flipCode = random.choice([-1, 0, 1, 2])
        if flipCode != 2:
            image = self.augment(image, flipCode)
            label = self.augment(label, flipCode)
            edge = self.augment(edge, flipCode)
    
        return image, label, edge

    def __len__(self):
        return len(self.imgs_path)


class Lung_Loadertest(Dataset):
    def __init__(self, data_path):
        self.data_path = data_path
        self.imgs_path = glob.glob(os.path.join(data_path, 'images/*.tif'))

    def augment(self, image, flipCode):
        flip = cv2.flip(image, flipCode)
        return flip
        
    def __getitem__(self, index):
        image_path = self.imgs_path[index]
        label_path = image_path.replace('images', 'masks').split('.')[0]+".tif"

        image = cv2.imread(image_path,0)
        label = cv2.imread(label_path, 0)
     
        image = image.reshape(1, image.shape[0], image.shape[1])
        label = label.reshape(1, label.shape[0], label.shape[1])
       
        if label.max() > 1:

            label = label / 255  
        return image, label

    def __len__(self):
        return len(self.imgs_path)




class CXR(Dataset):
    def __init__(self, data_path):
        self.data_path = data_path
        self.imgs_path = glob.glob(os.path.join(data_path, 'image/*.png'))

    def augment(self, image, flipCode):
        flip = cv2.flip(image, flipCode)
        return flip
        
    def __getitem__(self, index):
        image_path = self.imgs_path[index]
        label_path = image_path.replace('image', 'mask').split('.')[0]+".png"
        edge_path = image_path.replace('image', 'Edge').split('.')[0]+".png"

        image = cv2.imread(image_path,0)
        label = cv2.imread(label_path, 0)
        edge = cv2.imread(edge_path, 0)
      
        image = image.reshape(1, image.shape[0], image.shape[1])
        label = label.reshape(1, label.shape[0], label.shape[1])
        edge = edge.reshape(1, edge.shape[0], edge.shape[1])
        # 处理标签，将像素值为255的改为1
        if label.max() > 1:
            label = label / 255
        if edge.max() > 1:
            edge = edge / 255
        # 随机进行数据增强，为2时不做处理
        flipCode = random.choice([-1, 0, 1, 2])
        if flipCode != 2:
            image = self.augment(image, flipCode)
            label = self.augment(label, flipCode)
            edge = self.augment(edge, flipCode)
    
        return image, label, edge

    def __len__(self):
        return len(self.imgs_path)


class CXRtest(Dataset):
    def __init__(self, data_path):
        self.data_path = data_path
        self.imgs_path = glob.glob(os.path.join(data_path, 'image/*.png'))

    def augment(self, image, flipCode):
        flip = cv2.flip(image, flipCode)
        return flip
        
    def __getitem__(self, index):
        image_path = self.imgs_path[index]
        label_path = image_path.replace('image', 'mask').split('.')[0]+".png"

        image = cv2.imread(image_path,0)
        label = cv2.imread(label_path, 0)
     
        image = image.reshape(1, image.shape[0], image.shape[1])
        label = label.reshape(1, label.shape[0], label.shape[1])
       
        if label.max() > 1:

            label = label / 255  
        return image, label

    def __len__(self):
        return len(self.imgs_path)















class JZX_Loader(Dataset):
    def __init__(self, data_path):
        self.data_path = data_path
        self.imgs_path = glob.glob(os.path.join(data_path, 'image/*.bmp'))

    def augment(self, image, flipCode):
        flip = cv2.flip(image, flipCode)
        return flip
        
    def __getitem__(self, index):
        image_path = self.imgs_path[index]
        label_path = image_path.replace('image', 'GT').split('.')[0]+".bmp"
        edge_path = image_path.replace('image', 'Edge').split('.')[0]+"_edge.png"

        image = cv2.imread(image_path,0)
        label = cv2.imread(label_path, 0)
        edge = cv2.imread(edge_path, 0)
      
        image = image.reshape(1, image.shape[0], image.shape[1])
        label = label.reshape(1, label.shape[0], label.shape[1])
        edge = edge.reshape(1, edge.shape[0], edge.shape[1])
        label = label / 255
        edge = edge / 255
        flipCode = random.choice([-1, 0, 1, 2])
        if flipCode != 2:
            image = self.augment(image, flipCode)
            label = self.augment(label, flipCode)
            edge = self.augment(edge, flipCode)
       

        return image, label, edge

    def __len__(self):
        return len(self.imgs_path)


class JZX_Loaderval(Dataset):
    def __init__(self, data_path):
        self.data_path = data_path
        self.imgs_path = glob.glob(os.path.join(data_path, 'image/*.bmp'))

    def augment(self, image, flipCode):
        flip = cv2.flip(image, flipCode)
        return flip
        
    def __getitem__(self, index):
        image_path = self.imgs_path[index]
        label_path = image_path.replace('image', 'GT').split('.')[0]+".bmp"
       

        image = cv2.imread(image_path,0)
        label = cv2.imread(label_path, 0)
       
      
        image = image.reshape(1, image.shape[0], image.shape[1])
        label = label.reshape(1, label.shape[0], label.shape[1])
       
        label = label / 255
        
       
        return image, label

    def __len__(self):
        return len(self.imgs_path)                              


class COVD(Dataset):
    def __init__(self, data_path):
        # 初始化函数，读取所有data_path下的图片
        self.data_path = data_path
        self.imgs_path = glob.glob(os.path.join(data_path, 'Imgs/*.jpg'))
       
    def augment(self, image, flipCode):
        # 使用cv2.flip进行数据增强，filpCode为1水平翻转，0垂直翻转，-1水平+垂直翻转
        flip = cv2.flip(image, flipCode)
        return flip
        
    def __getitem__(self, index):
        # 根据index读取图片
        image_path = self.imgs_path[index]
        # 根据image_path生成label_path
        label_path = image_path.replace('Imgs', 'GT').split('.')[0]+".png"
        edge_path = image_path.replace('Imgs', 'Edge').split('.')[0]+".png"
        # 读取训练图片和标签图片
        
        image = cv2.imread(image_path, 0)
        label = cv2.imread(label_path, 0)
        edge = cv2.imread(edge_path, 0)
       
        # 将数据转为单通道的图片
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # label = cv2.cvtColor(label, cv2.COLOR_BGR2GRAY)
        image = image.reshape(1, image.shape[0], image.shape[1])
        label = label.reshape(1, label.shape[0], label.shape[1])
        edge = edge.reshape(1, edge.shape[0], edge.shape[1])
           

        # 处理标签，将像素值为255的改为1
        if label.max() > 1:
            label = label / 255
        if edge.max() > 1:
            edge = edge / 255
        # 随机进行数据增强，为2时不做处理
        flipCode = random.choice([-1, 0, 1, 2])
        if flipCode != 2:
            image = self.augment(image, flipCode)
            label = self.augment(label, flipCode)
            edge = self.augment(edge, flipCode)
    
        return image, label, edge

    def __len__(self):
        # 返回训练集大小
        return len(self.imgs_path)

class COVDtest(Dataset):
    def __init__(self, data_path):
        # 初始化函数，读取所有data_path下的图片
        self.data_path = data_path
        self.imgs_path = glob.glob(os.path.join(data_path, 'Imgs/*.jpg'))
        
    def augment(self, image, flipCode):
        # 使用cv2.flip进行数据增强，filpCode为1水平翻转，0垂直翻转，-1水平+垂直翻转
        flip = cv2.flip(image, flipCode)
        return flip
        
    def __getitem__(self, index):
        # 根据index读取图片
        image_path = self.imgs_path[index]
        # 根据image_path生成label_path
        label_path = image_path.replace('Imgs', 'GT').split('.')[0]+".png"
        edge_path = image_path.replace('Imgs', 'Edge').split('.')[0]+".png"
        # 读取训练图片和标签图片
        
        image = cv2.imread(image_path, 0)
        label = cv2.imread(label_path, 0)
        # edge = cv2.imread(edge_path, 0)
       
        # 将数据转为单通道的图片
      
        # 将数据转为单通道的图片
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # label = cv2.cvtColor(label, cv2.COLOR_BGR2GRAY)
        image = image.reshape(1, image.shape[0], image.shape[1])
        label = label.reshape(1, label.shape[0], label.shape[1])
        # edge = edge.reshape(1, edge.shape[0], edge.shape[1])


        # 处理标签，将像素值为255的改为1
        if label.max() > 1:
            label = label / 255
        # if edge.max() > 1:
        #     edge = edge / 255
        # 随机进行数据增强，为2时不做处理
        # flipCode = random.choice([-1, 0, 1, 2])
        # if flipCode != 2:
        #     image = self.augment(image, flipCode)
        #     label = self.augment(label, flipCode)
        #     edge = self.augment(edge, flipCode)
    
        return image, label

    def __len__(self):
        # 返回训练集大小
        return len(self.imgs_path)



###########################一个标准的的写法#############
import torch.utils.data as data
import PIL.Image as Image
import os
 
 
def make_dataset(root):
    imgs = []
    n = len(os.listdir(root)) // 2  #因为数据集中一套训练数据包含有训练图和mask图，所以要除2
    for i in range(n):
        img = os.path.join(root, "%03d.png" % i)
        mask = os.path.join(root, "%03d_mask.png" % i)
        imgs.append((img, mask))
    return imgs
 
 
class LiverDataset(data.Dataset):
    def __init__(self, root, transform=None, target_transform=None):
        imgs = make_dataset(root)
        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform
 
    def __getitem__(self, index):
        x_path, y_path = self.imgs[index]
        origin_x = Image.open(x_path)
        origin_y = Image.open(y_path)
        if self.transform is not None:
            img_x = self.transform(origin_x)
        if self.target_transform is not None:
            img_y = self.target_transform(origin_y)
        return img_x, img_y
 
    def __len__(self):
        return len(self.imgs)




class Lung1_Loader(Dataset):
    def __init__(self, data_path, transform=None, target_transform=None):
        self.data_path = data_path
        self.imgs_path = glob.glob(os.path.join(data_path, 'images/*.tif'))
        self.transform =transform
        self.target_transform =target_transform
        
    def __getitem__(self, index):
        image_path = self.imgs_path[index]
        label_path = image_path.replace('images', 'masks').split('.')[0]+".tif"
        edge_path = image_path.replace('images', 'edge').split('.')[0]+"_edge.png"
        image = Image.open(image_path)
        label = Image.open(label_path)
        edge =Image.open(edge_path)
        if self.transform is not None:
            image = self.transform(image)
        if self.target_transform is not None:
            label = self.target_transform(label)
            edge = self.target_transform(edge)
        return image, label, edge

    def __len__(self):
        return len(self.imgs_path)

class Lung1_Loadertest(Dataset):
    def __init__(self, data_path, transform=None, target_transform=None):
        self.data_path = data_path
        self.imgs_path = glob.glob(os.path.join(data_path, 'images/*.tif'))
        self.transform =transform
        self.target_transform =target_transform
        
    def __getitem__(self, index):
        image_path = self.imgs_path[index]
        label_path = image_path.replace('images', 'masks').split('.')[0]+".tif"
        
        image = Image.open(image_path)
        
        label = Image.open(label_path)
      
        if self.transform is not None:
            image = self.transform(image)
        if self.target_transform is not None:
            label = self.target_transform(label)
         
        return image, label

    def __len__(self):
        return len(self.imgs_path)





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
        label_path1 = image_path.replace('image', 'body-origin').split('.')[0]+"_mask.png"
        label_path2 = image_path.replace('image', 'detail-origin').split('.')[0]+"_mask.png"
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
        # label_path1 = image_path.replace('image', 'body-origin').split('.')[0]+"_mask.png"
        # label_path2 = image_path.replace('image', 'detail-origin').split('.')[0]+"_mask.png"
        # edge_path = image_path.replace('image', 'label').split('.')[0]+"_edge.png"
        # 读取训练图片和标签图片
        
        image = cv2.imread(image_path, 0)
        label = cv2.imread(label_path, 0)
        # label1 = cv2.imread(label_path1, 0)
        # label2 = cv2.imread(label_path2, 0)
        # edge = cv2.imread(edge_path, 0)
      
        # 将数据转为单通道的图片
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # label = cv2.cvtColor(label, cv2.COLOR_BGR2GRAY)
        image = image.reshape(1, image.shape[0], image.shape[1])
        label = label.reshape(1, label.shape[0], label.shape[1])
        # label1 = label1.reshape(1, label1.shape[0], label1.shape[1])
        # label2 = label2.reshape(1, label2.shape[0], label2.shape[1])
        # edge = edge.reshape(1, edge.shape[0], edge.shape[1])


        # 处理标签，将像素值为255的改为1
        if label.max() > 1:
            label = label / 255
            # label1 = label1 / 255
            # label2 = label2 / 255
        # if edge.max() > 1:
        #     edge = edge / 255
        # 随机进行数据增强，为2时不做处理
        # flipCode = random.choice([-1, 0, 1, 2])
        # if flipCode != 2:
        #     image = self.augment(image, flipCode)
        #     label = self.augment(label, flipCode)
        #     label1 = self.augment(label1, flipCode)
        #     label2 = self.augment(label2, flipCode)
            # edge = self.augment(edge, flipCode)
    
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
        # print(self.input_paths[1])
    
    def augment(self, image, flipCode):
        flip = cv2.flip(image, flipCode)
        return flip
    def __getitem__(self, index):
        
        

        image_path = self.input_paths[index]
        # print(image_path)
        
        label_path = image_path.replace('image', 'mask').split('.')[0]+".png"
        # print(label_path)
        
        label_path1 = image_path.replace('image', 'Edge').split('.')[0]+"_edge.png"
       
       
        
        image = Image.open(image_path).convert('RGB')
        
        label = Image.open(label_path).convert('L')
        label1 = Image.open(label_path1).convert('L')
       
        image = self.img_resize(image)
        label = self.label_resize(label)
        label1 = self.label_resize(label1)
      
        #
        
        image = self.img_transform(image)
        # print(image.shape)
        # image_hsv = self.hsv_transform(image_hsv)
        # image = torch.cat([image,image_hsv],0)
        
        label = self.label_transform(label)
        label1 = self.label_transform(label1)
      
        return image, label, label1,  
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
    
        image = Image.open(image_path).convert('RGB')
        label = Image.open(label_path).convert('L')
      
        image = self.img_resize(image)
        label = self.label_resize(label)
     
       
        
        image = self.img_transform(image)
        # image_hsv = self.hsv_transform(image_hsv)
        # image = torch.cat([image,image_hsv],0)
        
        label = self.label_transform(label)
        
        return image, label, image_path
    def __len__(self):

        return len(self.input_paths)        






############新定义########
class XSDataset1(torch.utils.data.Dataset):

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
        
        label_path1 = image_path.replace('image', 'Edge').split('.')[0]+"_edge.png"
       
       
        
        image = Image.open(image_path).convert('RGB')
        label = Image.open(label_path).convert('L')
        label1 = Image.open(label_path1).convert('L')
       
        image = self.img_resize(image)
        label = self.label_resize(label)
        label1 = self.label_resize(label1)

        df = direct_field(label.numpy()[0], norm=self.df_norm)
      
        #
        
        image = self.img_transform(image)
        # image_hsv = self.hsv_transform(image_hsv)
        # image = torch.cat([image,image_hsv],0)
        
        label = self.label_transform(label)
        label1 = self.label_transform(label1)
      
        return image, label, label1,  
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
    
        image = Image.open(image_path).convert('RGB')
        label = Image.open(label_path).convert('L')

      
        image = self.img_resize(image)
        label = self.label_resize(label)

        # mask1 = cv2.imread(label_path, 0)
        # mask1 = cv2.resize(mask1, (352,352), interpolation=cv2.INTER_AREA) 
        # mask1 = torch.from_numpy(mask1)
        # print(mask1.S)
     
       
        
        image = self.img_transform(image)
        # image_hsv = self.hsv_transform(image_hsv)
        # image = torch.cat([image,image_hsv],0)
        
        label = self.label_transform(label)
        # print(label.shape)
        
        return image, label, image_path
    def __len__(self):

        return len(self.input_paths)       










# if __name__ == "__main__":
#     isbi_dataset = ISBI_Loader("/home/wangkun/shape-attentive-unet/data/train_96")
#     print("数据个数：", len(isbi_dataset))
#     train_loader = torch.utils.data.DataLoader(dataset=isbi_dataset,
#                                                batch_size=4, 
#                                                shuffle=True)
#     for image, label, edge in train_loader:
#         print(image.shape)
    