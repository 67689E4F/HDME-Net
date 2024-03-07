from torchvision.transforms import ToTensor#用于把图片转化为张量
import numpy as np#用于将张量转化为数组，进行除法
from torchvision.datasets import ImageFolder#用于导入图片数据集

means = [0,0,0]
std = [0,0,0]#初始化均值和方差
transform=ToTensor()#可将图片类型转化为张量，并把0~255的像素值缩小到0~1之间
dataset=ImageFolder("/mnt/sdb1/67689e4f/dataset/coco_shuffle_cocwithnoise_05/image/",transform=transform)#导入数据集的图片，并且转化为张量
num_imgs=len(dataset)#获取数据集的图片数量
for img,a in dataset:#遍历数据集的张量和标签
    for i in range(3):#遍历图片的RGB三通道
        # 计算每一个通道的均值和标准差
        means[i] += img[i, :, :].mean()
        std[i] += img[i, :, :].std()
mean=np.array(means)/num_imgs
std=np.array(std)/num_imgs#要使数据集归一化，均值和方差需除以总图片数量
print(mean,std)#打印出结果

