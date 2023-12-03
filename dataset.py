import os
import cv2
import ast
import torch
import numpy as np
import random
from torch.utils.data import DataLoader, Dataset

cv2.setNumThreads(1)    #设置OpenCv的线程数设置为1，避免多线程处理时的冲突问题
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  #检查cuda是否可用
class VimeoDataset(Dataset):
    def __init__(self, dataset_name, batch_size=32):
        '''
        :param dataset_name:数据集的名称
        :param batch_size: 批的数量大小
        '''
        self.batch_size = batch_size
        self.dataset_name = dataset_name        
        self.h = 256
        self.w = 448   #设置图像的高度和宽度
        self.data_root = r'D:\vimeo_triplet'  #数据集的根目录
        self.image_root = os.path.join(self.data_root, 'sequences')
        train_fn = os.path.join(self.data_root, 'tri_trainlist.txt')
        test_fn = os.path.join(self.data_root, 'tri_testlist.txt')  #这些列表包含了数据集中每个视频序列的文件路径。
        with open(train_fn, 'r') as f:
            self.trainlist = f.read().splitlines()  #
        with open(test_fn, 'r') as f:
            self.testlist = f.read().splitlines()   
        self.load_data()

    def __len__(self):
        return len(self.meta_data)

    def load_data(self):
        cnt = int(len(self.trainlist) * 0.95)  #使用95%的数据进行训练，5%的数据进行验证。
        if self.dataset_name == 'train':
            self.meta_data = self.trainlist[:cnt]
        elif self.dataset_name == 'test':
            self.meta_data = self.testlist
        else:
            self.meta_data = self.trainlist[cnt:]
           
    def crop(self, img0, gt, img1, h, w):  #数据增强函数，随机裁剪三张图片img0,gt,img1到指定的高度和宽度，用于数据增强。
        ih, iw, _ = img0.shape
        x = np.random.randint(0, ih - h + 1)
        y = np.random.randint(0, iw - w + 1)
        img0 = img0[x:x+h, y:y+w, :]
        img1 = img1[x:x+h, y:y+w, :]
        gt = gt[x:x+h, y:y+w, :]
        return img0, gt, img1

    def getimg(self, index):        #用于从数据集中，取出三张图片，通常是连续的帧。
        imgpath = os.path.join(self.image_root, self.meta_data[index])
        imgpaths = [imgpath + '/im1.png', imgpath + '/im2.png', imgpath + '/im3.png']

        # Load images
        img0 = cv2.imread(imgpaths[0])
        gt = cv2.imread(imgpaths[1])
        img1 = cv2.imread(imgpaths[2])
        timestep = 0.5
        return img0, gt, img1, timestep
    
        # RIFEm with Vimeo-Septuplet            使用更大的数据集Video-Septuplet
        # imgpaths = [imgpath + '/im1.png', imgpath + '/im2.png', imgpath + '/im3.png', imgpath + '/im4.png', imgpath + '/im5.png', imgpath + '/im6.png', imgpath + '/im7.png']
        # ind = [0, 1, 2, 3, 4, 5, 6]
        # random.shuffle(ind)
        # ind = ind[:3]
        # ind.sort()
        # img0 = cv2.imread(imgpaths[ind[0]])
        # gt = cv2.imread(imgpaths[ind[1]])
        # img1 = cv2.imread(imgpaths[ind[2]])        
        # timestep = (ind[1] - ind[0]) * 1.0 / (ind[2] - ind[0] + 1e-6)
            
    def __getitem__(self, index):        
        img0, gt, img1, timestep = self.getimg(index)
        if self.dataset_name == 'train':  # 若为训练集
            img0, gt, img1 = self.crop(img0, gt, img1, 224, 224) #数据增强函数
            if random.uniform(0, 1) < 0.5:  #以50%的概率翻转所有图像
                img0 = img0[:, :, ::-1]
                img1 = img1[:, :, ::-1]
                gt = gt[:, :, ::-1]
            if random.uniform(0, 1) < 0.5:
                img0 = img0[::-1]
                img1 = img1[::-1]
                gt = gt[::-1]
            if random.uniform(0, 1) < 0.5:
                img0 = img0[:, ::-1]
                img1 = img1[:, ::-1]
                gt = gt[:, ::-1]
            if random.uniform(0, 1) < 0.5:  #交换顺序，t变成1-t，倒放
                tmp = img1
                img1 = img0
                img0 = tmp
                timestep = 1 - timestep
            # random rotation
            p = random.uniform(0, 1)
            if p < 0.25:#将图像进行顺时针旋转90度，180度和反向旋转90度
                img0 = cv2.rotate(img0, cv2.ROTATE_90_CLOCKWISE)
                gt = cv2.rotate(gt, cv2.ROTATE_90_CLOCKWISE)
                img1 = cv2.rotate(img1, cv2.ROTATE_90_CLOCKWISE)
            elif p < 0.5:
                img0 = cv2.rotate(img0, cv2.ROTATE_180)
                gt = cv2.rotate(gt, cv2.ROTATE_180)
                img1 = cv2.rotate(img1, cv2.ROTATE_180)
            elif p < 0.75:
                img0 = cv2.rotate(img0, cv2.ROTATE_90_COUNTERCLOCKWISE)
                gt = cv2.rotate(gt, cv2.ROTATE_90_COUNTERCLOCKWISE)
                img1 = cv2.rotate(img1, cv2.ROTATE_90_COUNTERCLOCKWISE)
        #将OpenCv读取到的NumPy数组转换为张量，并调整通道顺序。OpenCv读取到的是BGR格式，PyTorch读取到的是RGB格式。
        img0 = torch.from_numpy(img0.copy()).permute(2, 0, 1)
        img1 = torch.from_numpy(img1.copy()).permute(2, 0, 1)
        gt = torch.from_numpy(gt.copy()).permute(2, 0, 1)
        # 将时间步长换为Pytorch张量
        timestep = torch.tensor(timestep).reshape(1, 1, 1)
        return torch.cat((img0, img1, gt), 0), timestep  #返回的是一个包含处理过的img0,img1,gt的合并张量，以及时间步长
