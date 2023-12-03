import os
import cv2
import math
import time
import torch
import torch.distributed as dist
import numpy as np
import random
import argparse
from model.RIFE import Model
from dataset import *
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data.distributed import DistributedSampler

device = torch.device("cuda")

log_path = r'D:\ECCV2022-RIFE-main\train_log'

def get_learning_rate(step):
    if step < 2000:
        mul = step / 2000.
        return 3e-4 * mul
    else:
        mul = np.cos((step - 2000) / (args.epoch * args.step_per_epoch - 2000.) * math.pi) * 0.5 + 0.5
        return (3e-4 - 3e-6) * mul + 3e-6

#将一个光流向量场转为一种可以直观，可视化的RGB格式
def flow2rgb(flow_map_np):
    h, w, _ = flow_map_np.shape  #获取flow_map_np的形状，高度h，宽度w，颜色通道数这里并不使用
    rgb_map = np.ones((h, w, 3)).astype(np.float32) #创建一个全1数组，用于存储转换后的RGB图像
    normalized_flow_map = flow_map_np / (np.abs(flow_map_np).max())  #归一化光流图的值，规范到[0,1]范围内，flow_map_np除以改图中绝对值的最大值
    #normalized_flow_map是归一化的光流图
    rgb_map[:, :, 0] += normalized_flow_map[:, :, 0] #将归一化的光流图的第一个通道（通常表示水平方向的流动）加到红色通道上。
    rgb_map[:, :, 1] -= 0.5 * (normalized_flow_map[:, :, 0] + normalized_flow_map[:, :, 1]) #从绿色通道减去归一化光流图的两个通道的平均值的一半。这种处理有助于在 RGB 图像中可视化光流的方向和大小。
    rgb_map[:, :, 2] += normalized_flow_map[:, :, 1]  #更新 rgb_map 的蓝色通道，将归一化的光流图的第二个通道（通常表示垂直方向的流动）加到蓝色通道上。
    return rgb_map.clip(0, 1)   #返回rgb_map中的图像，其中所有值限制在[0,1]的范围内

def train(model):

    writer = SummaryWriter('train')
    writer_val = SummaryWriter('validate')
    step = 0        #训练步骤计数器
    nr_eval = 0     #评估计数器
    dataset = VimeoDataset('train')
    train_data = DataLoader(dataset,batch_size=args.batch_size,pin_memory=True,drop_last=True)
    # train_data = DataLoader(dataset, batch_size=args.batch_size, pin_memory=True, drop_last=True)
    args.step_per_epoch = train_data.__len__()

    #加载验证集dataset_val,val_data
    dataset_val = VimeoDataset('validation')
    val_data = DataLoader(dataset_val,batch_size=32,pin_memory=True)

    print('training')
    time_stamp = time.time()

    for n in range(args.epoch):
        for i,data in enumerate(train_data):
            data_time_interval = time.time() - time_stamp
            time_stamp = time.time()

            data_gpu,timestep = data
            data_gpu = data_gpu.to(device,non_blocking=True)/255.
            timestep = timestep.to(device,non_blocking=True)

            imgs = data_gpu[:,:6]
            gt = data_gpu[:,6:9]

            learning_rate = get_learning_rate(step)

            pred,info = model.update(imgs,gt,learning_rate,training=True)

            time_stamp = time.time()

            if step % 200 == 1:
                # if step % 200 == 1 and local_rank == 0:
                '''
                如果当前步数除200余数为一，且本地排名(local rank)为0,通常为主节点，那么执行下面的代码块，目的是200步记录一次数据
                使用TensorBoard的"writter"记录当前学习率和不同类型的损失值
                '''
                writer.add_scalar('learning_rate', learning_rate, step)
                writer.add_scalar('loss/l1', info['loss_l1'], step)
                writer.add_scalar('loss/tea', info['loss_tea'], step)
                writer.add_scalar('loss/distill', info['loss_distill'], step)

            if step % 1000 == 1:
                '''
                1000步进行一次更详细的记录
                '''
                gt = (gt.permute(0, 2, 3, 1).detach().cpu().numpy() * 255).astype('uint8')
                mask = (torch.cat((info['mask'], info['mask_tea']), 3).permute(0, 2, 3, 1).detach().cpu().numpy() * 255).astype('uint8')
                pred = (pred.permute(0, 2, 3, 1).detach().cpu().numpy() * 255).astype('uint8')
                merged_img = (info['merged_tea'].permute(0, 2, 3, 1).detach().cpu().numpy() * 255).astype('uint8')
                flow0 = info['flow'].permute(0, 2, 3, 1).detach().cpu().numpy()
                flow1 = info['flow_tea'].permute(0, 2, 3, 1).detach().cpu().numpy()
                #以上的步骤是为了将模型的输出，如真值gt,预测pred,融合图像merged_img，光流flow从GPU转至CPU，并转换为可视化的格式。
                for i in range(5):  # 将前五个样本，将融合图像，预测图像和真值图像合并，转换光流为RGB格式，然后通过writter记录到TensorBoard
                    imgs = np.concatenate((merged_img[i], pred[i], gt[i]), 1)[:, :, ::-1]
                    writer.add_image(str(i) + '/img', imgs, step, dataformats='HWC')
                    writer.add_image(str(i) + '/flow', np.concatenate((flow2rgb(flow0[i]), flow2rgb(flow1[i])), 1),
                                     step, dataformats='HWC')
                    writer.add_image(str(i) + '/mask', mask[i], step, dataformats='HWC')
                writer.flush()  # 确保所有的数据都被写入
            print('epoch {} number {} train'.format(n,i))
            step += 1
        nr_eval += 1
        if nr_eval % 5 == 0:
            evaluate(model,val_data,step,writer_val)

        print('epoch:{} finished'.format(n))
        model.save_model(log_path)

def evaluate(model,val_data,nr_eval,writer_val):
    loss_l1_list = []
    loss_distill_list = []
    loss_tea_list = []
    psnr_list = []
    psnr_list_teacher = []

    time_stamp = time.time()

    for i,data in enumerate(val_data):
        data_gpu,timestep = data
        data_gpu = data_gpu.to(device, non_blocking=True) / 255.  # 将数据移动到GPU，并且将图像数据归一化。
        imgs = data_gpu[:, :6]
        gt = data_gpu[:, 6:9]  # 从数据中提取输入图像和真值(groud truth)

        with torch.no_grad():   #在不计算梯度的情况下，进行前向传播，获取预测结果和其他信息
            pred, info = model.update(imgs, gt, training=False)
            merged_img = info['merged_tea']

        loss_l1_list.append(info['loss_l1'].cpu().numpy())
        loss_tea_list.append(info['loss_tea'].cpu().numpy())
        loss_distill_list.append(info['loss_distill'].cpu().numpy())
        #计算得到的不同类型的损失值添加到相应的列表中
        for j in range(gt.shape[0]):#计算预测结果和真值之间的峰值信噪比PSNR。
            psnr = -10 * math.log10(torch.mean((gt[j] - pred[j]) * (gt[j] - pred[j])).cpu().data)
            psnr_list.append(psnr)
            psnr = -10 * math.log10(torch.mean((merged_img[j] - gt[j]) * (merged_img[j] - gt[j])).cpu().data)
            psnr_list_teacher.append(psnr)

        gt = (gt.permute(0, 2, 3, 1).cpu().numpy() * 255).astype('uint8')
        pred = (pred.permute(0, 2, 3, 1).cpu().numpy() * 255).astype('uint8')
        merged_img = (merged_img.permute(0, 2, 3, 1).cpu().numpy() * 255).astype('uint8')
        flow0 = info['flow'].permute(0, 2, 3, 1).cpu().numpy()
        flow1 = info['flow_tea'].permute(0, 2, 3, 1).cpu().numpy()

        if i == 0:
            for j in range(10):
                imgs = np.concatenate((merged_img[j], pred[j], gt[j]), 1)[:, :, ::-1]
                writer_val.add_image(str(j) + '/img', imgs.copy(), nr_eval, dataformats='HWC')
                writer_val.add_image(str(j) + '/flow', flow2rgb(flow0[j][:, :, ::-1]), nr_eval, dataformats='HWC')
    eval_time_interval = time.time() - time_stamp

    writer_val.add_scalar('psnr', np.array(psnr_list).mean(), nr_eval)
    writer_val.add_scalar('psnr_teacher', np.array(psnr_list_teacher).mean(), nr_eval)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', default=10, type=int)
    parser.add_argument('--batch_size', default=32, type=int, help='minibatch size')
    args = parser.parse_args()

    seed = 1234
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)#设置随机生成器的种子，确保实验的可重复性

    torch.backends.cudnn.benchmark = True  # 启用cuDNN的自动调优功能，这可以提高深度学习训练的性能，特别是输入大小不变时。
    model = Model()  # 创建模型实例，接受local_rank作为参数
    train(model)  # 使用train函数开始训练模型
