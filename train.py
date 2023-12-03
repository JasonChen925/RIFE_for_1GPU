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

epoch = 10
batch_size = 32

def get_learning_rate(step):
    if step < 2000:
        mul = step / 2000.
        return 3e-4 * mul
    else:
        mul = np.cos((step - 2000) / (epoch * step_per_epoch - 2000.) * math.pi) * 0.5 + 0.5
        return (3e-4 - 3e-6) * mul + 3e-6

def flow2rgb(flow_map_np):  #将一个光流向量场转为一种可以直观，可视化的RGB格式
    h, w, _ = flow_map_np.shape  #获取flow_map_np的形状，高度h，宽度w，颜色通道数这里并不使用
    rgb_map = np.ones((h, w, 3)).astype(np.float32) #创建一个全1数组，用于存储转换后的RGB图像
    normalized_flow_map = flow_map_np / (np.abs(flow_map_np).max())  #归一化光流图的值，规范到[0,1]范围内，flow_map_np除以改图中绝对值的最大值
    #normalized_flow_map是归一化的光流图
    rgb_map[:, :, 0] += normalized_flow_map[:, :, 0] #将归一化的光流图的第一个通道（通常表示水平方向的流动）加到红色通道上。
    rgb_map[:, :, 1] -= 0.5 * (normalized_flow_map[:, :, 0] + normalized_flow_map[:, :, 1]) #从绿色通道减去归一化光流图的两个通道的平均值的一半。这种处理有助于在 RGB 图像中可视化光流的方向和大小。
    rgb_map[:, :, 2] += normalized_flow_map[:, :, 1]  #更新 rgb_map 的蓝色通道，将归一化的光流图的第二个通道（通常表示垂直方向的流动）加到蓝色通道上。
    return rgb_map.clip(0, 1)   #返回rgb_map中的图像，其中所有值限制在[0,1]的范围内

def train(model):   #model为训练模型，local_rank为分布式训练的本地排名
# def train(model, local_rank):  # model为训练模型，local_rank为分布式训练的本地排名
#     if local_rank == 0:
        #根据local_rank的值初始化了TensorBoard的日志记录器“SummaryWritter"。如果"local_rank"为0（通常表示主节点），
        # 则在train和validate目录下创建日志记录器，否则不记录
    writer = SummaryWriter('train')
    writer_val = SummaryWriter('validate')
#SummaryWriter 类的作用是将训练或验证过程中的数据（如损失值、准确率、模型参数、图像等）记录下来，然后可以使用 TensorBoard 工具进行可视化。
# 这对于监控模型训练过程、调试和比较不同训练运行非常有用。
    # else:
    #     writer = None
    #     writer_val = None

    step = 0  #训练步骤计数器
    nr_eval = 0  #评估计数器
    dataset = VimeoDataset('train')  #创建一个数据集Vimeo
    # sampler = DistributedSampler(dataset)   #采用分布式采样器DistributedSample

    train_data = DataLoader(dataset, batch_size=batch_size, num_workers=8, pin_memory=True, drop_last=True)
    # train_data = DataLoader(dataset, batch_size=args.batch_size, num_workers=8, pin_memory=True, drop_last=True, sampler=sampler)
    #num_works=8  8个进程并行的处理数据
    #pin_memory=True Dataloader再返回之前，将数据张量复制到CUDA固定内存之中，有利于数据更快地转移到GPU中
    #数据集不满足批次整除时，丢弃最后一个数据集
    #sample  指的是从数据集中抽取数据的策略，这里指的是DistributeSample
    # args.step_per_epoch = train_data.__len__()#计算每个epoch的步数并且保存到args.step_per_epoch
    step_per_epoch = train_data.__len__()

    #加载一个验证dataset_val和val_data
    dataset_val = VimeoDataset('validation')
    val_data = DataLoader(dataset_val, batch_size=16, pin_memory=True, num_workers=8)

    print('training...')
    time_stamp = time.time()  #记录时间戳，用于计算数据加载和训练的时间。

    for epoc in range(epoch):
        # sampler.set_epoch(epoch)  #在分布式训练中设置当前的epoch，以确保数据的随机性
        for i, data in enumerate(train_data):  #开始遍历train_data(Dataloader)中的批次

            data_time_interval = time.time() - time_stamp  #计算数据的时间间隔
            time_stamp = time.time()

            data_gpu, timestep = data  #dataset().__getitem__返回的是一个包含处理过的img0,img1,gt的合并张量，以及时间步长
            data_gpu = data_gpu.to(device, non_blocking=True) / 255.
            timestep = timestep.to(device, non_blocking=True)      #将数据和时间步长移动到gpu上，并使图像数据归一化

            imgs = data_gpu[:, :6]
            gt = data_gpu[:, 6:9]   #从数据中提取输入图像和真值

            learning_rate = get_learning_rate(step) * args.world_size / 4       #计算当前步骤的学习率

            pred, info = model.update(imgs, gt, learning_rate, training=True) # pass timestep if you are training RIFEm
            #使用模型进行一次训练迭代，并获取预测结果和其他信息

            train_time_interval = time.time() - time_stamp
            time_stamp = time.time()   #计算训练一个批次的时间间隔

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

                for i in range(5): #将前五个样本，将融合图像，预测图像和真值图像合并，转换光流为RGB格式，然后通过writter记录到TensorBoard
                    imgs = np.concatenate((merged_img[i], pred[i], gt[i]), 1)[:, :, ::-1]
                    writer.add_image(str(i) + '/img', imgs, step, dataformats='HWC')
                    writer.add_image(str(i) + '/flow', np.concatenate((flow2rgb(flow0[i]), flow2rgb(flow1[i])), 1), step, dataformats='HWC')
                    writer.add_image(str(i) + '/mask', mask[i], step, dataformats='HWC')
                writer.flush()    #确保所有的数据都被写入

            # if local_rank == 0:   #如果是主节点，那么打印出当前epoch的信息，包括epoch数、步数，数据加载时间，训练时间和L1损失
            #     print('epoch:{} {}/{} time:{:.2f}+{:.2f} loss_l1:{:.4e}'.format(epoch, i, args.step_per_epoch, data_time_interval, train_time_interval, info['loss_l1']))
            step += 1#截止此处，step+1

        nr_eval += 1  #增加评估计数器nr_eval，每五次训练，对模型进行一次评估
        if nr_eval % 5 == 0:
            evaluate(model, val_data, step, writer_val)
            # evaluate(model, val_data, step, local_rank, writer_val)

        # model.save_model(log_path, local_rank)  #保存模型到指定的路径log_path
        model.save_model(log_path)
        # dist.barrier()  #在分布式训练中，使用dist.barrier()来同步不同进程的状态，确保所有进程在进行下一个epoch之前都完成了当前epoch的工作

def evaluate(model, val_data, nr_eval, writer_val):
    '''

    :param model: 模型
    :param val_data: 验证数据
    :param nr_eval: 评估次数
    :param local_rank: 本地排名
    :param writer_val: 用于记录验证结果的TensorBoard写入器
    :return:
    '''
    loss_l1_list = []
    loss_distill_list = []
    loss_tea_list = []
    psnr_list = []
    psnr_list_teacher = []   #初始胡各种损失和评价指标

    time_stamp = time.time()  #记录当前时间，用于计算评估所需的时间

    for i, data in enumerate(val_data):  #遍历验证数据集中的所有批次
        data_gpu, timestep = data
        data_gpu = data_gpu.to(device, non_blocking=True) / 255. #将数据移动到GPU，并且将图像数据归一化。

        imgs = data_gpu[:, :6]
        gt = data_gpu[:, 6:9]       #从数据中提取输入图像和真值(groud truth)

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

        #将预测结果、真值、融合图像和光流从 GPU 移至 CPU，并转换为适合可视化的格式。
        gt = (gt.permute(0, 2, 3, 1).cpu().numpy() * 255).astype('uint8')
        pred = (pred.permute(0, 2, 3, 1).cpu().numpy() * 255).astype('uint8')
        merged_img = (merged_img.permute(0, 2, 3, 1).cpu().numpy() * 255).astype('uint8')
        flow0 = info['flow'].permute(0, 2, 3, 1).cpu().numpy()
        flow1 = info['flow_tea'].permute(0, 2, 3, 1).cpu().numpy()

        # if i == 0 and local_rank == 0:#如果是第一个批次，并且 local_rank 为 0，则记录前 10 个样本的图像、预测结果和光流到 TensorBoard。
        if i == 0:
            for j in range(10):
                imgs = np.concatenate((merged_img[j], pred[j], gt[j]), 1)[:, :, ::-1]
                writer_val.add_image(str(j) + '/img', imgs.copy(), nr_eval, dataformats='HWC')
                writer_val.add_image(str(j) + '/flow', flow2rgb(flow0[j][:, :, ::-1]), nr_eval, dataformats='HWC')
    
    eval_time_interval = time.time() - time_stamp  #计算评估所需要的总时间

    # if local_rank != 0:     #如果不是主节点，直接返回不执行后面的操作
    #     return

    #在主节点上，使用writer_val记录平均PSNR值和老师模型的平均PSNR值
    writer_val.add_scalar('psnr', np.array(psnr_list).mean(), nr_eval)
    writer_val.add_scalar('psnr_teacher', np.array(psnr_list_teacher).mean(), nr_eval)
        
if __name__ == "__main__":    
    # parser = argparse.ArgumentParser()      #创建一个ArgumentParser对象用用解析命令行参数
    #     #使用 add_argument 方法为脚本添加了几个参数：
    #     # epoch（训练周期数）
    #     # batch_size（每批数据的大小）
    #     # local_rank（用于分布式训练的本地排名）
    #     # world_size（分布式训练的世界大小，即参与训练的总进程数）
    # parser.add_argument('--epoch', default=300, type=int)
    # parser.add_argument('--batch_size', default=16, type=int, help='minibatch size')
    # # parser.add_argument('--local_rank', default=0, type=int, help='local rank')
    # parser.add_argument('--world_size', default=4, type=int, help='world size')
    epoch = 10
    batch_size =32

    # args = parser.parse_args()  #解析命令行参数，并将结果保存到args

    # torch.distributed.init_process_group(backend="nccl", world_size=args.world_size)   #初始化Pytorch的分布式训练环境，使用NCCL作为后端，设置参与训练的进程总数

    # torch.cuda.set_device(args.local_rank)      #根据local_rank决定使用哪个GPU
    seed = 1234
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)#设置随机生成器的种子，确保实验的可重复性

    torch.backends.cudnn.benchmark = True       #启用cuDNN的自动调优功能，这可以提高深度学习训练的性能，特别是输入大小不变时。
    model = Model()              #创建模型实例，接受local_rank作为参数
    train(model)               #使用train函数开始训练模型
        
