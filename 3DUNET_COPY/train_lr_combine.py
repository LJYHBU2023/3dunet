import os

import torch
import torch.nn as nn

import numpy as np

from torch.utils.data import DataLoader

import torch.optim as optim

import torch.optim.lr_scheduler as lr_scheduler

from tqdm import tqdm

from collections import OrderedDict



from dataset.dataset_lits_train import Train_Dataset

from dataset.dataset_lits_val import Val_Dataset

from models import ResUNet, UNet, SegNet, KiUNet_min

from utils import logger, weights_init, metrics, common, loss

import config

class DiceLoss(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, pred, target):

        # pred = pred.squeeze(dim=1)

        smooth = 1

        dice = 0.
        # dice系数的定义
        for i in range(pred.size(1)):
            dice += 2 * (pred[:,i] * target[:,i]).sum(dim=1).sum(dim=1).sum(dim=1) / (pred[:,i].pow(2).sum(dim=1).sum(dim=1).sum(dim=1) +
                                                target[:,i].pow(2).sum(dim=1).sum(dim=1).sum(dim=1) + smooth)
        # 返回的是dice距离
        dice = dice / pred.size(1)
        return torch.clamp((1 - dice).mean(), 0, 1)




def val(model, val_loader, loss_func, n_labels):
    model.eval()
    val_loss = metrics.LossAverage()
    val_dice = metrics.DiceAverage(n_labels)
    
    with torch.no_grad():
        for idx, (data, target, dist_map) in tqdm(enumerate(val_loader), total=len(val_loader)):
            data, target = data.float(), target.long()
            target = common.to_one_hot_3d(target, n_labels)
            data, target = data.to(device), target.to(device)
            
            # 如果您的模型和损失函数需要dist_map，那么保留下面的代码
            # dist_map = dist_map.to(device)
            # output = model(data, dist_map)  # 假设模型接受data和dist_map作为输入
            # loss = loss_func(output, target, dist_map)  # 假设损失函数接受output、target和dist_map作为输入

            # 如果您的模型和损失函数不需要dist_map，那么使用下面的代码
            output = model(data)
            loss = loss_func(output, target)
            
            val_loss.update(loss.item(), data.size(0))
            val_dice.update(output, target)
    
    val_log = OrderedDict({'Val_Loss': val_loss.avg, 'Val_dice_liver': val_dice.avg[1]})
    if n_labels == 3:
        val_log.update({'Val_dice_tumor': val_dice.avg[2]})
    return val_log



def train(model, train_loader, optimizer, loss_func, n_labels, alpha, boundary_loss_func):
    print("=======Epoch:{}=======lr:{}".format(epoch, optimizer.param_groups[0]['lr']))
    model.train()
    train_loss = metrics.LossAverage()
    train_dice = metrics.DiceAverage(n_labels)

    for idx, (data, target, dist_map) in tqdm(enumerate(train_loader), total=len(train_loader)):
        data, target = data.float(), target.long()
        target = common.to_one_hot_3d(target, n_labels)
        data, target, dist_map = data.to(device), target.to(device), dist_map.to(device)

        optimizer.zero_grad()

        output = model(data)
        loss0 = loss_func(output[0], target)
        loss1 = loss_func(output[1], target)
        loss2 = loss_func(output[2], target)
        loss3 = loss_func(output[3], target)

        # 计算边界损失
        boundary_loss = boundary_loss_func(torch.softmax(output[3], dim=1), dist_map)

        loss = loss3 + alpha * (loss0 + loss1 + loss2) + 0.1 * boundary_loss  # 将边界损失乘以0.1加入总损失

        loss.backward()
        optimizer.step()

        train_loss.update(loss3.item(), data.size(0))
        train_dice.update(output[3], target)

    train_log = OrderedDict({'Train_Loss': train_loss.avg, 'Train_dice_liver': train_dice.avg[1]})
    if n_labels == 3:
        train_log.update({'Train_dice_tumor': train_dice.avg[2]})
    return train_log




if __name__ == '__main__':
    args = config.args
    save_path = os.path.join('./experiments', args.save)
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    device = torch.device('cpu' if args.cpu else 'cuda')

    # 数据加载
    train_loader = DataLoader(dataset=Train_Dataset(args), batch_size=args.batch_size, 
                              num_workers=args.n_threads, shuffle=True)

    val_loader = DataLoader(dataset=Val_Dataset(args), batch_size=1, 
                            num_workers=args.n_threads, shuffle=False)

    # 模型定义
    model = ResUNet(in_channel=1, out_channel=args.n_labels, training=True).to(device)
    model.apply(weights_init.init_model)
    model = torch.nn.DataParallel(model, device_ids=args.gpu_id)  # 多GPU支持

    # 优化器与学习率调度器
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-5)

    # 损失函数
    loss_func = loss.TverskyLoss()
    boundary_loss_func = loss.BoundaryLoss()  # 初始化边界损失函

    # 日志记录器
    log = logger.Train_Logger(save_path, "train_log")

    # 初始化最优模型与控制变量
    best = [0, 0]  # 最优模型的 epoch 和性能
    trigger = 0  # early stopping 计数器
    alpha = 0.4  # 深监督衰减系数初始值

    for epoch in range(1, args.epochs + 1):
        # 动态调整学习率
        scheduler.step(epoch - 1)
        current_lr = optimizer.param_groups[0]['lr']
        print("=======Epoch:{}=======lr:{}".format(epoch, current_lr))

        # 训练与验证
        train_log = train(model, train_loader, optimizer, loss_func, args.n_labels, alpha, boundary_loss_func)
        val_log = val(model, val_loader, loss_func, args.n_labels)
        log.update(epoch, train_log, val_log)

        # 保存最新模型
        state = {'net': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch}
        torch.save(state, os.path.join(save_path, 'latest_model.pth'))
        trigger += 1

        # 保存最优模型
        if val_log['Val_dice_liver'] > best[1]:
            print('Saving best model')
            torch.save(state, os.path.join(save_path, 'best_model.pth'))
            best[0] = epoch
            best[1] = val_log['Val_dice_liver']
            trigger = 0

        print('Best performance at Epoch: {} | {}'.format(best[0], best[1]))

        # 深监督系数衰减
        if epoch % 30 == 0:
            alpha *= 0.8

        # early stopping
        if args.early_stop is not None and trigger >= args.early_stop:
            print("=> early stopping")
            break

        torch.cuda.empty_cache()
