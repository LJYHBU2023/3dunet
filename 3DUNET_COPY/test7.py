import os
import torch
import numpy as np
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from tqdm import tqdm
from collections import OrderedDict
from scipy.ndimage import distance_transform_edt

from dataset.dataset_lits_train import Train_Dataset
from dataset.dataset_lits_val import Val_Dataset
from models import ResUNet, UNet, SegNet, KiUNet_min
from utils import logger, weights_init, metrics, common

import config

class BoundaryLoss(torch.nn.Module):
    def __init__(self, alpha=0.5):
        super(BoundaryLoss, self).__init__()
        self.alpha = alpha

    def forward(self, logits, targets):
        # 如果targets是稀疏张量，将其转换为密集张量
        if targets.is_sparse:
            targets = targets.to_dense()
        
        # 将logits转换为概率
        probs = torch.sigmoid(logits)
        
        # 将targets转换为one-hot编码
        num_classes = probs.shape[1]
        targets_one_hot = torch.nn.functional.one_hot(targets.long(), num_classes=num_classes)
        
        # 调整targets_one_hot的形状以匹配probs
        targets_one_hot = targets_one_hot.permute(0, 4, 1, 2, 3).float()
        
        # 计算距离变换
        dist_transform = self.distance_transform_3d(targets_one_hot)
        
        # 计算边界损失
        boundary_loss = self.compute_boundary_loss(probs, dist_transform)
        
        return boundary_loss

    def distance_transform_3d(self, targets_one_hot):
        # 将one-hot编码转换为二进制掩码
        binary_targets = targets_one_hot.sum(dim=1, keepdim=True)
        
        # 计算每个类别的距离变换
        dist_transform = torch.zeros_like(binary_targets)
        for i in range(binary_targets.shape[1]):
            # 将二进制掩码转换为NumPy数组，并计算距离变换
            dist_np = distance_transform_edt((1 - binary_targets[:, i, ...].cpu().numpy()))
            dist_transform[:, i, ...] = torch.from_numpy(dist_np).to(binary_targets.device)
        
        return dist_transform

    def compute_boundary_loss(self, probs, dist_transform):
        # 计算边界损失
        loss = 0
        for i in range(probs.shape[1]):
            # 确保probs和dist_transform的形状一致
            prob = probs[:, i, ...]
            dist = dist_transform[:, i, ...]
            loss += torch.sum(prob * dist) / prob.numel()  # 使用元素总数进行平均
        loss /= probs.shape[1]
        return loss

def val(model, val_loader, loss_func, n):
    model.eval()
    val_loss = metrics.LossAverage()
    val_dice = metrics.DiceAverage(n_labels)
    with torch.no_grad():
        for idx, (data, target) in tqdm(enumerate(val_loader), total=len(val_loader)):
            data, target = data.float(), target.long()
            target = common.to_one_hot_3d(target, n_labels)
            data, target = data.to(device), target.to(device)
            output = model(data)
            # 确保只使用模型输出的最后一个元素
            loss = loss_func(output[-1], target)
            val_loss.update(loss.item(), data.size(0))
            val_dice.update(output[-1], target)
    val_log = OrderedDict({'Val_Loss': val_loss.avg, 'Val_dice_liver': val_dice.avg[1]})
    if n_labels == 3:
        val_log.update({'Val_dice_tumor': val_dice.avg[2]})
    return val_log

def train(model, train_loader, optimizer, loss_func, n_labels, alpha):
    print("=======Epoch:{}=======lr:{}".format(epoch, optimizer.param_groups[0]['lr']))
    model.train()
    train_loss = metrics.LossAverage()
    train_dice = metrics.DiceAverage(n_labels)
    for idx, (data, target) in tqdm(enumerate(train_loader), total=len(train_loader)):
        data, target = data.float(), target.long()
        target = common.to_one_hot_3d(target, n_labels)
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        # 确保只使用模型输出的最后一个元素
        loss = loss_func(output[-1], target)
        loss.backward()
        optimizer.step()
        train_loss.update(loss.item(), data.size(0))
        train_dice.update(output[-1], target)
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
    loss_func = BoundaryLoss()  # 使用BoundaryLoss

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
        train_log = train(model, train_loader, optimizer, loss_func, args.n_labels, alpha)
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
