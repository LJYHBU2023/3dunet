import os
import torch
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
from scipy.ndimage import distance_transform_edt
from typing import List

def one_hot2dist(seg: np.ndarray) -> np.ndarray:
    assert one_hot(torch.Tensor(seg), axis=0)
    C: int = seg.shape[0]  # 通道数，即类别数
    res = np.zeros_like(seg)
    for c in range(C):
        posmask = seg[c].astype(bool)  # 使用内置的 bool 类型
        if posmask.any():
            negmask = ~posmask
            # 对3D数据使用distance_transform_edt函数
            res[c] = distance_transform_edt(~posmask) * negmask - (distance_transform_edt(posmask) - 1) * posmask
    return res

def probs2one_hot(probs: torch.Tensor) -> torch.Tensor:
    _, C, _, _, _ = probs.shape
    assert simplex(probs)
    res = class2one_hot(probs2class(probs), C)
    assert res.shape == probs.shape
    assert one_hot(res)
    return res

def class2one_hot(seg: torch.Tensor, C: int) -> torch.Tensor:
    if len(seg.shape) == 4:
        seg = seg.unsqueeze(dim=1)
    assert sset(seg, list(range(C)))
    b, w, h, d = seg.shape
    res = torch.stack([seg == c for c in range(C)], dim=1).type(torch.int32)
    assert res.shape == (b, C, w, h, d)
    assert one_hot(res)
    return res

def one_hot(t: torch.Tensor, axis=1) -> bool:
    return simplex(t, axis) and sset(t, [0, 1])

def simplex(t: torch.Tensor, axis=1) -> bool:
    _sum = t.sum(axis).type(torch.float32)
    _ones = torch.ones_like(_sum, dtype=torch.float32)
    return torch.allclose(_sum, _ones)

def sset(a: torch.Tensor, sub: List[int]) -> bool:
    return set(torch.unique(a.cpu()).numpy()).issubset(sub)

def probs2class(probs: torch.Tensor) -> torch.Tensor:
    b, _, w, h, d = probs.shape
    res = probs.argmax(dim=1)
    assert res.shape == (b, w, h, d)
    return res

class SurfaceLoss():
    def __init__(self):
        self.idc: List[int] = [1]  # Assuming the foreground class is 1

    def __call__(self, probs: torch.Tensor, dist_maps: torch.Tensor, _: torch.Tensor) -> torch.Tensor:
        assert simplex(probs)
        assert not one_hot(dist_maps)
        pc = probs[:, self.idc, ...].type(torch.float32)
        dc = dist_maps[:, self.idc, ...].type(torch.float32)
        # 确保 pc 和 dc 都是四维张量
        if len(pc.shape) != 4 or len(dc.shape) != 4:
            raise ValueError("pc and dc must be 4-dimensional tensors")
        multiplied = torch.einsum("bcxd->bcx", pc, dc)  # 修改方程式以匹配操作数的维度
        loss = multiplied.mean()
        return loss

def train(model, train_loader, optimizer, loss_func, boundary_loss, n_labels, alpha):

    print("=======Epoch:{}=======lr:{}".format(epoch, optimizer.param_groups[0]['lr']))

    model.train()

    train_loss = metrics.LossAverage()

    train_dice = metrics.DiceAverage(n_labels)

    for idx, (data, target) in tqdm(enumerate(train_loader), total=len(train_loader)):

        data, target = data.float(), target.long()

        target_one_hot = common.to_one_hot_3d(target, n_labels)  # 确保这是正确的one-hot编码

        data, target_one_hot = data.to(device), target_one_hot.to(device)

        optimizer.zero_grad()

        output = model(data)

        
        loss0 = loss_func(output[0], target_one_hot)

        loss1 = loss_func(output[1], target_one_hot)

        loss2 = loss_func(output[2], target_one_hot)

        loss3 = loss_func(output[3], target_one_hot)

        loss = loss3 + alpha * (loss0 + loss1 + loss2)

        

        # 计算Boundary Loss

        dist_maps = []

        for i in range(target_one_hot.size(0)):  # 遍历批次中的每个样本

            sample_one_hot = target_one_hot[i].cpu().numpy()  # 获取单个样本的one-hot编码

            sample_dist_map = one_hot2dist(sample_one_hot)  # 计算单个样本的距离图

            dist_maps.append(sample_dist_map)

        dist_maps = np.stack(dist_maps, axis=0)  # 将所有样本的距离图堆叠成一个数组

        dist_maps = torch.from_numpy(dist_maps).to(device)

        boundary_loss_value = boundary_loss(output[3], dist_maps, None)

        loss += boundary_loss_value

        loss.backward()

        optimizer.step()

        train_loss.update(loss3.item(), data.size(0))

        train_dice.update(output[3], target_one_hot)

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
    train_loader = DataLoader(dataset=Train_Dataset(args), batch_size=args.batch_size, num_workers=args.n_threads, shuffle=True)
    val_loader = DataLoader(dataset=Val_Dataset(args), batch_size=1, num_workers=args.n_threads, shuffle=False)
    model = ResUNet(in_channel=1, out_channel=args.n_labels, training=True).to(device)
    model.apply(weights_init.init_model)
    model = torch.nn.DataParallel(model, device_ids=args.gpu_id)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-5)
    loss_func = loss.TverskyLoss()
    boundary_loss = SurfaceLoss()
    log = logger.Train_Logger(save_path, "train_log")
    best = [0, 0]
    trigger = 0
    alpha = 0.4
    for epoch in range(1, args.epochs + 1):
        scheduler.step(epoch - 1)
        current_lr = optimizer.param_groups[0]['lr']
        print("=======Epoch:{}=======lr:{}".format(epoch, current_lr))
        train_log = train(model, train_loader, optimizer, loss_func, boundary_loss, args.n_labels, alpha)
        val_log = val(model, val_loader, loss_func, args.n_labels)
        log.update(epoch, train_log, val_log)
        state = {'net': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch}
        torch.save(state, os.path.join(save_path, 'latest_model.pth'))
        trigger += 1
        if val_log['Val_dice_liver'] > best[1]:
            print('Saving best model')
            torch.save(state, os.path.join(save_path, 'best_model.pth'))
            best[0] = epoch
            best[1] = val_log['Val_dice_liver']
            trigger = 0
        print('Best performance at Epoch: {} | {}'.format(best[0], best[1]))
        if epoch % 30 == 0:
            alpha *= 0.8
        if args.early_stop is not None and trigger >= args.early_stop:
            print("=> early stopping")
            break
        torch.cuda.empty_cache()
