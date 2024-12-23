import os
import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from tqdm import tqdm
from collections import OrderedDict
from dataset.dataset_lits_train import Train_Dataset
from dataset.dataset_lits_val import Val_Dataset
from models import ResUNet, UNet, SegNet, KiUNet_min
from utils import logger, weights_init, metrics, common
import config
from scipy.ndimage import distance_transform_edt

# 定义Dice Loss
class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, outputs, target):
        smooth = 1.
        dice = 0.
        for output in outputs:
            for i in range(output.size(1)):
                dice += 2 * (output[:, i] * target[:, i]).sum(dim=1).sum(dim=1).sum(dim=1) / (
                        output[:, i].pow(2).sum(dim=1).sum(dim=1).sum(dim=1) +
                        target[:, i].pow(2).sum(dim=1).sum(dim=1).sum(dim=1) + smooth)
        dice = dice / len(outputs)
        return 1 - dice.mean()

# 定义Boundary Loss
class BoundaryLoss(nn.Module):
    def __init__(self):
        super(BoundaryLoss, self).__init__()

    def forward(self, pred, target, dist_maps):
        pred = torch.sigmoid(pred)
        target = target.float()
        intersection = (pred * target).sum(dim=(1, 2, 3))
        boundary_intersection = (pred * target * dist_maps).sum(dim=(1, 2, 3))
        union = pred.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3)) - intersection
        boundary_union = pred.sum(dim=(1, 2, 3)) + dist_maps.sum(dim=(1, 2, 3)) - boundary_intersection
        dice = (2 * intersection + 1) / (union + 1)
        boundary_dice = (2 * boundary_intersection + 1) / (boundary_union + 1)
        return 1 - dice, 1 - boundary_dice

def probs2class(probs: torch.Tensor) -> torch.Tensor:

    b, c, d, h, w = probs.shape

    res = probs.argmax(dim=1)

    return res



def probs2one_hot(probs: torch.Tensor) -> torch.Tensor:

    b, c, d, h, w = probs.shape

    res = class2one_hot(probs2class(probs), c)

    return res



def class2one_hot(seg: torch.Tensor, C: int) -> torch.Tensor:

    assert len(seg.shape) == 4  # 确保seg是4D张量

    b, d, h, w = seg.shape

    res = torch.stack([seg == c for c in range(C)], dim=1).type(torch.int32)

    return res

def one_hot2dist(seg: np.ndarray) -> np.ndarray:

    assert one_hot(torch.Tensor(seg), axis=1)

    C: int = len(seg)

    res = np.zeros_like(seg)

    for c in range(C):

        posmask = seg[c].astype(bool)  # 将np.bool替换为bool

        if posmask.any():

            negmask = ~posmask

            res[c] = distance_transform_edt(negmask) * negmask - (distance_transform_edt(posmask) - 1) * posmask

    return res



def simplex(t: torch.Tensor, axis=1) -> bool:

    _sum = t.sum(axis).type(torch.float32)

    _ones = torch.ones_like(_sum, dtype=torch.float32)

    return torch.allclose(_sum, _ones)



def one_hot(t: torch.Tensor, axis=1) -> bool:

    return simplex(t, axis) and sset(t, [0, 1])



def uniq(a: torch.Tensor) -> set:

    return set(torch.unique(a.cpu()).numpy())



def sset(a: torch.Tensor, sub: list) -> bool:

    return uniq(a).issubset(sub)


# 训练函数
def train(model, train_loader, optimizer, loss_func, boundary_loss_func, n_labels, alpha):
    print("=======Epoch:{}=======lr:{}".format(epoch, optimizer.param_groups[0]['lr']))

    model.train()

    train_loss = metrics.LossAverage()
    train_dice = metrics.DiceAverage(n_labels)

    for idx, (data, target) in tqdm(enumerate(train_loader), total=len(train_loader)):
        data, target = data.float(), target.long()

        target = common.to_one_hot_3d(target, n_labels)

        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()

        outputs = model(data)

        dice_loss = loss_func(outputs, target)

        dist_maps = one_hot2dist(target.cpu().numpy())
        dist_maps = torch.tensor(dist_maps).to(device)

        boundary_loss = boundary_loss_func(outputs[-1], target, dist_maps)

        loss = dice_loss + alpha * boundary_loss

        loss.backward()

        optimizer.step()

        train_loss.update(loss.item(), data.size(0))

        train_dice.update(outputs[-1], target)

    train_log = OrderedDict({'Train_Loss': train_loss.avg, 'Train_dice_liver': train_dice.avg[1]})

    if n_labels == 3:
        train_log.update({'Train_dice_tumor': train_dice.avg[2]})

    return train_log

# 验证函数
def val(model, val_loader, loss_func, n_labels):
    model.eval()

    val_loss = metrics.LossAverage()
    val_dice = metrics.DiceAverage(n_labels)

    with torch.no_grad():
        for idx, (data, target) in tqdm(enumerate(val_loader), total=len(val_loader)):
            data, target = data.float(), target.long()

            target = common.to_one_hot_3d(target, n_labels)

            data, target = data.to(device), target.to(device)

            output = model(data)

            loss = loss_func(output, target)

            val_loss.update(loss.item(), data.size(0))

            val_dice.update(output, target)

    val_log = OrderedDict({'Val_Loss': val_loss.avg, 'Val_dice_liver': val_dice.avg[1]})

    if n_labels == 3:
        val_log.update({'Val_dice_tumor': val_dice.avg[2]})

    return val_log

if __name__ == '__main__':
    args = config.args
    save_path = os.path.join('./experiments', args.save)

    if not os.path.exists(save_path):
        os.mkdir(save_path)

    device = torch.device('cpu' if args.cpu else 'cuda')

    train_loader = DataLoader(dataset=Train_Dataset(args), batch_size=args.batch_size,
                              num_workers=args.n_threads, shuffle=True)

    val_loader = DataLoader(dataset=Val_Dataset(args), batch_size=1,
                            num_workers=args.n_threads, shuffle=False)

    model = ResUNet(in_channel=1, out_channel=args.n_labels, training=True).to(device)
    model.apply(weights_init.init_model)
    model = torch.nn.DataParallel(model, device_ids=args.gpu_id)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-5)

    loss_func = DiceLoss()
    boundary_loss_func = BoundaryLoss()

    log = logger.Train_Logger(save_path, "train_log")

    best = [0, 0]
    trigger = 0
    alpha = 0.1

    for epoch in range(1, args.epochs + 1):
        scheduler.step(epoch - 1)
        current_lr = optimizer.param_groups[0]['lr']
        print("=======Epoch:{}=======lr:{}".format(epoch, current_lr))

        train_log = train(model, train_loader, optimizer, loss_func, boundary_loss_func, args.n_labels, alpha)
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
