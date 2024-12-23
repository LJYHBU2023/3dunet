import os
import torch
import numpy as np
from torch.utils.data import DataLoader
import torch.optim as optim
from tqdm import tqdm
from collections import OrderedDict
from dataset.dataset_lits_train import Train_Dataset
from dataset.dataset_lits_val import Val_Dataset
from models import ResUNet, UNet, SegNet, KiUNet_min
from utils import logger, weights_init, metrics, common,loss
import config
from scipy.ndimage import distance_transform_edt
#from loss import TverskyLoss  # 确保这个导入语句与您的文件结构一致

# 定义3D Boundary Loss
class SurfaceLoss3D():
    def __init__(self):
        self.idc = [1]  # 假设1是前景类别

    def __call__(self, probs: torch.Tensor, dist_maps: torch.Tensor) -> torch.Tensor:
        pc = probs[:, self.idc, ...].type(torch.float32)
        dc = dist_maps[:, self.idc, ...].type(torch.float32)
        multiplied = torch.einsum('bcxyz, bcxyz->bcxyz', pc, dc)
        loss = multiplied.mean()
        return loss

# 定义 CombinedLoss
class CombinedLoss(torch.nn.Module):
    def __init__(self):
        super(CombinedLoss, self).__init__()
        self.surface_loss = SurfaceLoss3D()
        self.tversky_loss = loss.TverskyLoss()

    def forward(self, probs, target):
        # 计算 SurfaceLoss3D
        dist_maps = one_hot2dist(probs2one_hot(probs).cpu().numpy())
        dist_maps = torch.tensor(dist_maps, device=probs.device, dtype=torch.float32)
        surface_loss = self.surface_loss(probs, dist_maps)
        
        # 计算 TverskyLoss
        tversky_loss = self.tversky_loss(probs, target)
        
        # 直接相加两种损失函数
        combined_loss = surface_loss + tversky_loss
        return combined_loss

def probs2class(probs: torch.Tensor) -> torch.Tensor:

    # 将概率转换为类别

    b, c, d, h, w = probs.shape  # 形状: (batch_size, channels, depth, height, width)

    res = probs.argmax(dim=1)  # 沿通道维度取最大值，形状: (batch_size, depth, height, width)

    return res





def probs2one_hot(probs: torch.Tensor) -> torch.Tensor:

    # 将概率转换为one-hot编码

    b, c, d, h, w = probs.shape  # 形状: (batch_size, channels, depth, height, width)

    res = class2one_hot(probs2class(probs), c)  # 调用class2one_hot函数

    return res





def class2one_hot(seg: torch.Tensor, C: int) -> torch.Tensor:

    # 将类别转换为one-hot编码

    assert len(seg.shape) == 4  # 确保seg是4D张量

    b, d, h, w = seg.shape  # 形状: (batch_size, depth, height, width)

    res = torch.stack([seg == c for c in range(C)], dim=1).type(

        torch.int32)  # 形状: (batch_size, C, depth, height, width)

    return res





def one_hot2dist(seg: np.ndarray) -> np.ndarray:

    # 将one-hot编码转换为距离图

    assert one_hot(torch.Tensor(seg), axis=1)  # 确保one-hot编码正确

    C: int = len(seg)  # 类别数

    res = np.zeros_like(seg)  # 初始化距离图，形状与seg相同

    for c in range(C):

        posmask = seg[c].astype(bool)  # 将第c个通道视为正样本mask

        if posmask.any():

            negmask = ~posmask  # 负样本mask

            res[c] = distance_transform_edt(negmask) * negmask - (distance_transform_edt(posmask) - 1) * posmask

    return res





def simplex(t: torch.Tensor, axis=1) -> bool:

    # 检查是否为simplex（概率分布）

    _sum = t.sum(axis).type(torch.float32)  # 沿axis求和，形状: (batch_size,)

    _ones = torch.ones_like(_sum, dtype=torch.float32)  # 形状: (batch_size,)

    return torch.allclose(_sum, _ones)  # 检查是否接近1





def one_hot(t: torch.Tensor, axis=1) -> bool:

    # 检查是否为one-hot编码

    return simplex(t, axis) and sset(t, [0, 1])  # 检查是否为simplex且值在[0, 1]





def uniq(a: torch.Tensor) -> set:

    # 返回张量中的唯一值集合

    return set(torch.unique(a.cpu()).numpy())  # 形状: (num_unique_values,)





def sset(a: torch.Tensor, sub: list) -> bool:

    # 检查张量的所有值是否为子集

    return uniq(a).issubset(sub)  # 返回布尔值





# 训练和验证函数
def val(model, val_loader, loss_func, n_labels, device):
    model.eval()
    val_loss = metrics.LossAverage()
    val_dice = metrics.DiceAverage(n_labels)
    with torch.no_grad():
        for idx, (data, target) in tqdm(enumerate(val_loader), total=len(val_loader)):
            data, target = data.float(), target.long()
            target = common.to_one_hot_3d(target, n_labels).to(device)
            data, target = data.to(device), target
            output = model(data)
            if isinstance(output, tuple):  # 检查输出是否为元组
                output = output[0]  # 取第一个元素作为输出
            output = probs2one_hot(output)
            dist_maps = one_hot2dist(output.cpu().numpy())
            dist_maps = torch.tensor(dist_maps).to(device)
            loss = loss_func(output, target)
            val_loss.update(loss.item(), data.size(0))
            val_dice.update(output, target)
    val_log = OrderedDict({'Val_Loss': val_loss.avg, 'Val_dice_liver': val_dice.avg[1]})
    if n_labels == 3:
        val_log.update({'Val_dice_tumor': val_dice.avg[2]})
    return val_log




def train(model, train_loader, optimizer, tversky_loss_func, surface_loss_func, n_labels, device, epoch, alpha=0.1):

    print("=======Epoch:{}=======lr:{}".format(epoch, optimizer.param_groups[0]['lr']))

    model.train()

    train_loss = metrics.LossAverage()

    train_dice = metrics.DiceAverage(n_labels)



    for idx, (data, target) in tqdm(enumerate(train_loader), total=len(train_loader)):

        data, target = data.float(), target.long()

        target = common.to_one_hot_3d(target, n_labels).to(device)

        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()



        output = model(data)

        # 假设模型输出了四个不同尺度的特征图，我们使用最后一个作为主要输出

        main_output = output[-1]

        

        # 原有的 TverskyLoss 计算方式

        loss0 = tversky_loss_func(output[0], target)

        loss1 = tversky_loss_func(output[1], target)

        loss2 = tversky_loss_func(output[2], target)

        loss3 = tversky_loss_func(output[3], target)

        

        # 计算边界损失

        boundary_loss = surface_loss_func(main_output, target)

        

        # 结合原有的损失和边界损失

        loss = loss3 + alpha * (loss0 + loss1 + loss2) + boundary_loss



        loss.backward()

        optimizer.step()



        train_loss.update(loss.item(), data.size(0))

        train_dice.update(main_output, target)



    train_log = OrderedDict({'Train_Loss': train_loss.avg, 'Train_dice': train_dice.avg})

    return train_log

if __name__ == '__main__':

    # 确保配置和数据加载等代码与您的原始代码一致

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

    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-5)



    tversky_loss_func = loss.TverskyLoss()  # 使用 TverskyLoss

    surface_loss_func = SurfaceLoss3D()  # 使用 SurfaceLoss3D 作为边界损失



    log = logger.Train_Logger(save_path, "train_log")

    best = [0, 0]

    trigger = 0

    alpha = 0.1  # 设置边界损失的权重



    for epoch in range(1, args.epochs + 1):

        scheduler.step(epoch - 1)

        current_lr = optimizer.param_groups[0]['lr']

        print("=======Epoch:{}=======lr:{}".format(epoch, current_lr))



        train_log = train(model, train_loader, optimizer, tversky_loss_func, surface_loss_func, args.n_labels, device, epoch, alpha)

        val_log = val(model, val_loader, tversky_loss_func, args.n_labels, device)



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



        if args.early_stop is not None and trigger >= args.early_stop:

            print("=> early stopping")

            break



        torch.cuda.empty_cache()
