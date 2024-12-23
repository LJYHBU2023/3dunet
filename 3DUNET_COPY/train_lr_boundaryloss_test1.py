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

def one_hot2dist_3d(seg: torch.Tensor) -> torch.Tensor:

    # Assuming seg is a 4D tensor with shape [batch_size, depth, height, width] and last dimension is one-hot encoding

    B, D, H, W = seg.shape

    C = seg.shape[1]  # Number of classes, assuming one-hot encoding

    res = torch.zeros((B, C, D, H, W), dtype=torch.float32, device=seg.device)

    

    # Convert one-hot encoding to distance maps for each class

    for b in range(B):

        for c in range(C):

            posmask = seg[b, c, ...].cpu().numpy()  # Positive mask for class c

            if posmask.any():

                negmask = 1 - posmask  # Negative mask for class c

                # Calculate distance transforms

                pos_dist = distance_transform_edt(~posmask)  # Distance transform for positive mask

                neg_dist = distance_transform_edt(negmask)  # Distance transform for negative mask

                # Store distance maps in the result tensor

                res[b, c, ...] = torch.from_numpy(neg_dist).to(seg.device) * torch.from_numpy(posmask).to(seg.device) - \
                               (torch.from_numpy(pos_dist).to(seg.device) - 1) * torch.from_numpy(negmask).to(seg.device)

    

    return res

class SurfaceLoss3D():
    def __init__(self):
        self.idc: list[int] = [1]  # Assuming the foreground class is 1

    def __call__(self, probs: torch.Tensor, dist_maps: torch.Tensor) -> torch.Tensor:
        assert simplex(probs)
        assert not one_hot(dist_maps)
        pc = probs[:, self.idc, ...].type(torch.float32)
        dc = dist_maps[:, self.idc, ...].type(torch.float32)
        multiplied = torch.einsum("bcthw,bcthw->bcthw", pc, dc)
        loss = multiplied.mean()
        return loss

# Other helper functions
def simplex(t: torch.Tensor, axis=1) -> bool:
    _sum = t.sum(axis).type(torch.float32)
    _ones = torch.ones_like(_sum, dtype=torch.float32)
    return torch.allclose(_sum, _ones)

def one_hot(t: torch.Tensor) -> bool:
    return simplex(t, axis=1) and sset(t, [0, 1])

def uniq(a: torch.Tensor) -> set:
    return set(torch.unique(a.cpu()).numpy())

def sset(a: torch.Tensor, sub: list) -> bool:
    return uniq(a).issubset(sub)

def probs2one_hot_3d(labels: torch.Tensor) -> torch.Tensor:
    # Assuming labels is a 4D tensor with shape [batch_size, depth, height, width]
    B, D, H, W = labels.shape
    # Get the number of classes (assuming labels are integers from 0 to C-1)
    C = torch.max(labels).int() + 1
    # Create a one-hot encoding tensor
    res = torch.zeros((B, C, D, H, W), dtype=torch.float32, device=labels.device)
    res.scatter_(1, labels.unsqueeze(1), 1)
    return res

def one_hot2dist_3d(seg: torch.Tensor) -> torch.Tensor:
    C: int = seg.shape[1]
    res = torch.zeros_like(seg)
    for c in range(C):
        posmask = seg[:, c, ...].type(torch.bool)
        if posmask.any():
            negmask = ~posmask
            res[:, c, ...] = distance_transform_edt(negmask.cpu().numpy()) * negmask - (distance_transform_edt(posmask.cpu().numpy()) - 1) * posmask
    return res

# Validation function
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

# Training function
def train(model, train_loader, optimizer, loss_func, n_labels):
    print("=======Epoch:{}=======lr:{}".format(epoch, optimizer.param_groups[0]['lr']))
    model.train()
    train_loss = metrics.LossAverage()
    train_dice = metrics.DiceAverage(n_labels)
    for idx, (data, target) in tqdm(enumerate(train_loader), total=len(train_loader)):
        data, target = data.float(), target.long()
        target_one_hot = probs2one_hot_3d(target)
        dist_maps = one_hot2dist_3d(target_one_hot)
        data, target, dist_maps = data.to(device), target_one_hot.to(device), dist_maps.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = loss_func(output, dist_maps)
        loss.backward()
        optimizer.step()
        train_loss.update(loss.item(), data.size(0))
        train_dice.update(output, target)
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

    # Data loading
    train_loader = DataLoader(dataset=Train_Dataset(args), batch_size=args.batch_size,
                              num_workers=args.n_threads, shuffle=True)
    val_loader = DataLoader(dataset=Val_Dataset(args), batch_size=1,
                            num_workers=args.n_threads, shuffle=False)

    # Model definition
    model = ResUNet(in_channel=1, out_channel=args.n_labels, training=True).to(device)
    model.apply(weights_init.init_model)
    model = torch.nn.DataParallel(model, device_ids=args.gpu_id)  # Multi-GPU support

    # Optimizer and learning rate scheduler
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-5)

    # Loss function
    loss_func = SurfaceLoss3D()

    # Logger
    log = logger.Train_Logger(save_path, "train_log")

    # Initialize best model and control variables
    best = [0, 0]  # Best model's epoch and performance
    trigger = 0  # Early stopping counter
    alpha = 0.4  # Deep supervision attenuation coefficient initial value

    for epoch in range(1, args.epochs + 1):
        # Dynamically adjust learning rate
        scheduler.step(epoch - 1)
        current_lr = optimizer.param_groups[0]['lr']
        print("=======Epoch:{}=======lr:{}".format(epoch, current_lr))

        # Training and validation
        train_log = train(model, train_loader, optimizer, loss_func, args.n_labels)
        val_log = val(model, val_loader, loss_func, args.n_labels)
        log.update(epoch, train_log, val_log)

        # Save the latest model
        state = {'net': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch}
        torch.save(state, os.path.join(save_path, 'latest_model.pth'))
        trigger += 1

        # Save the best model
        if val_log['Val_dice_liver'] > best[1]:
            print('Saving best model')
            torch.save(state, os.path.join(save_path, 'best_model.pth'))
            best[0] = epoch
            best[1] = val_log['Val_dice_liver']
            trigger = 0
        print('Best performance at Epoch: {} | {}'.format(best[0], best[1]))

        # Deep supervision coefficient attenuation
        if epoch % 30 == 0:
            alpha *= 0.8

        # Early stopping
        if args.early_stop is not None and trigger >= args.early_stop:
            print("=> early stopping")
            break

        torch.cuda.empty_cache()
