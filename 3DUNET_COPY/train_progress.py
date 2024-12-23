from dataset.dataset_lits_val import Val_Dataset

from dataset.dataset_lits_train import Train_Dataset



from torch.utils.data import DataLoader

import torch

import torch.optim as optim

from tqdm import tqdm

import config



from models import UNet, ResUNet, KiUNet_min, SegNet



from utils import logger, weights_init, metrics, common, loss

import os

import numpy as np

from collections import OrderedDict

import torch.nn.functional as F





def train(model, train_loader, optimizer, loss_func, n_labels, alpha):

    print("=======Epoch:{}=======lr:{}".format(epoch, optimizer.state_dict()['param_groups'][0]['lr']))

    model.train()

    train_loss = metrics.LossAverage()

    train_dice = metrics.DiceAverage(n_labels)



    for idx, (data, target) in tqdm(enumerate(train_loader), total=len(train_loader)):

        data, target = data.float(), target.long()



        # Ensure target is a tensor before passing to to_one_hot_3d

        if not isinstance(target, torch.Tensor):

            target = torch.tensor(target, dtype=torch.long)



        # Convert target to one-hot encoding and move to device

        target = common.to_one_hot_3d(target, n_labels)

        data, target = data.to(device), target.to(device)



        optimizer.zero_grad()

        output = model(data)



        # Adjust target size to match the model's output size

        if output[0].shape[2:] != target.shape[2:]:

            target = F.interpolate(target.float(), size=output[0].shape[2:], mode='trilinear', align_corners=True)



        # Compute losses for deep supervision

        loss0 = loss_func(output[0], target)

        loss1 = loss_func(output[1], target)

        loss2 = loss_func(output[2], target)

        loss3 = loss_func(output[3], target)



        # Combining losses with the alpha coefficient for deep supervision

        loss = loss3 + alpha * (loss0 + loss1 + loss2)

        loss.backward()

        optimizer.step()



        train_loss.update(loss3.item(), data.size(0))

        train_dice.update(output[3], target)



    val_log = OrderedDict({'Train_Loss': train_loss.avg, 'Train_dice_liver': train_dice.avg[1]})

    if n_labels == 3:

        val_log.update({'Train_dice_tumor': train_dice.avg[2]})

    return val_log





def val(model, val_loader, loss_func, n_labels):

    model.eval()

    val_loss = metrics.LossAverage()

    val_dice = metrics.DiceAverage(n_labels)

    with torch.no_grad():

        for idx, (data, target) in tqdm(enumerate(val_loader), total=len(val_loader)):

            data, target = data.float(), target.long()



            # One-hot encode the target

            target = common.to_one_hot_3d(target, n_labels)



            # Move to device

            data, target = data.to(device), target.to(device)



            # Model prediction

            output = model(data)



            # Ensure output and target have the same spatial dimensions

            target_size = target.shape[2:]  # Extract (D, H, W)

            if output[0].shape[2:] != target_size:

                output_resized = F.interpolate(output[0], size=target_size, mode='trilinear', align_corners=True)

            else:

                output_resized = output[0]



            # Ensure output and target have the same number of channels

            if output_resized.shape[1] != target.shape[1]:

                raise ValueError(

                    f"Mismatch in number of channels: output={output_resized.shape[1]}, target={target.shape[1]}")



            # Compute loss

            loss = loss_func(output_resized, target)



            val_loss.update(loss.item(), data.size(0))

            val_dice.update(output_resized, target)



    val_log = OrderedDict({'Val_Loss': val_loss.avg, 'Val_dice_liver': val_dice.avg[1]})

    if n_labels == 3:

        val_log.update({'Val_dice_tumor': val_dice.avg[2]})

    return val_log





if __name__ == '__main__':




    args = config.args

    save_path = os.path.join('./experiments', args.save)

    if not os.path.exists(save_path): os.mkdir(save_path)

    device = torch.device('cpu' if args.cpu else 'cuda')



    # Data info

    train_loader = DataLoader(dataset=Train_Dataset(args), batch_size=args.batch_size, num_workers=args.n_threads,

                              shuffle=True)

    val_loader = DataLoader(dataset=Val_Dataset(args), batch_size=1, num_workers=args.n_threads, shuffle=False)



    # Model info

    model = ResUNet(in_channel=1, out_channel=args.n_labels, training=True).to(device)



    model.apply(weights_init.init_model)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    common.print_network(model)

    model = torch.nn.DataParallel(model, device_ids=args.gpu_id)  # Multi-GPU



    loss = loss.TverskyLoss()



    log = logger.Train_Logger(save_path, "train_log")



    best = [0, 0]  # Initialize best model epoch and performance

    trigger = 0  # Early stop counter

    alpha = 0.4  # Initial deep supervision decay coefficient



    for epoch in range(1, args.epochs + 1):

        common.adjust_learning_rate(optimizer, epoch, args)

        train_log = train(model, train_loader, optimizer, loss, args.n_labels, alpha)

        val_log = val(model, val_loader, loss, args.n_labels)

        log.update(epoch, train_log, val_log)



        # Save checkpoint.

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



        # Deep supervision coefficient decay

        if epoch % 30 == 0: alpha *= 0.8



        # Early stopping

        if args.early_stop is not None:

            if trigger >= args.early_stop:

                print("=> early stopping")

                break

        torch.cuda.empty_cache()


