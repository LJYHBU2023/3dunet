from posixpath import join
from torch.utils.data import DataLoader, Dataset
import os
import sys
import random
from torchvision.transforms import RandomCrop
import numpy as np
import SimpleITK as sitk
import torch
from .transforms import RandomCrop, RandomFlip_LR, RandomFlip_UD, Center_Crop, Compose3, Resize

class Val_Dataset(Dataset):
    def __init__(self, args):
        self.args = args
        self.filename_list = self.load_file_name_list(os.path.join(args.dataset_path, 'val_path_list.txt'))
        self.transforms = Compose3([
            RandomCrop(self.args.crop_size),
            RandomFlip_LR(prob=0.5),
            RandomFlip_UD(prob=0.5),
            # RandomRotate()
        ])

    def __getitem__(self, index):
        ct_path, seg_path, dist_map_path = self.filename_list[index]
        ct = sitk.ReadImage(ct_path, sitk.sitkInt16)
        seg = sitk.ReadImage(seg_path, sitk.sitkUInt8)
        dist_map = sitk.ReadImage(dist_map_path, sitk.sitkUInt8)

        ct_array = sitk.GetArrayFromImage(ct)
        seg_array = sitk.GetArrayFromImage(seg)
        dist_map_array = sitk.GetArrayFromImage(dist_map)

        ct_array = ct_array / self.args.norm_factor
        ct_array = ct_array.astype(np.float32)

        ct_array = torch.FloatTensor(ct_array).unsqueeze(0)
        seg_array = torch.FloatTensor(seg_array).unsqueeze(0)
        dist_map_array = torch.FloatTensor(dist_map_array).unsqueeze(0)

        if self.transforms:
            ct_array, seg_array, dist_map_array = self.transforms(ct_array, seg_array, dist_map_array)

        return ct_array, seg_array.squeeze(0), dist_map_array.squeeze(0)

    def __len__(self):
        return len(self.filename_list)

    def load_file_name_list(self, file_path):
        file_name_list = []
        with open(file_path, 'r') as file_to_read:
            while True:
                lines = file_to_read.readline().strip()  # 整行读取数据
                if not lines:
                    break
                file_name_list.append(lines.split())
        return file_name_list

if __name__ == "__main__":
    sys.path.append('/ssd/lzq/3DUNet')
    from config import args
    val_ds = Val_Dataset(args)
    # 定义数据加载
    val_dl = DataLoader(val_ds, 2, False, num_workers=1)
    for i, (ct, seg, dist_map) in enumerate(val_dl):
        print(i, ct.size(), seg.size(), dist_map.size())
