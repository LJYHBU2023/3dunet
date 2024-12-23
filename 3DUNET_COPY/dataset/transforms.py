"""
This part is based on the dataset class implemented by pytorch, 
including train_dataset and test_dataset, as well as data augmentation
"""
from torch.utils.data import Dataset
import torch
import numpy as np
import random
import torch.nn.functional as F
from torchvision import transforms
from torchvision.transforms.functional import normalize

#----------------------data augment-------------------------------------------
class Resize:
    def __init__(self, scale):
        self.scale = scale

    def __call__(self, img, mask, dist_map):
        img = img.unsqueeze(0)
        mask = mask.unsqueeze(0).float()
        dist_map = dist_map.unsqueeze(0).float()

        img = F.interpolate(img, scale_factor=(1, self.scale, self.scale), mode='trilinear', align_corners=False, recompute_scale_factor=True)
        mask = F.interpolate(mask, scale_factor=(1, self.scale, self.scale), mode="nearest", recompute_scale_factor=True)
        dist_map = F.interpolate(dist_map, scale_factor=(1, self.scale, self.scale), mode="nearest", recompute_scale_factor=True)

        return img[0], mask[0], dist_map[0]

class RandomResize:
    def __init__(self, s_rank, w_rank, h_rank):
        self.w_rank = w_rank
        self.h_rank = h_rank
        self.s_rank = s_rank

    def __call__(self, img, mask, dist_map):
        random_w = random.randint(self.w_rank[0], self.w_rank[1])
        random_h = random.randint(self.h_rank[0], self.h_rank[1])
        random_s = random.randint(self.s_rank[0], self.s_rank[1])
        self.shape = [random_s, random_h, random_w]
        img = img.unsqueeze(0).float()
        mask = mask.unsqueeze(0).float()
        dist_map = dist_map.unsqueeze(0).float()

        img = F.interpolate(img, size=self.shape, mode='trilinear', align_corners=False)
        mask = F.interpolate(mask, size=self.shape, mode="nearest")
        dist_map = F.interpolate(dist_map, size=self.shape, mode="nearest")

        return img[0], mask[0].long(), dist_map[0].long()

class RandomCrop:
    def __init__(self, slices):
        self.slices = slices

    def _get_range(self, slices, crop_slices):
        if slices < crop_slices:
            return 0, slices  # 如果请求的切片数大于实际切片数，返回整个范围
        else:
            start = random.randint(0, slices - crop_slices)
            end = start + crop_slices
            return start, end

    def __call__(self, img, mask, dist_map):
        ss, es = self._get_range(img.size(1), self.slices)
        
        # 确保裁剪后的尺寸不为0
        if es <= ss:  # 这种情况不应该发生，因为我们已经保证了切片数不会超过实际切片数
            return img, mask, dist_map  # 直接返回原始数据

        tmp_img = torch.zeros((img.size(0), es-ss, img.size(2), img.size(3)))
        tmp_mask = torch.zeros((mask.size(0), es-ss, mask.size(2), mask.size(3)))
        tmp_dist_map = torch.zeros((dist_map.size(0), es-ss, dist_map.size(2), dist_map.size(3)))

        tmp_img[:, :] = img[:, ss:es]
        tmp_mask[:, :] = mask[:, ss:es]
        tmp_dist_map[:, :] = dist_map[:, ss:es]

        return tmp_img, tmp_mask, tmp_dist_map

class RandomFlip_LR:
    def __init__(self, prob=0.5):
        self.prob = prob

    def _flip(self, img, prob):
        if prob <= self.prob:
            img = img.flip(2)
        return img

    def __call__(self, img, mask, dist_map):
        prob = random.uniform(0, 1)
        return self._flip(img, prob), self._flip(mask, prob), self._flip(dist_map, prob)

class RandomFlip_UD:
    def __init__(self, prob=0.5):
        self.prob = prob

    def _flip(self, img, prob):
        if prob <= self.prob:
            img = img.flip(3)
        return img

    def __call__(self, img, mask, dist_map):
        prob = random.uniform(0, 1)
        return self._flip(img, prob), self._flip(mask, prob), self._flip(dist_map, prob)

class RandomRotate:
    def __init__(self, max_cnt=3):
        self.max_cnt = max_cnt

    def _rotate(self, img, cnt):
        img = torch.rot90(img, cnt, [1, 2])
        return img

    def __call__(self, img, mask, dist_map):
        cnt = random.randint(0, self.max_cnt)
        return self._rotate(img, cnt), self._rotate(mask, cnt), self._rotate(dist_map, cnt)

class Center_Crop:
    def __init__(self, base, max_size):
        self.base = base  
        self.max_size = max_size 
        if self.max_size % self.base:
            self.max_size = self.max_size - self.max_size % self.base

    def __call__(self, img, mask, dist_map):
        if img.size(1) < self.base:
            return None
        slice_num = img.size(1) - img.size(1) % self.base
        slice_num = min(self.max_size, slice_num)

        left = img.size(1) // 2 - slice_num // 2
        right = img.size(1) // 2 + slice_num // 2

        crop_img = img[:, left:right]
        crop_label = mask[:, left:right]
        crop_dist_map = dist_map[:, left:right]
        return crop_img, crop_label, crop_dist_map

class ToTensor:
    def __call__(self, img, mask, dist_map):
        img = transforms.ToTensor()(img)
        mask = torch.from_numpy(np.array(mask)).long()
        dist_map = torch.from_numpy(np.array(dist_map)).long()
        return img, mask, dist_map

class Normalize:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, img, mask, dist_map):
        return normalize(img, self.mean, self.std, False), mask, dist_map

class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, mask):
        for t in self.transforms:
            img, mask = t(img, mask)
        return img, mask

class Compose3:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, mask, dist_map):
        for t in self.transforms:
            img, mask, dist_map = t(img, mask, dist_map)
        return img, mask, dist_map
