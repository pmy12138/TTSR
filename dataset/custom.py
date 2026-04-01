# dataset/custom.py

import os
from imageio import imread
from PIL import Image
import numpy as np
import glob
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision import transforms

import warnings

warnings.filterwarnings("ignore")


class RandomFlip(object):
    def __call__(self, sample):
        if (np.random.randint(0, 2) == 1):
            for key in sample:
                sample[key] = np.fliplr(sample[key]).copy()
        if (np.random.randint(0, 2) == 1):
            for key in sample:
                sample[key] = np.flipud(sample[key]).copy()
        return sample


class RandomRotate(object):
    def __call__(self, sample):
        k = np.random.randint(0, 4)
        for key in sample:
            sample[key] = np.rot90(sample[key], k).copy()
        return sample


class ToTensor(object):
    def __call__(self, sample):
        out = {}
        for key in sample:
            out[key] = torch.from_numpy(sample[key].transpose((2, 0, 1)).copy()).float()
        return out


class TrainSet(Dataset):
    def __init__(self, args, transform=transforms.Compose([RandomFlip(), RandomRotate(), ToTensor()])):
        self.input_list = sorted([
            os.path.join(args.dataset_dir, 'train/input', name)
            for name in os.listdir(os.path.join(args.dataset_dir, 'train/input'))
        ])
        self.ref_list = sorted([
            os.path.join(args.dataset_dir, 'train/ref', name)
            for name in os.listdir(os.path.join(args.dataset_dir, 'train/ref'))
        ])
        self.crop_size = args.train_crop_size  # LR patch size, e.g. 40
        self.scale = 4
        self.transform = transform

    def __len__(self):
        return len(self.input_list)

    def __getitem__(self, idx):
        ### 读取320x180有噪LR图
        LR_full = imread(self.input_list[idx])  # (180, 320, 3)
        h_lr, w_lr = LR_full.shape[:2]

        ### 读取1280x720无噪HR图（同时作为Ref和GT）
        HR_full = imread(self.ref_list[idx])  # (720, 1280, 3)
        h_hr, w_hr = HR_full.shape[:2]

        ### 确保HR尺寸是scale的倍数
        h_hr = h_hr // self.scale * self.scale
        w_hr = w_hr // self.scale * self.scale
        HR_full = HR_full[:h_hr, :w_hr, :]

        ### 对应的LR尺寸
        h_lr = h_hr // self.scale
        w_lr = w_hr // self.scale
        LR_full = LR_full[:h_lr, :w_lr, :]

        ### 随机裁剪LR patch (crop_size x crop_size)
        lr_h = random.randint(0, h_lr - self.crop_size)
        lr_w = random.randint(0, w_lr - self.crop_size)
        hr_h = lr_h * self.scale
        hr_w = lr_w * self.scale
        hr_crop_size = self.crop_size * self.scale  # 160

        LR = LR_full[lr_h:lr_h + self.crop_size, lr_w:lr_w + self.crop_size, :]  # (40, 40, 3)
        HR = HR_full[hr_h:hr_h + hr_crop_size, hr_w:hr_w + hr_crop_size, :]  # (160, 160, 3)

        ### LR_sr: 将有噪LR patch双三次上采样到HR尺寸
        LR_sr = np.array(Image.fromarray(LR).resize(
            (hr_crop_size, hr_crop_size), Image.BICUBIC))  # (160, 160, 3)

        ### Ref: 使用同位置的HR patch作为参考图（干净无噪）
        Ref = HR.copy()  # (160, 160, 3)

        ### Ref_sr: 参考图下采样再上采样（模拟TTSR的Ref_sr）
        Ref_sr = np.array(Image.fromarray(Ref).resize(
            (hr_crop_size // self.scale, hr_crop_size // self.scale), Image.BICUBIC))
        Ref_sr = np.array(Image.fromarray(Ref_sr).resize(
            (hr_crop_size, hr_crop_size), Image.BICUBIC))  # (160, 160, 3)

        ### 转float32, 归一化到[-1, 1]
        LR = LR.astype(np.float32) / 127.5 - 1.
        LR_sr = LR_sr.astype(np.float32) / 127.5 - 1.
        HR = HR.astype(np.float32) / 127.5 - 1.
        Ref = Ref.astype(np.float32) / 127.5 - 1.
        Ref_sr = Ref_sr.astype(np.float32) / 127.5 - 1.

        sample = {'LR': LR, 'LR_sr': LR_sr, 'HR': HR, 'Ref': Ref, 'Ref_sr': Ref_sr}

        if self.transform:
            sample = self.transform(sample)
        return sample


class TestSet(Dataset):
    def __init__(self, args, transform=transforms.Compose([ToTensor()])):
        self.input_list = sorted([
            os.path.join(args.dataset_dir, 'test/input', name)
            for name in os.listdir(os.path.join(args.dataset_dir, 'test/input'))
        ])
        self.ref_list = sorted([
            os.path.join(args.dataset_dir, 'test/ref', name)
            for name in os.listdir(os.path.join(args.dataset_dir, 'test/ref'))
        ])
        self.scale = 4
        self.transform = transform

    def __len__(self):
        return len(self.input_list)

    def __getitem__(self, idx):
        ### 读取320x180有噪LR图
        LR = imread(self.input_list[idx])
        h_lr, w_lr = LR.shape[:2]

        ### 读取1280x720无噪HR图（同时作为Ref和GT）
        HR = imread(self.ref_list[idx])
        h_hr, w_hr = HR.shape[:2]
        h_hr = h_hr // self.scale * self.scale
        w_hr = w_hr // self.scale * self.scale
        HR = HR[:h_hr, :w_hr, :]

        h_lr = h_hr // self.scale
        w_lr = w_hr // self.scale
        LR = LR[:h_lr, :w_lr, :]

        ### LR_sr: 有噪LR双三次上采样到HR尺寸
        LR_sr = np.array(Image.fromarray(LR).resize((w_hr, h_hr), Image.BICUBIC))

        ### Ref = HR (同场景干净图)
        Ref = HR.copy()

        ### Ref_sr
        Ref_sr = np.array(Image.fromarray(Ref).resize(
            (w_hr // self.scale, h_hr // self.scale), Image.BICUBIC))
        Ref_sr = np.array(Image.fromarray(Ref_sr).resize(
            (w_hr, h_hr), Image.BICUBIC))

        ### 转float32, 归一化到[-1, 1]
        LR = LR.astype(np.float32) / 127.5 - 1.
        LR_sr = LR_sr.astype(np.float32) / 127.5 - 1.
        HR = HR.astype(np.float32) / 127.5 - 1.
        Ref = Ref.astype(np.float32) / 127.5 - 1.
        Ref_sr = Ref_sr.astype(np.float32) / 127.5 - 1.

        sample = {'LR': LR, 'LR_sr': LR_sr, 'HR': HR, 'Ref': Ref, 'Ref_sr': Ref_sr}

        if self.transform:
            sample = self.transform(sample)
        return sample