import torchvision.transforms.functional as TF
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import os
import random
import numpy as np
import torch

class TrainDataset(Dataset):
    def __init__(self, lr_dir, hr_dir, imgsz, scale=4, augment=False):
        self.lr_files = sorted(os.listdir(lr_dir))
        self.hr_files = sorted(os.listdir(hr_dir))
        self.lr_dir = lr_dir
        self.hr_dir = hr_dir
        self.scale = scale  # Tỷ lệ phóng đại, ví dụ: 4x
        self.imgsz = imgsz  # Kích thước ảnh đầu vào, có thể thay đổi tùy theo yêu cầ
        self.augment = augment

    def __getitem__(self, idx):
        lr_image = Image.open(os.path.join(self.lr_dir, self.lr_files[idx])).convert('RGB')
        hr_image = Image.open(os.path.join(self.hr_dir, self.hr_files[idx])).convert('RGB')
        lr_image, hr_image = self.transform_fn(lr_image, hr_image, self.imgsz, self.scale, self.augment)
        return lr_image, hr_image
    
    def transform_fn(self, lr_img, hr_img, imgsz, scale, augment=False):
        if isinstance(imgsz, int):
            if imgsz > lr_img.size[0] or imgsz > lr_img.size[1]:
                lr_img = lr_img.resize((imgsz, imgsz), Image.BILINEAR)
                hr_img = hr_img.resize((imgsz*scale, imgsz*scale), Image.BILINEAR)
            else:
                i, j, h, w = transforms.RandomCrop.get_params(lr_img, output_size=(imgsz//scale, imgsz//scale))
                lr_img = TF.crop(lr_img, i, j, h, w)
                hr_img = TF.crop(hr_img, i * scale, j * scale, h * scale, w * scale)
        
        if augment:
            # Áp dụng cùng một phép lật ngang
            if random.random() > 0.5:
                lr_img = TF.hflip(lr_img)
                hr_img = TF.hflip(hr_img)

            # Áp dụng cùng một phép lật dọc
            if random.random() > 0.5:
                lr_img = TF.vflip(lr_img)
                hr_img = TF.vflip(hr_img)

        # Chuyển sang tensor
        lr_img = TF.to_tensor(lr_img)
        hr_img = TF.to_tensor(hr_img)

        return lr_img, hr_img
    
    def __len__(self):
        return len(self.lr_files)


class TestDataset(Dataset):
    def __init__(self, lr_dir, hr_dir):
        self.lr_files = sorted(os.listdir(lr_dir))
        self.hr_files = sorted(os.listdir(hr_dir))
        self.lr_dir = lr_dir
        self.hr_dir = hr_dir
        
    def transform_fn(self, lr_img, hr_img):
        lr_img = TF.to_tensor(lr_img)
        hr_img = TF.to_tensor(hr_img)

        return lr_img, hr_img
    
    def __len__(self):
        return len(self.lr_files)


    def __getitem__(self, idx):
        lr_image = Image.open(os.path.join(self.lr_dir, self.lr_files[idx])).convert('RGB')
        hr_image = Image.open(os.path.join(self.hr_dir, self.hr_files[idx])).convert('RGB')
        lr_image, hr_image = self.transform_fn(lr_image, hr_image)
        return lr_image, hr_image
    
