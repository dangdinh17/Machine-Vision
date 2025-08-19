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
    
class CombinedTrainDataset(Dataset):
    def __init__(self, lr_dir, hr_dir, labels_dir, augment=False):
        """
        Dataset cho pipeline kết hợp Image Restoration và Object Detection
        
        Args:
            lr_dir (str): Thư mục ảnh chất lượng thấp (152x152)
            hr_dir (str): Thư mục ảnh chất lượng cao (608x608)
            labels_dir (str): Thư mục chứa annotations (YOLO format)
            augment (bool): Áp dụng data augmentation (vflip và hflip)
        """
        self.lr_dir = lr_dir
        self.hr_dir = hr_dir
        self.labels_dir = labels_dir
        self.augment = augment
        
        # Danh sách file ảnh (giả sử tên file giống nhau giữa các thư mục)
        self.lr_files = sorted([f for f in os.listdir(lr_dir) if f.endswith(('.jpg','.jpeg','.png','.bmp'))])
        self.hr_files = sorted([f for f in os.listdir(hr_dir) if f.endswith(('.jpg','.jpeg','.png','.bmp'))])
        self.label_files = sorted([f for f in os.listdir(labels_dir) if f.endswith('.txt')])

        # Kích thước cố định theo yêu cầu
        self.lr_size = 152
        self.hr_size = 608

    def __len__(self):
        return len(self.valid_files)
    
    def __getitem__(self, idx):
        lr_path = os.path.join(self.lr_dir, self.lr_files[idx])
        hr_path = os.path.join(self.hr_dir, self.hr_files[idx])
        label_path = os.path.join(self.labels_dir, self.label_files[idx])
        
        # Đọc ảnh
        lr_img = Image.open(lr_path).convert('RGB')
        hr_img = Image.open(hr_path).convert('RGB')
        
        # Kiểm tra kích thước
        if lr_img.size != (self.lr_size, self.lr_size):
            lr_img = lr_img.resize((self.lr_size, self.lr_size), Image.BILINEAR)
        
        if hr_img.size != (self.hr_size, self.hr_size):
            hr_img = hr_img.resize((self.hr_size, self.hr_size), Image.BILINEAR)
        
        # Đọc annotations
        boxes = []
        class_ids = []
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    data = line.strip().split()
                    if len(data) >= 5:
                        class_id = int(data[0])
                        cx = float(data[1])  # Tọa độ trung tâm x (chuẩn hóa)
                        cy = float(data[2])  # Tọa độ trung tâm y (chuẩn hóa)
                        w = float(data[3])   # Chiều rộng (chuẩn hóa)
                        h = float(data[4])   # Chiều cao (chuẩn hóa)
                        
                        boxes.append([cx, cy, w, h])
                        class_ids.append(class_id)
        
        boxes = np.array(boxes, dtype=np.float32)
        class_ids = np.array(class_ids, dtype=np.int64)
        
        # Áp dụng augmentation đồng bộ
        if self.augment:
            lr_img, hr_img, boxes = self.apply_augmentations(lr_img, hr_img, boxes)
        
        # Chuẩn bị target tensor
        if len(boxes) > 0:
            targets = np.hstack((class_ids.reshape(-1, 1), boxes))
        else:
            targets = np.zeros((0, 5), dtype=np.float32)
        
        # Chuyển sang tensor
        lr_img = transforms.ToTensor()(lr_img)
        hr_img = transforms.ToTensor()(hr_img)
        
        return lr_img, hr_img, torch.tensor(targets)
    
    def apply_augmentations(self, lr_img, hr_img, boxes):
        # Random horizontal flip (lật ngang)
        if random.random() > 0.5:
            lr_img = TF.hflip(lr_img)
            hr_img = TF.hflip(hr_img)
            if len(boxes) > 0:
                boxes[:, 0] = 1.0 - boxes[:, 0]  # Đảo ngược tọa độ x
        
        # Random vertical flip (lật dọc)
        if random.random() > 0.5:
            lr_img = TF.vflip(lr_img)
            hr_img = TF.vflip(hr_img)
            if len(boxes) > 0:
                boxes[:, 1] = 1.0 - boxes[:, 1]  # Đảo ngược tọa độ y
        
        return lr_img, hr_img, boxes
class CombinedTestDataset(Dataset):
    def __init__(self, lr_dir, hr_dir, labels_dir):
        """
        Dataset cho pipeline kết hợp Image Restoration và Object Detection
        
        Args:
            lr_dir (str): Thư mục ảnh chất lượng thấp (152x152)
            hr_dir (str): Thư mục ảnh chất lượng cao (608x608)
            labels_dir (str): Thư mục chứa annotations (YOLO format)
            augment (bool): Áp dụng data augmentation (vflip và hflip)
        """
        self.lr_dir = lr_dir
        self.hr_dir = hr_dir
        self.labels_dir = labels_dir
        
        # Danh sách file ảnh (giả sử tên file giống nhau giữa các thư mục)
        self.lr_files = sorted([f for f in os.listdir(lr_dir) if f.endswith(('.jpg','.jpeg','.png','.bmp'))])
        self.hr_files = sorted([f for f in os.listdir(hr_dir) if f.endswith(('.jpg','.jpeg','.png','.bmp'))])
        self.label_files = sorted([f for f in os.listdir(labels_dir) if f.endswith('.txt')])

        # Kích thước cố định theo yêu cầu
        self.lr_size = 152
        self.hr_size = 608

    def __len__(self):
        return len(self.valid_files)
    
    def __getitem__(self, idx):
        lr_path = os.path.join(self.lr_dir, self.lr_files[idx])
        hr_path = os.path.join(self.hr_dir, self.hr_files[idx])
        label_path = os.path.join(self.labels_dir, self.label_files[idx])
        
        # Đọc ảnh
        lr_img = Image.open(lr_path).convert('RGB')
        hr_img = Image.open(hr_path).convert('RGB')
        
        # # Kiểm tra kích thước
        # if lr_img.size != (self.lr_size, self.lr_size):
        #     lr_img = lr_img.resize((self.lr_size, self.lr_size), Image.BILINEAR)
        
        # if hr_img.size != (self.hr_size, self.hr_size):
        #     hr_img = hr_img.resize((self.hr_size, self.hr_size), Image.BILINEAR)
        
        # Đọc annotations
        boxes = []
        class_ids = []
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    data = line.strip().split()
                    if len(data) >= 5:
                        class_id = int(data[0])
                        cx = float(data[1])  # Tọa độ trung tâm x (chuẩn hóa)
                        cy = float(data[2])  # Tọa độ trung tâm y (chuẩn hóa)
                        w = float(data[3])   # Chiều rộng (chuẩn hóa)
                        h = float(data[4])   # Chiều cao (chuẩn hóa)
                        
                        boxes.append([cx, cy, w, h])
                        class_ids.append(class_id)
        
        boxes = np.array(boxes, dtype=np.float32)
        class_ids = np.array(class_ids, dtype=np.int64)
        
        # Chuẩn bị target tensor
        if len(boxes) > 0:
            targets = np.hstack((class_ids.reshape(-1, 1), boxes))
        else:
            targets = np.zeros((0, 5), dtype=np.float32)
        
        # Chuyển sang tensor
        lr_img = transforms.ToTensor()(lr_img)
        hr_img = transforms.ToTensor()(hr_img)
        
        return lr_img, hr_img, torch.tensor(targets)
    
def collate_fn(batch):
    """Hàm xử lý batch với số lượng object khác nhau"""
    lr_imgs, hr_imgs, targets = [], [], []
    
    for lr, hr, t in batch:
        lr_imgs.append(lr)
        hr_imgs.append(hr)
        targets.append(t)
    
    return torch.stack(lr_imgs), torch.stack(hr_imgs), targets