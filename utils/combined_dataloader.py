import torchvision.transforms.functional as TF
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import os
import random
import numpy as np
import torch

class CombinedTrainDataset(Dataset):
    def __init__(self, lr_dir, hr_dir, labels_dir, augment=False):
        """
        Dataset for a pipeline that combines Image Restoration and Object Detection.
        
        Args:
            lr_dir (str): Directory containing low-resolution images (152x152)
            hr_dir (str): Directory containing high-resolution images (608x608)
            labels_dir (str): Directory containing annotations (YOLO format)
            augment (bool): Apply data augmentation (vertical and horizontal flips)
        """
        self.lr_dir = lr_dir
        self.hr_dir = hr_dir
        self.labels_dir = labels_dir
        self.augment = augment
        
        # List of image files (assume filenames are consistent across directories)
        self.lr_files = sorted([f for f in os.listdir(lr_dir) if f.endswith(('.jpg','.jpeg','.png','.bmp'))])
        self.hr_files = sorted([f for f in os.listdir(hr_dir) if f.endswith(('.jpg','.jpeg','.png','.bmp'))])
        self.label_files = sorted([f for f in os.listdir(labels_dir) if f.endswith('.txt')])

        # Fixed input sizes
        self.lr_size = 152
        self.hr_size = 608

    def __len__(self):
        return len(self.hr_files)
    
    def __getitem__(self, idx):
        lr_path = os.path.join(self.lr_dir, self.lr_files[idx])
        hr_path = os.path.join(self.hr_dir, self.hr_files[idx])
        label_path = os.path.join(self.labels_dir, self.label_files[idx])
        
        # Load images
        lr_img = Image.open(lr_path).convert('RGB')
        hr_img = Image.open(hr_path).convert('RGB')
        
        # Resize if sizes don’t match expected dimensions
        if lr_img.size != (self.lr_size, self.lr_size):
            lr_img = lr_img.resize((self.lr_size, self.lr_size), Image.BILINEAR)
        
        if hr_img.size != (self.hr_size, self.hr_size):
            hr_img = hr_img.resize((self.hr_size, self.hr_size), Image.BILINEAR)
        
        # Load annotations
        boxes = []
        class_ids = []
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    data = line.strip().split()
                    if len(data) >= 5:
                        class_id = int(data[0])
                        cx = float(data[1])  # normalized center x
                        cy = float(data[2])  # normalized center y
                        w = float(data[3])   # normalized width
                        h = float(data[4])   # normalized height
                        
                        boxes.append([cx, cy, w, h])
                        class_ids.append(class_id)
        
        boxes = np.array(boxes, dtype=np.float32)
        class_ids = np.array(class_ids, dtype=np.int64)
        
        # Apply synchronized augmentations
        if self.augment:
            lr_img, hr_img, boxes = self.apply_augmentations(lr_img, hr_img, boxes)
        
        # Build target tensor
        if len(boxes) > 0:
            targets = np.hstack((class_ids.reshape(-1, 1), boxes))
        else:
            targets = np.zeros((0, 5), dtype=np.float32)
        
        # Convert to tensor
        lr_img = transforms.ToTensor()(lr_img)
        hr_img = transforms.ToTensor()(hr_img)
        
        return lr_img, hr_img, torch.tensor(targets)
    
    def apply_augmentations(self, lr_img, hr_img, boxes):
        # Random horizontal flip
        if random.random() > 0.5:
            lr_img = TF.hflip(lr_img)
            hr_img = TF.hflip(hr_img)
            if len(boxes) > 0:
                boxes[:, 0] = 1.0 - boxes[:, 0]  # flip x coordinate
        
        # Random vertical flip
        if random.random() > 0.5:
            lr_img = TF.vflip(lr_img)
            hr_img = TF.vflip(hr_img)
            if len(boxes) > 0:
                boxes[:, 1] = 1.0 - boxes[:, 1]  # flip y coordinate
        
        return lr_img, hr_img, boxes


class CombinedTestDataset(Dataset):
    def __init__(self, lr_dir, hr_dir, labels_dir):
        """
        Dataset for evaluation in the pipeline combining Image Restoration and Object Detection.
        
        Args:
            lr_dir (str): Directory containing low-resolution images (152x152)
            hr_dir (str): Directory containing high-resolution images (608x608)
            labels_dir (str): Directory containing annotations (YOLO format)
        """
        self.lr_dir = lr_dir
        self.hr_dir = hr_dir
        self.labels_dir = labels_dir
        
        # List of image files (assume filenames are consistent across directories)
        self.lr_files = sorted([f for f in os.listdir(lr_dir) if f.endswith(('.jpg','.jpeg','.png','.bmp'))])
        self.hr_files = sorted([f for f in os.listdir(hr_dir) if f.endswith(('.jpg','.jpeg','.png','.bmp'))])
        self.label_files = sorted([f for f in os.listdir(labels_dir) if f.endswith('.txt')])

        # Fixed input sizes
        self.lr_size = 152
        self.hr_size = 608

    def __len__(self):
        return len(self.hr_files)
    
    def __getitem__(self, idx):
        lr_path = os.path.join(self.lr_dir, self.lr_files[idx])
        hr_path = os.path.join(self.hr_dir, self.hr_files[idx])
        label_path = os.path.join(self.labels_dir, self.label_files[idx])
        
        # Load images
        lr_img = Image.open(lr_path).convert('RGB')
        hr_img = Image.open(hr_path).convert('RGB')
        
        # Resize if sizes don’t match expected dimensions
        if lr_img.size != (self.lr_size, self.lr_size):
            lr_img = lr_img.resize((self.lr_size, self.lr_size), Image.BILINEAR)
        
        if hr_img.size != (self.hr_size, self.hr_size):
            hr_img = hr_img.resize((self.hr_size, self.hr_size), Image.BILINEAR)
        
        # Load annotations
        boxes = []
        class_ids = []
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    data = line.strip().split()
                    if len(data) >= 5:
                        class_id = int(data[0])
                        cx = float(data[1])  # normalized center x
                        cy = float(data[2])  # normalized center y
                        w = float(data[3])   # normalized width
                        h = float(data[4])   # normalized height
                        
                        boxes.append([cx, cy, w, h])
                        class_ids.append(class_id)
        
        boxes = np.array(boxes, dtype=np.float32)
        class_ids = np.array(class_ids, dtype=np.int64)
        
        # Build target tensor
        if len(boxes) > 0:
            targets = np.hstack((class_ids.reshape(-1, 1), boxes))
        else:
            targets = np.zeros((0, 5), dtype=np.float32)
        
        # Convert to tensor
        lr_img = transforms.ToTensor()(lr_img)
        hr_img = transforms.ToTensor()(hr_img)
        
        return lr_img, hr_img, torch.tensor(targets)

def collate_fn(batch):
    lr, hr, tlist = zip(*batch)
    lr = torch.stack(lr, 0)
    hr = torch.stack(hr, 0)
    cls, bboxes, batch_idx = [], [], []
    for i, t in enumerate(tlist):
        if t.numel():
            cls.append(t[:, 0:1].float())
            bboxes.append(t[:, 1:5].float()) # normalized xywh
            batch_idx.append(torch.full((t.shape[0], 1), i, dtype=torch.float32))
    if cls:
        labels = {"cls": torch.cat(cls, 0),
        "bboxes": torch.cat(bboxes, 0),
        "batch_idx": torch.cat(batch_idx, 0)}
    else:
        z = torch.zeros((0, 1), dtype=torch.float32)
        labels = {"cls": z, 
                  "bboxes": z.new_zeros((0, 4)),
                  "batch_idx": z}
    return lr, hr, labels
# def collate_fn(batch):
#     """Collate function to handle variable number of objects per image"""
#     lr_imgs, hr_imgs, targets = [], [], []
    
#     for lr, hr, t in batch:
#         lr_imgs.append(lr)
#         hr_imgs.append(hr)
#         targets.append(t)
    
#     return torch.stack(lr_imgs), torch.stack(hr_imgs), targets

# def collate_fn(batch):
#     """
#     Custom collate function to match YOLOv8 loss format.
#     Each batch will output:
#     - lr_imgs: tensor [B, 3, H, W]
#     - hr_imgs: tensor [B, 3, H, W]
#     - targets: dict with keys 'batch_idx', 'cls', 'bboxes'
#     """
#     lr_imgs, hr_imgs, labels = zip(*batch)  # labels: [N, 5] (cls, cx, cy, w, h)

#     lr_imgs = torch.stack(lr_imgs, 0)
#     hr_imgs = torch.stack(hr_imgs, 0)

#     batch_idx = []
#     cls = []
#     bboxes = []

#     for i, label in enumerate(labels):
#         if label.numel() > 0:  # Có bbox
#             batch_idx.append(torch.full((label.shape[0],), i, dtype=torch.float32))
#             cls.append(label[:, 0].long())
#             bboxes.append(label[:, 1:].float())

#     if len(batch_idx) > 0:
#         targets = {
#             "batch_idx": torch.cat(batch_idx, 0),
#             "cls": torch.cat(cls, 0),
#             "bboxes": torch.cat(bboxes, 0),
#         }
#     else:
#         targets = {
#             "batch_idx": torch.zeros((0,), dtype=torch.int64),
#             "cls": torch.zeros((0,), dtype=torch.int64),
#             "bboxes": torch.zeros((0, 4), dtype=torch.float32),
#         }

#     return lr_imgs, hr_imgs, targets
