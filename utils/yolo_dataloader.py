import torchvision.transforms.functional as TF
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import os
import random
import numpy as np
import torch

class YOLOTrainDataset(Dataset):
    def __init__(self,  img_dir, labels_dir, augment=False):
        """
        Dataset for a pipeline that combines Image Restoration and Object Detection.
        
        Args:
            img_dir (str): Directory containing high-resolution images (608x608)
            labels_dir (str): Directory containing annotations (YOLO format)
            augment (bool): Apply data augmentation (vertical and horizontal flips)
        """
        self.img_dir = img_dir
        self.labels_dir = labels_dir
        self.augment = augment
        
        # List of image files (assume filenames are consistent across directories)
        self.img_files = sorted([f for f in os.listdir(img_dir) if f.endswith(('.jpg','.jpeg','.png','.bmp'))])
        self.label_files = sorted([f for f in os.listdir(labels_dir) if f.endswith('.txt')])

        # Fixed input sizes
        self.lr_size = 152
        self.img_size = 608

    def __len__(self):
        return len(self.img_files)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_files[idx])
        label_path = os.path.join(self.labels_dir, self.label_files[idx])
        
        # Load images
        img_img = Image.open(img_path).convert('RGB')

        
        if img_img.size != (self.img_size, self.img_size):
            img_img = img_img.resize((self.img_size, self.img_size), Image.BILINEAR)
        
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
        
        # Apply syncimgonized augmentations
        if self.augment:
            lr_img, img_img, boxes = self.apply_augmentations(lr_img, img_img, boxes)
        
        # Build target tensor
        if len(boxes) > 0:
            targets = np.hstack((class_ids.reshape(-1, 1), boxes))
        else:
            targets = np.zeros((0, 5), dtype=np.float32)
        
        # Convert to tensor
        lr_img = transforms.ToTensor()(lr_img)
        img_img = transforms.ToTensor()(img_img)
        
        return lr_img, img_img, torch.tensor(targets)
    
    def apply_augmentations(self,img_img, boxes):
        # Random horizontal flip
        if random.random() > 0.5:
            img_img = TF.hflip(img_img)
            if len(boxes) > 0:
                boxes[:, 0] = 1.0 - boxes[:, 0]  # flip x coordinate
        
        # Random vertical flip
        if random.random() > 0.5:
            img_img = TF.vflip(img_img)
            if len(boxes) > 0:
                boxes[:, 1] = 1.0 - boxes[:, 1]  # flip y coordinate
        
        return img_img, boxes


class YOLOTestDataset(Dataset):
    def __init__(self, img_dir, labels_dir):
        """
        Dataset for evaluation in the pipeline combining Image Restoration and Object Detection.
        
        Args:
            img_dir (str): Directory containing high-resolution images (608x608)
            labels_dir (str): Directory containing annotations (YOLO format)
        """
        self.img_dir = img_dir
        self.labels_dir = labels_dir
        
        # List of image files (assume filenames are consistent across directories)
        self.img_files = sorted([f for f in os.listdir(img_dir) if f.endswith(('.jpg','.jpeg','.png','.bmp'))])
        self.label_files = sorted([f for f in os.listdir(labels_dir) if f.endswith('.txt')])

        # Fixed input sizes
        self.lr_size = 152
        self.img_size = 608

    def __len__(self):
        return len(self.img_files)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_files[idx])
        label_path = os.path.join(self.labels_dir, self.label_files[idx])
        
        # Load images
        img_img = Image.open(img_path).convert('RGB')
        
        if img_img.size != (self.img_size, self.img_size):
            img_img = img_img.resize((self.img_size, self.img_size), Image.BILINEAR)
        
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
        img_img = transforms.ToTensor()(img_img)
        
        return lr_img, img_img, torch.tensor(targets)

def collate_fn(batch):
    img, tlist = zip(*batch)
    img = torch.stack(img, 0)
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
    return img, labels