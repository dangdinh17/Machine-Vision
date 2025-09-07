import torchvision.transforms.functional as TF
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import os
import random
import numpy as np
import torch

class FasterRCNNTrainDataset(Dataset):
    def __init__(self, img_dir, labels_dir, augment=False):
        """
        Dataset for Faster R-CNN training from YOLO format annotations.
        
        Args:
            img_dir (str): Directory containing images
            labels_dir (str): Directory containing annotations (YOLO format: cls cx cy w h)
            augment (bool): Apply data augmentation (flip)
        """
        self.img_dir = img_dir
        self.labels_dir = labels_dir
        self.augment = augment
        
        self.img_files = sorted([f for f in os.listdir(img_dir) if f.endswith(('.jpg','.jpeg','.png','.bmp'))])
        self.label_files = sorted([f for f in os.listdir(labels_dir) if f.endswith('.txt')])
        
        self.img_size = 608

    def __len__(self):
        return len(self.img_files)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_files[idx])
        label_path = os.path.join(self.labels_dir, self.label_files[idx])

        # Load image
        img = Image.open(img_path).convert('RGB')
        w, h = img.size

        if img.size != (self.img_size, self.img_size):
            img = img.resize((self.img_size, self.img_size), Image.BILINEAR)
            w, h = self.img_size, self.img_size

        # Load YOLO labels
        boxes = []
        labels = []
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    data = line.strip().split()
                    if len(data) >= 5:
                        class_id = int(data[0])
                        cx = float(data[1]) * w
                        cy = float(data[2]) * h
                        bw = float(data[3]) * w
                        bh = float(data[4]) * h

                        # Convert xywh -> xmin, ymin, xmax, ymax
                        x_min = cx - bw/2
                        y_min = cy - bh/2
                        x_max = cx + bw/2
                        y_max = cy + bh/2

                        boxes.append([x_min, y_min, x_max, y_max])
                        labels.append(class_id + 1)  # +1 vì FasterRCNN coi 0 là background

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)

        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([idx])
        }

        # Apply augmentations (optional)
        if self.augment:
            img, target = self.apply_augmentations(img, target)

        # Convert image to tensor
        img = transforms.ToTensor()(img)
        
        return img, target
    
    def apply_augmentations(self, img, target):
        w, h = img.size
        boxes = target["boxes"]

        # Horizontal flip
        if random.random() > 0.5:
            img = TF.hflip(img)
            if len(boxes) > 0:
                boxes[:, [0, 2]] = w - boxes[:, [2, 0]]  # flip xmin, xmax
                target["boxes"] = boxes

        # Vertical flip
        if random.random() > 0.5:
            img = TF.vflip(img)
            if len(boxes) > 0:
                boxes[:, [1, 3]] = h - boxes[:, [3, 1]]  # flip ymin, ymax
                target["boxes"] = boxes

        return img, target
    
class FasterRCNNTestDataset(Dataset):
    def __init__(self, img_dir, labels_dir, augment=False):
        """
        Dataset for Faster R-CNN training from YOLO format annotations.
        
        Args:
            img_dir (str): Directory containing images
            labels_dir (str): Directory containing annotations (YOLO format: cls cx cy w h)
            augment (bool): Apply data augmentation (flip)
        """
        self.img_dir = img_dir
        self.labels_dir = labels_dir
        self.augment = augment
        
        self.img_files = sorted([f for f in os.listdir(img_dir) if f.endswith(('.jpg','.jpeg','.png','.bmp'))])
        self.label_files = sorted([f for f in os.listdir(labels_dir) if f.endswith('.txt')])
        
        self.img_size = 608

    def __len__(self):
        return len(self.img_files)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_files[idx])
        label_path = os.path.join(self.labels_dir, self.label_files[idx])

        # Load image
        img = Image.open(img_path).convert('RGB')
        w, h = img.size

        if img.size != (self.img_size, self.img_size):
            img = img.resize((self.img_size, self.img_size), Image.BILINEAR)
            w, h = self.img_size, self.img_size

        # Load YOLO labels
        boxes = []
        labels = []
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    data = line.strip().split()
                    if len(data) >= 5:
                        class_id = int(data[0])
                        cx = float(data[1]) * w
                        cy = float(data[2]) * h
                        bw = float(data[3]) * w
                        bh = float(data[4]) * h

                        # Convert xywh -> xmin, ymin, xmax, ymax
                        x_min = cx - bw/2
                        y_min = cy - bh/2
                        x_max = cx + bw/2
                        y_max = cy + bh/2

                        boxes.append([x_min, y_min, x_max, y_max])
                        labels.append(class_id + 1)  # +1 vì FasterRCNN coi 0 là background

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)

        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([idx])
        }

        # Convert image to tensor
        img = transforms.ToTensor()(img)
        
        return img, target


def rcnn_collate_fn(batch):
    return tuple(zip(*batch))
