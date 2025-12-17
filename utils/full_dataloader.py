import albumentations as A
import cv2
import numpy as np
import os
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms

class FullCombinedTrainDataset(Dataset):
    def __init__(self, lr_dir, hr_dir, labels_dir, augment=False, imgsz=52):
        self.lr_dir = lr_dir
        self.hr_dir = hr_dir
        self.labels_dir = labels_dir
        self.augment = augment
        
        self.lr_files = sorted([f for f in os.listdir(lr_dir) if f.endswith(('.jpg','.jpeg','.png','.bmp'))])
        self.hr_files = sorted([f for f in os.listdir(hr_dir) if f.endswith(('.jpg','.jpeg','.png','.bmp'))])
        self.label_files = sorted([f for f in os.listdir(labels_dir) if f.endswith('.txt')])

        # --- CẤU HÌNH ALBUMENTATIONS ---
        
        # 1. Transform dùng chung (Geometric): Để đảm bảo HR và LR lật giống hệt nhau
        # Sử dụng additional_targets để xử lý ảnh LR cùng lúc với HR
        if self.augment:
            self.sync_transform = A.Compose([
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
            ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_ids']),
               additional_targets={'image_lr': 'image'},
               is_check_shapes=False) # Định nghĩa key 'image_lr' là một loại ảnh
               
        else:
            self.sync_transform = None

        # 2. Transform cho HR: Resize/Pad lên 608 và xử lý lại Bbox
        self.hr_transform = A.Compose([
            # Đảm bảo ảnh là 600 trước khi pad (nếu ảnh đầu vào chưa chuẩn)
            # Pad lên 608 (Center padding giống Letterbox)
            A.PadIfNeeded(
                min_height=imgsz*4, min_width=imgsz*4,
                border_mode=cv2.BORDER_CONSTANT, value=(114, 114, 114),
                position='center' 
            )
        ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_ids']))

        # 3. Transform cho LR: Resize/Pad lên 152
        self.lr_transform = A.Compose([
            A.PadIfNeeded(
                min_height=imgsz, min_width=imgsz,
                border_mode=cv2.BORDER_CONSTANT, value=(114, 114, 114),
                position='center' # Phải khớp position với HR
            )
        ])

    def __len__(self):
        return len(self.hr_files)
    
    def __getitem__(self, idx):
        lr_path = os.path.join(self.lr_dir, self.lr_files[idx])
        hr_path = os.path.join(self.hr_dir, self.hr_files[idx])
        label_path = os.path.join(self.labels_dir, self.label_files[idx])
        
        # 1. Load ảnh & Chuyển sang Numpy (Albumentations cần Numpy, không phải PIL)
        lr_img = np.array(Image.open(lr_path).convert('RGB'))
        hr_img = np.array(Image.open(hr_path).convert('RGB'))
        
        # 2. Load Labels
        boxes = []
        class_ids = []
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    data = line.strip().split()
                    if len(data) >= 5:
                        class_ids.append(int(data[0]))
                        # cx, cy, w, h
                        boxes.append([float(x) for x in data[1:5]])
        
        # Xử lý trường hợp không có bbox (để tránh lỗi Albumentations)
        if len(boxes) == 0:
            boxes = []
            class_ids = []

        # 3. Áp dụng Augmentation
        
        # Bước A: Đồng bộ Flip (nếu augment=True)
        if self.sync_transform:
            # Truyền cả HR (image) và LR (image_lr) vào cùng 1 pipeline
            transformed = self.sync_transform(
                image=hr_img, 
                image_lr=lr_img, 
                bboxes=boxes, 
                class_ids=class_ids
            )
            hr_img = transformed['image']
            lr_img = transformed['image_lr']
            boxes = transformed['bboxes']
            class_ids = transformed['class_ids']

        # Bước B: Padding riêng biệt (nhưng Deterministic do dùng 'center')
        
        # Xử lý HR + Bbox (lên 608)
        # Lúc này Bbox sẽ được Albumentations tự động tính lại theo kích thước mới 608
        hr_trans = self.hr_transform(image=hr_img, bboxes=boxes, class_ids=class_ids)
        hr_img = hr_trans['image']
        boxes = hr_trans['bboxes']
        class_ids = hr_trans['class_ids'] # Lấy lại class_ids từ transform
        
        # Xử lý LR (lên 152)
        lr_trans = self.lr_transform(image=lr_img)
        lr_img = lr_trans['image']

        # 4. Finalize: Tạo Target Tensor và Convert Image to Tensor
        
        # Chuyển boxes về tensor format cho model
        if len(boxes) > 0:
            boxes_np = np.array(boxes, dtype=np.float32)
            classes_np = np.array(class_ids, dtype=np.float32).reshape(-1, 1)
            # targets format: [class_id, x, y, w, h]
            targets = np.hstack((classes_np, boxes_np))
        else:
            targets = np.zeros((0, 5), dtype=np.float32)

        # Chuyển ảnh về Tensor (Chia 255 và channel first)
        # transforms.ToTensor() xử lý tốt numpy array (H, W, C) -> Tensor (C, H, W) range [0.0, 1.0]
        lr_tensor = transforms.ToTensor()(lr_img)
        hr_tensor = transforms.ToTensor()(hr_img)
        
        return lr_tensor, hr_tensor, torch.tensor(targets)

# Test Dataset (Tương tự nhưng không có Augmentation Random)
class FullCombinedTestDataset(Dataset):
    def __init__(self, lr_dir, hr_dir, labels_dir, imgsz=56):
        self.lr_dir = lr_dir
        self.hr_dir = hr_dir
        self.labels_dir = labels_dir
        
        self.lr_files = sorted([f for f in os.listdir(lr_dir) if f.endswith(('.jpg','.jpeg','.png','.bmp'))])
        self.hr_files = sorted([f for f in os.listdir(hr_dir) if f.endswith(('.jpg','.jpeg','.png','.bmp'))])
        self.label_files = sorted([f for f in os.listdir(labels_dir) if f.endswith('.txt')])

        # Chỉ cần Padding transform, không cần Flip
        self.hr_transform = A.Compose([
            A.PadIfNeeded(
                min_height=imgsz*4, min_width=imgsz*4,
                border_mode=cv2.BORDER_CONSTANT, value=(114, 114, 114),
                position='center'
            )
        ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_ids']))

        self.lr_transform = A.Compose([
            A.PadIfNeeded(
                min_height=imgsz, min_width=imgsz,
                border_mode=cv2.BORDER_CONSTANT, value=(114, 114, 114),
                position='center'
            )
        ])

    def __len__(self):
        return len(self.hr_files)
    
    def __getitem__(self, idx):
        # ... (Phần Load ảnh giống hệt TrainDataset) ...
        lr_path = os.path.join(self.lr_dir, self.lr_files[idx])
        hr_path = os.path.join(self.hr_dir, self.hr_files[idx])
        label_path = os.path.join(self.labels_dir, self.label_files[idx])
        
        lr_img = np.array(Image.open(lr_path).convert('RGB'))
        hr_img = np.array(Image.open(hr_path).convert('RGB'))
        
        boxes = []
        class_ids = []
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    data = line.strip().split()
                    if len(data) >= 5:
                        class_ids.append(int(data[0]))
                        boxes.append([float(x) for x in data[1:5]])

        # Padding
        hr_trans = self.hr_transform(image=hr_img, bboxes=boxes, class_ids=class_ids)
        hr_img = hr_trans['image']
        boxes = hr_trans['bboxes'] # Tọa độ đã chuẩn hóa theo 608
        class_ids = hr_trans['class_ids']
        
        lr_trans = self.lr_transform(image=lr_img)
        lr_img = lr_trans['image']

        # Finalize
        if len(boxes) > 0:
            boxes_np = np.array(boxes, dtype=np.float32)
            classes_np = np.array(class_ids, dtype=np.float32).reshape(-1, 1)
            targets = np.hstack((classes_np, boxes_np))
        else:
            targets = np.zeros((0, 5), dtype=np.float32)

        lr_tensor = transforms.ToTensor()(lr_img)
        hr_tensor = transforms.ToTensor()(hr_img)
        
        return lr_tensor, hr_tensor, torch.tensor(targets)