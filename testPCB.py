import os
os.environ['MPLCONFIGDIR'] = '/tmp/mpl_cache'

import cv2
import numpy as np
from PIL import Image
import shutil
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from torch.nn.parallel import DataParallel
from torch.cuda.amp import autocast, GradScaler
import torch.nn.functional as F
from torchvision.utils import save_image
import time
from tqdm import tqdm
from models import *
from utils import *
import ultralytics
from ultralytics import YOLO
from ultralytics.utils import ops
from PIL import ImageDraw, ImageFont
from torchmetrics.detection.mean_ap import MeanAveragePrecision
import yaml
import matplotlib
matplotlib.use('Agg')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

test_quality = [0]
num_imgs = 0
mode = 0
if mode == 0:
    quality_subs =   ['ESR', 'QESR', 'SRQE']
    detect_subs = ['_ESR', '_QESR', '_SRQE']
    loss_subs = ['HumanLoss', "MachineLoss", "TotalLoss"]
elif mode == 1:
    quality_subs =   ['SR', 'QE', 'QESR', 'QE+SR', 'SRQE','SR+QE']
    detect_subs = ['_SR', '_QE', '_QESR', '_QE+SR']
    loss_subs = ['NAFNet_Total_01_Machine']
elif mode == 2:
    quality_subs =   ['SR', 'QE', 'QESR', 'QE+SR', 'SRQE','SR+QE']
    detect_subs = ['_SR', '_QE', '_QESR', '_QE+SR']
    loss_subs = ['Human', 'UW_Machine']
# elif mode == 3:
    # quality_subs =   ['ESR', 'QE', 'QESR', 'SRQE', 'QE+SR', 'SR+QE']
    # detect_subs = ['_ESR', '_QE', '_QESR', '_SRQE', '_QE+SR', '_SR+QE']
    # loss_subs = ['HumanLoss', "TotalLoss", 'TotalLoss_01', 'TotalLoss_05', 'TotalLoss_DWA', 'TotalLoss_UncertaintyWeight']
    
hr_img_path = f'output/test_600/images'
hr_label_path = f'output/test_600/labels'
iqe_types = 'small'
imgsz = 64
# iqe = IQE().to(device)
if iqe_types=='small':
    iqe = Enhancer().to(device)
else:
    iqe = Enhancer(in_nc=3, out_nc=3,nf=64, level=2, num_blocks=[2, 4, 4]).to(device)
iqe = NAFNet().to(device)
isr = ESR(scale_factor=4, use_canny=True).to(device)

from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure

def calculate_metrics(img1, img2, max_pixel_value=1.0):
    psnr = PeakSignalNoiseRatio(data_range=1.0).to(device)
    ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    psnr_value = psnr(img1, img2)
    ssim_value = ssim(img1, img2)
    return psnr_value, ssim_value

yolov8 = YOLO('exp/weights/bestyolov8.pt')
detection = yolov8.model
detection.requires_grad_(False)
extra_args = {
    'box': 7.5, 'cls': 0.5, 'dfl': 1.5,
}
detection.args.update(extra_args)

class_names = {
    0: 'mouse_bite',
    1: 'spur',
    2: 'missing_hole',
    3: 'short',
    4: 'open_circuit',
    5: 'spurious_copper'
}
label_colors = {
    0: (200, 0, 0),      # Red (Đỏ đậm hơn chút)
    1: (255, 140, 0),      # Green -> Chuyển sang Xanh lá đậm
    2: (0, 0, 200),      # Blue (Đã ổn)
    3: (200, 150, 0),    # Yellow -> Chuyển sang màu Vàng cam/Nâu vàng (Dark Gold)
    4: (180, 0, 180),    # Magenta -> Tím đậm hơn
    5: (0, 150, 150)     # Cyan -> Chuyển sang màu Teal (Xanh mòng két)
}
def draw_and_save_predictions(image, boxes, labels, scores, class_names, save_path=None, font_path=None):
    """
    Vẽ bounding box, nhãn và độ tự tin lên ảnh, sau đó lưu ảnh nếu cần.
    
    Args:
        image (PIL.Image.Image): Ảnh đầu vào.
        boxes (torch.Tensor): Tensor chứa các bounding box, định dạng [x1, y1, x2, y2].
        labels (torch.Tensor): Tensor chứa nhãn các bounding box.
        scores (torch.Tensor): Tensor chứa điểm tự tin của các bounding box.
        class_names (dict): Mapping từ ID nhãn sang tên lớp.
        save_path (str): Đường dẫn để lưu ảnh (nếu không truyền, ảnh sẽ không được lưu).
        font_path (str): Đường dẫn tới file font TrueType (nếu không truyền sẽ dùng font mặc định).
    
    Returns:
        PIL.Image.Image: Ảnh đã được vẽ bounding box.
    """
    # Tạo bản sao ảnh để vẽ
    draw_image = image.copy()
    draw = ImageDraw.Draw(draw_image)
    
    # Tải font (nếu có)
    if font_path:
        try:
            font = ImageFont.truetype(font_path, size=45)
        except Exception as e:
            print(f"Không thể tải font từ {font_path}. Sử dụng font mặc định.")
            font = ImageFont.load_default()
    else:
        try:
            font = ImageFont.truetype("arial.ttf", size=30)
        except:
            font = ImageFont.load_default(size=20)
    
    # Vẽ từng bounding box
    for box, label, score in zip(boxes, labels, scores):
        x1, y1, x2, y2 = box
        label_text = f"{class_names.get(label.item(), 'Unknown')} {score:.2f}"
        color = label_colors.get(label.item())
        # Vẽ hình chữ nhật
        draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
        
        # Vẽ nhãn với nền
        if hasattr(draw, "textbbox"):
            text_bbox = draw.textbbox((x1, y1), label_text, font=font)
            text_width, text_height = text_bbox[2] - text_bbox[0], text_bbox[3] - text_bbox[1]
        # else:
            # text_width, text_height = draw.textsize(label_text, font=font)
        draw.rectangle(
            [x1, y1 - text_height, x1 + text_width, y1],
            fill=color
        )
        draw.text((x1, y1 - text_height-4), label_text, fill="white", font=font)
    
    # Lưu ảnh nếu `save_path` được cung cấp
    if save_path:
        draw_image.save(save_path)


def yolo_to_xyxy(bboxes, img_w, img_h):
    """
    Convert YOLO-format [cx, cy, w, h] normalized -> [x1, y1, x2, y2] absolute
    """
    cx, cy, w, h = bboxes[:, 0], bboxes[:, 1], bboxes[:, 2], bboxes[:, 3]
    x1 = (cx - w/2) * img_w
    y1 = (cy - h/2) * img_h
    x2 = (cx + w/2) * img_w
    y2 = (cy + h/2) * img_h
    return torch.stack([x1, y1, x2, y2], dim=1)

def run_inference(image, model):
    results = model.predict(image, verbose=False)
    predictions = results[0].boxes
    # Convert to numpy for WBF compatibility
    boxes = predictions.xyxy.cpu().numpy()
    scores = predictions.conf.cpu().numpy()
    labels = predictions.cls.cpu().numpy()
    return boxes, scores, labels

def normalize_boxes(boxes, image_size):
    """Normalize box coordinates to [0, 1] range"""
    width, height = image_size
    normalized_boxes = boxes.copy()
    normalized_boxes[:, [0, 2]] /= width
    normalized_boxes[:, [1, 3]] /= height
    return normalized_boxes

def denormalize_boxes(boxes, image_size):
    """Convert normalized boxes back to pixel coordinates"""
    width, height = image_size
    denormalized_boxes = boxes.copy()
    denormalized_boxes[:, [0, 2]] *= width
    denormalized_boxes[:, [1, 3]] *= height
    return denormalized_boxes



def read_label_file(label_path, img_width, img_height):
    boxes = []
    labels = []
    with open(label_path, "r") as f:
        for line in f:
            box = list(map(float, line.strip().split()))
            labels.append(int(box[0]))
            boxes.append(yolo_to_xyxy(box, img_width, img_height))
    return boxes, labels

import utils
def validate_standalone(sub, iqe, isr, detection, valid_loader, output_image_dir, method_output_dir, image_files):
    isr.eval()
    iqe.eval()
    detection.eval()
    val_ssim, val_psnr = 0, 0
    with torch.no_grad():
        metrics = MeanAveragePrecision(iou_thresholds=[x/100 for x in range(50, 100, 5)], iou_type="bbox", class_metrics=True)
        pbar = tqdm(valid_loader, desc=f'Val : ', unit='batch', leave=False)
        for i, (lr_images, hr_images, labels) in enumerate(pbar):
            lr_images = lr_images.to(device)
            hr_images = hr_images.to(device)  # (B T C H W)s
            
            if sub == 'SR':
                enhanced = isr(lr_images)
            elif sub == 'QE':
                enhanced = iqe(lr_images)   
                enhanced = nn.functional.interpolate(enhanced, scale_factor=4, mode='bicubic', align_corners=False)
            elif sub == 'SRQE' or sub == 'SR+QE':
                enhanced = isr(lr_images)
                enhanced = iqe(enhanced)
            elif sub == 'QESR' or sub == 'QE+SR':
                enhanced = iqe(lr_images)
                enhanced = isr(enhanced)
            pred = detection(enhanced)    
            psnr,ssim = calculate_metrics(enhanced, hr_images)
            
            val_psnr += psnr
            val_ssim += ssim
            preds_for_metric = ops.non_max_suppression(pred,
                                                        conf_thres=0.25, # low conf for mAP
                                                        iou_thres=0.7,
                                                        agnostic=False,
                                                        max_det=300,
                                                        nc=6)
            predictions, targets = utils.post_process(preds_for_metric, labels, 608, 608)             
            # print(f'pred: {predictions}\nlabels: {targets}')
           
            metrics.update(predictions, targets)
            
            output_image_path = os.path.join(output_image_dir, image_files[i])
            output_image = enhanced.squeeze(0).to(device)  # Loại bỏ batch dimension và chuyển tensor sang CPU
            output_image = transforms.ToPILImage()(output_image)  # Chuyển tensor thành ảnh PIL
            # output_image.save(output_image_path, ) 
            save_path = os.path.join(method_output_dir, image_files[i])  # Lưu với cùng tên ảnh
            draw_and_save_predictions(
                image=transforms.ToPILImage()(enhanced.squeeze(0).cpu()).copy(),
                boxes=predictions[0]['boxes'].numpy(),
                labels=predictions[0]['labels'].numpy(),
                scores=predictions[0]['scores'].numpy(),
                class_names=class_names,
                save_path=save_path
            )
    results = metrics.compute()
    avg_psnr = val_psnr/len(valid_loader)
    avg_ssim = val_ssim/len(valid_loader)
    return results, avg_psnr, avg_ssim



isr.to(device)
iqe.to(device)
detection.to(device)
base_output_dir = "runs"
os.makedirs(base_output_dir, exist_ok=True)
for quality in test_quality:
    lr_dir = f'dataset/{quality}_e2e/test/LQ'
    hr_dir = f'dataset/{quality}_e2e/test/HQ'
    labels_dir = f'dataset/{quality}_e2e/test/labels'
    valid_dataset = utils.CombinedTestDataset(lr_dir, hr_dir, labels_dir)
    valid_loader = DataLoader(valid_dataset, collate_fn=utils.combined_collate_fn)
    image_files = sorted(os.listdir(lr_dir))
    os.makedirs(f'runs/QF{quality}', exist_ok=True)
    log_fp = open(f'runs/QF{quality}/E2E_detect_results.txt', 'a')
    # output_image_dir = f'output/E2E_Loss_Training/test_{quality}_150_{sub}_{loss}/images'
    # os.makedirs(output_image_dir, exist_ok = True)
    print(f"Evaluating dataset at QF {quality}...\n")
    log_fp.write(f"Evaluating dataset at QF {quality}..\n")
    log_fp.flush()
    for sub in quality_subs:
        for loss in loss_subs:
            output_image_dir = f'output/E2E_Loss_Training/test_{quality}_150_{sub}_{loss}/images'
            os.makedirs(output_image_dir, exist_ok = True)
            method_output_dir = os.path.join(base_output_dir, f"QF{quality}",f'{sub}_{loss}')
            os.makedirs(method_output_dir, exist_ok=True)
            print(f"Running inference with on {sub}_{loss}...\n")
            ckp_path = os.path.join('exp', f'{sub}_{loss}_QF{quality}', 'best_weight.pth')
            print(ckp_path)
            if os.path.exists(ckp_path):
                ckp = torch.load(ckp_path)
                if 'isr' in ckp:
                    isr.load_state_dict(ckp['isr'])
                    print(f'Load checkpoint for isr from {ckp_path}.....')
                if 'iqe' in ckp:
                    iqe.load_state_dict(ckp['iqe'])
                    print(f'Load checkpoint for iqe from {ckp_path}.....')
            else:
                isrpath = os.path.join('exp', f'SR_{loss}_QF{quality}', 'best_weight.pth')
                iqepath = os.path.join('exp', f'QE_{loss}_QF{quality}', 'best_weight.pth')
                isr.load_state_dict(torch.load(isrpath)['isr'])
                iqe.load_state_dict(torch.load(iqepath)['iqe'])
                print(f'Load checkpoint for iqe and isr from {iqepath} and {isrpath}.....')
                # continue

            # Tính toán kết quả sau khi xử lý tất cả ảnh
            results, avg_psnr, avg_ssim = validate_standalone(sub, iqe, isr, detection, valid_loader, output_image_dir, method_output_dir, image_files)
            print(f"\nResults for on {sub}_{loss}:")
            log_fp.write(f"\nResults for on {sub}_{loss}:\n")
            map50_per_class = results['map_per_class']
            for class_idx, map_value in enumerate(map50_per_class):
                print(f"Class {class_idx}: mAP@50 = {map_value.item():.3f}")
                log_fp.write(f"Class {class_idx}: mAP@50 = {map_value.item():.3f}\n")
                log_fp.flush()
            print(f"Average: mAP50: [{results['map_50']:.3f}], mAP50-95: [{results['map']:.2f}]\n")
            print(f"PSNR/SSIM: [{avg_psnr:.3f}] / [{avg_ssim:.3f}]")
            log_fp.write(f"Average: mAP50: [{results['map_50']:.3f}], mAP50-95: [{results['map']:.2f}]\n")
            log_fp.write(f"PSNR/SSIM: [{avg_psnr:.2f}] / [{avg_ssim:.3f}]\n")
            log_fp.flush()
            print("-" * 50)
