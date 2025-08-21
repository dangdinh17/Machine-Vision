import torch
from torchmetrics import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure

def calculate_psnr_ssim(img1, img2, max_pixel_value=1.0):
    
    assert isinstance(img1, torch.Tensor) and isinstance(img2, torch.Tensor), "Input images must be PyTorch tensors"
    assert img1.shape == img2.shape, "Input images must have the same shape"
    assert img1.dtype == torch.float32 and img2.dtype == torch.float32, "Input images must be of type float32"  
    assert img1.device == img2.device, "Input images must be on the same device"

    psnr = PeakSignalNoiseRatio(data_range=1.0).to(img1.device)
    ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(img1.device)
    psnr_value = psnr(img1, img2)
    ssim_value = ssim(img1, img2)
    return psnr_value, ssim_value

def calculate_psnr(img1, img2):
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    max_pixel = 1.0
    psnr = 20 * torch.log10(max_pixel / torch.sqrt(mse))
    return psnr.item()

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

def post_process(preds_for_metric, labels, img_w=640, img_h=640):
    # preds_for_metric là list chứa 1 tensor [N, 6]
    pred_tensor = preds_for_metric[0]  # [x1, y1, x2, y2, conf, cls]
    boxes = pred_tensor[:, 0:4]
    scores = pred_tensor[:, 4]
    labels_pred = pred_tensor[:, 5].long()

    predictions = [{
        'boxes': boxes.detach().cpu(),
        'scores': scores.detach().cpu(),
        'labels': labels_pred.detach().cpu()
    }]

    # convert labels['bboxes'] (YOLO format) -> xyxy
    gt_boxes = yolo_to_xyxy(labels['bboxes'], img_w, img_h)
    gt_labels = labels['cls'].squeeze(1).long()

    targets = [{
        'boxes': gt_boxes.detach().cpu(),
        'labels': gt_labels.detach().cpu()
    }]

    return predictions, targets