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