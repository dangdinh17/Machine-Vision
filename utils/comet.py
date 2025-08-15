import torch
from torchvision import transforms
from PIL import Image


def tensor_batch_to_pil(tensor):    
    if isinstance(tensor, torch.Tensor):
        output_image = tensor.squeeze(0).cpu()  # Loại bỏ batch dimension và chuyển tensor sang CPU
        output_image = transforms.ToPILImage()(output_image)  # Chuyển tensor thành ảnh PIL

    elif isinstance(tensor, Image.Image):
        output_image = tensor
    return output_image

def concat_triplet_batch(lr, iqe_out, out):
    # for i in range(lr.size(0)):
    lr_img = tensor_batch_to_pil(lr)
    iqe_img = tensor_batch_to_pil(iqe_out)
    out_img = tensor_batch_to_pil(out)
    w, h = lr_img.size
    combined = Image.new("RGB", (w*3, h))
    combined.paste(lr_img, (0,0))
    combined.paste(iqe_img, (1*w,0))
    combined.paste(out_img, (2*w,0))
    return combined

