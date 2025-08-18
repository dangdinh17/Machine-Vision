import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.ops import MultiScaleRoIAlign
from torchinfo import summary
import torch.nn as nn

class CustomFasterRCNN(FasterRCNN):
    def __init__(self, num_classes, backbone_name='resnet50', pretrained=True):
        """
        Custom Faster R-CNN implementation
        
        Args:
            num_classes (int): Số lớp (bao gồm background)
            backbone_name (str): Tên backbone ('resnet50', 'resnet101', 'mobilenet_v3_large', ...)
            pretrained (bool): Có sử dụng trọng số pretrained hay không
        """
        # Tạo backbone theo tên
        backbone = self.create_backbone(backbone_name, pretrained)
        
        # Tạo anchor generator
        anchor_generator = AnchorGenerator(
            sizes=((32, 64, 128, 256, 512),),
            aspect_ratios=((0.5, 1.0, 2.0),)
        )
        
        # Tạo ROI pooling
        roi_pooler = MultiScaleRoIAlign(
            featmap_names=['0'],
            output_size=7,
            sampling_ratio=2
        )
        
        # Gọi constructor của lớp cha
        super().__init__(
            backbone,
            num_classes,
            rpn_anchor_generator=anchor_generator,
            box_roi_pool=roi_pooler
        )
    
    def create_backbone(self, name, pretrained):
        """Tạo backbone network dựa trên tên"""
        if name == 'resnet50':
            backbone = torchvision.models.resnet50(pretrained=pretrained)
            # Loại bỏ các lớp cuối (avgpool và fc)
            backbone = nn.Sequential(*list(backbone.children())[:-2])
            backbone.out_channels = 2048
            return backbone
        
        elif name == 'resnet101':
            backbone = torchvision.models.resnet101(pretrained=pretrained)
            backbone = nn.Sequential(*list(backbone.children())[:-2])
            backbone.out_channels = 2048
            return backbone
        
        elif name == 'mobilenet_v3_large':
            backbone = torchvision.models.mobilenet_v3_large(pretrained=pretrained)
            # Lấy các lớp convolution
            backbone = backbone.features
            # MobileNetV3 có output channels là 960
            backbone.out_channels = 960
            return backbone
        
        else:
            raise ValueError(f"Backbone '{name}' không được hỗ trợ")

# Sử dụng class để tạo mô hình
def create_model(num_classes, backbone='resnet50'):
    return CustomFasterRCNN(num_classes=num_classes, backbone_name=backbone)

model = create_model(10)
print(model)