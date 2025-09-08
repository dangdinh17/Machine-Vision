import torch
import torch.nn as nn
import torch.nn.functional as F
class CharbonnierLoss(torch.nn.Module):
    def __init__(self, eps=1e-6):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps

    def forward(self, X, Y):
        diff = X - Y
        error = torch.sqrt(diff * diff + self.eps)
        loss = torch.mean(error)
        return loss
    
class MSELoss(nn.Module):
    def __init__(self):
        super(MSELoss, self).__init__()

    def forward(self, X, Y):
        return F.mse_loss(X, Y)

class FPNLoss(nn.Module):
    def __init__(self, backbone):
        """
        Khởi tạo loss function với backbone tích hợp sẵn.
        
        Args:
            backbone (nn.Module): Mô hình backbone sẽ được sử dụng để trích xuất features
        """
        super(FPNLoss, self).__init__()
        self.backbone = backbone
        self.mse_loss = nn.MSELoss(reduction='mean')
        
        # Đảm bảo backbone không bị cập nhật trọng số khi tính loss
        for param in self.backbone.parameters():
            param.requires_grad = False
        
    def forward(self, x, y):
        """
        Tính MSE giữa các feature maps từ nhiều mức FPN của backbone.
        
        Args:
            x (Tensor): Tensor đầu vào thứ nhất
            y (Tensor): Tensor đầu vào thứ hai
        
        Returns:
            torch.Tensor: Loss value trung bình trên tất cả các mức
        """
        # Đặt backbone ở chế độ evaluation
        self.backbone.eval()
        
        # Trích xuất features từ backbone
        with torch.no_grad():  # Không tính gradient cho backbone
            x_features = self.backbone(x)
            y_features = self.backbone(y)
        
        # Kiểm tra số lượng feature maps phải bằng nhau
        if len(x_features) != len(y_features):
            raise ValueError("Số lượng feature maps từ hai đầu vào phải bằng nhau")
        
        total_loss = 0.0
        num_levels = len(x_features)
        
        # Duyệt qua từng cặp feature maps
        for (_, x_feat), (_, y_feat) in zip(x_features.items(), y_features.items()):
            # Kiểm tra kích thước feature maps
            total_loss += self.mse_loss(x_feat, y_feat)
        
        # Trả về loss trung bình trên tất cả các mức
        return total_loss / num_levels