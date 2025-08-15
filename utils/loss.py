import torch
import torch.nn as nn
import torch.nn.functional as F
import ultralytics
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