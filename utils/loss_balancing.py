import math
import random
from typing import List, Optional
import numpy as np
import torch
import torch.nn as nn


class UncertaintyWeighting(nn.Module):
    """Uncertainty weighting (Kendall et al.).
    total = sum(0.5*exp(-log_var[i])*L_i + log_var[i])
    """
    def __init__(self, num_tasks: int):
        super().__init__()
        self.log_vars = nn.Parameter(torch.zeros(num_tasks))


    def forward(self, losses: List[torch.Tensor]) -> torch.Tensor:
        assert len(losses) == self.log_vars.numel()
        total = 0.0
        for i, L in enumerate(losses):
            precision = torch.exp(-self.log_vars[i])
            total = total + 0.5 * precision * L + self.log_vars[i]
        return total
    def get_weights(self) -> np.ndarray:
        """Trả về trọng số hiện tại dưới dạng numpy array"""
        with torch.no_grad():
            weights = 0.5 * torch.exp(-self.log_vars)
        return weights.cpu().numpy()
class DynamicWeightAveraging:
    def __init__(self, num_tasks, T=2.0):
        """
        num_tasks: số lượng task (loss)
        T: temperature cho softmax
        """
        self.num_tasks = num_tasks
        self.T = T
        self.loss_history = [[] for _ in range(num_tasks)]
        self.weights = np.ones(num_tasks) / num_tasks  # ban đầu chia đều

    def update(self, current_losses):
        """
        Cập nhật weight dựa trên losses của epoch hiện tại.
        
        current_losses: list hoặc numpy array [loss_task1, loss_task2, ...]
        """
        for i in range(self.num_tasks):
            self.loss_history[i].append(current_losses[i])

        # Nếu chưa đủ 2 epoch thì giữ trọng số đều
        if len(self.loss_history[0]) < 3:
            self.weights = np.ones(self.num_tasks) / self.num_tasks
            return self.weights

        # Tính ratio r_i^t = L_i(t-1)/L_i(t-2)
        ratios = []
        for i in range(self.num_tasks):
            L_t1 = self.loss_history[i][-2]  # epoch t-1
            L_t2 = self.loss_history[i][-3]  # epoch t-2
            r_i = L_t1 / (L_t2 + 1e-8)       # tránh chia 0
            ratios.append(r_i)

        ratios = np.array(ratios)

        # Softmax với temperature T
        exp_ratios = np.exp(ratios / self.T)
        self.weights = exp_ratios / np.sum(exp_ratios)

        return self.weights

    def get_weights(self):
        """Lấy trọng số hiện tại"""
        return self.weights
