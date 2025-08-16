import math
from collections import Counter
from torch.optim.lr_scheduler import _LRScheduler

class MultiStepRestartLR(_LRScheduler):
    """ 这是一个带有重新启动机制的多步学习率方案的类，用于 PyTorch 中的优化器。以下是其主要参数和功能：
        optimizer（torch.nn.optimizer）：PyTorch 优化器的实例，它将根据学习率方案进行更新。
        milestones（list）：会降低学习率的迭代步骤。
        gamma（float）：学习率降低的比例，默认为 0.1。
        restarts（list）：重新启动的迭代步骤，默认为 [0]，表示在第0个迭代时重新启动。
        restart_weights（list）：在每次重新启动迭代步骤时的权重，默认为 [1]。
        last_epoch（int）：在 _LRScheduler 中使用的参数，默认为 -1。
    """

    def __init__(self,
                 optimizer,
                 milestones,
                 gamma=0.1,
                 restarts=[0],
                 restart_weights=[1],
                 last_epoch=-1):
        self.milestones = Counter(milestones)
        self.gamma = gamma
        self.restarts = restarts
        self.restart_weights = restart_weights
        assert len(self.restarts) == len(
            self.restart_weights), 'restarts and their weights do not match.'
        super(MultiStepRestartLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch in self.restarts:
            weight = self.restart_weights[self.restarts.index(self.last_epoch)]
            return [
                group['initial_lr'] * weight
                for group in self.optimizer.param_groups
            ]
        if self.last_epoch not in self.milestones:
            return [group['lr'] for group in self.optimizer.param_groups]
        return [
            group['lr'] * self.gamma ** self.milestones[self.last_epoch]
            for group in self.optimizer.param_groups
        ]


def get_position_from_periods(iteration, cumulative_period):
    """从周期列表中获取位置。
    它将返回周期列表中最接近右侧的数字的索引。
    例如，cumulative_period = [100, 200, 300, 400]，
    如果 iteration == 50，则返回 0；
    如果 iteration == 210，则返回 2；
    如果 iteration == 300，则返回 2。
    参数：
        iteration (int)：当前迭代次数。
        cumulative_period (list[int])：累积周期列表。
    返回：
        int：在周期列表中最接近右侧的数字的位置。
    """

    for i, period in enumerate(cumulative_period):
        if iteration <= period:
            return i


class CosineAnnealingRestartLR(_LRScheduler):
    """ Cosine annealing with restarts learning rate scheme.
        余弦退火与重启的学习率方案。
        一个配置的示例：
        periods = [10, 10, 10, 10]
        restart_weights = [1, 0.5, 0.5, 0.5]
        eta_min=1e-7
        它有四个周期，每个周期有10次迭代。在第10、20、30次迭代时，调度器将使用restart_weights重新启动。
        参数：
            optimizer (torch.nn.optimizer)：PyTorch优化器。
            periods (list)：每个余弦退火周期的迭代次数。
            restart_weights (list)：在每次重启迭代时的重启权重。默认值：[1]。
            eta_min (float)：最小学习率。默认值：0。
            last_epoch (int)：在_LRScheduler中使用的参数。默认值：-1。
    """

    def __init__(self,
                 optimizer,
                 periods,
                 restart_weights=[1],
                 eta_min=0,
                 last_epoch=-1):
        self.periods = periods
        self.restart_weights = restart_weights
        self.eta_min = eta_min
        assert (len(self.periods) == len(
            self.restart_weights)), 'periods and restart_weights should have the same length.'
        self.cumulative_period = [sum(self.periods[0:i + 1]) for i in range(0, len(self.periods))]
        super(CosineAnnealingRestartLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        idx = get_position_from_periods(self.last_epoch, self.cumulative_period)
        if idx is None:  # quá số step định nghĩa
            idx = len(self.restart_weights) - 1
        current_weight = self.restart_weights[idx]
        nearest_restart = 0 if idx == 0 else self.cumulative_period[idx - 1]
        current_period = self.periods[idx]

        return [
            self.eta_min + current_weight * 0.5 * (base_lr - self.eta_min) *
            (1 + math.cos(math.pi * ((self.last_epoch - nearest_restart) / current_period)))
            for base_lr in self.base_lrs
        ]


class StepLRScheduler(_LRScheduler):
    def __init__(self, optimizer, step_size=10**5, gamma=0.5, last_epoch=-1):
        self.step_size = step_size
        self.gamma = gamma
        super(StepLRScheduler, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        # Giảm lr sau mỗi step_size step
        if self.last_epoch == 0:
            return [group['initial_lr'] for group in self.optimizer.param_groups]
        if self.last_epoch % self.step_size != 0:
            return [group['lr'] for group in self.optimizer.param_groups]
        return [group['lr'] * self.gamma for group in self.optimizer.param_groups]