from torch.optim.lr_scheduler import _LRScheduler

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