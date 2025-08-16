import os
import math
import random
import numpy as np
import torch
import torch.nn as nn
import torch.fft as fft
import torch.distributed as dist
import torch.multiprocessing as tmp
from functools import partial
from torch.utils.data.sampler import Sampler
from torch.utils.data import DataLoader as DataLoader


# 设置随机种子
def set_random_seed(seed):
    """Set random seeds."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# 初始化分布式训练的环境
def init_dist(local_rank=0, backend='nccl'):
    tmp.set_start_method('spawn')
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend=backend)


# 获取分布式训练的信息
def get_dist_info():
    if dist.is_available():
        initialized = dist.is_initialized()
    else:
        initialized = False

    if initialized:
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        rank = 0
        world_size = 1

    return rank, world_size


# ==========
# Dataloader
# ==========
class DistSampler(Sampler):
    """Sampler that restricts data loading to a subset of the dataset.
    Modified from torch.utils.data.distributed.DistributedSampler
    Support enlarging the dataset for iteration-based training, for saving
    time when restart the dataloader after each epoch.

    Args:
        dataset (torch.utils.data.Dataset): Dataset used for sampling.
        num_replicas (int | None): Number of processes participating in
            the training. It is usually the world_size.
        rank (int | None): Rank of the current process within num_replicas.
        ratio (int): Enlarging ratio. Default: 1.
    """

    def __init__(
            self, dataset, num_replicas=None, rank=None, ratio=1
    ):
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        # enlarged by ratio, and then divided by num_replicas
        self.num_samples = math.ceil(
            len(self.dataset) * ratio / self.num_replicas
        )
        self.total_size = self.num_samples * self.num_replicas

    def __iter__(self):
        # deterministically shuffle based on ite_epoch
        g = torch.Generator()
        g.manual_seed(self.epoch)
        indices = torch.randperm(self.total_size, generator=g).tolist()

        # enlarge indices
        dataset_size = len(self.dataset)
        indices = [v % dataset_size for v in indices]

        # ==========subsample
        # e.g., self.rank=1, self.total_size=4, self.num_replicas=2
        # indices = indices[1:4:2] = indices[i for i in [1, 3]]
        # for the other worker, indices = indices[i for i in [0, 2]]
        # ==========
        indices = indices[self.rank:self.total_size:self.num_replicas]
        assert len(indices) == self.num_samples

        return iter(indices)

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        """For shuffling data at each epoch. See train.py."""
        self.epoch = epoch


def create_dataloader(dataset, opts_dict, sampler=None, phase='train', seed=None):
    """Create dataloader."""
    if phase == 'train':
        # >I don't know why BasicSR have to detect `is_dist`
        dataloader_args = dict(
            dataset=dataset,
            batch_size=opts_dict['dataset']['train']['batch_size_per_gpu'],
            shuffle=False,  # sampler will shuffle at train.py
            num_workers=opts_dict['dataset']['train']['num_worker_per_gpu'],
            sampler=sampler,
            drop_last=True,
            pin_memory=True
        )
        if sampler is None:
            dataloader_args['shuffle'] = True
        dataloader_args['worker_init_fn'] = partial(
            _worker_init_fn,
            num_workers=opts_dict['dataset']['train']['num_worker_per_gpu'],
            rank=opts_dict['train']['rank'],
            seed=seed
        )

    elif phase == 'val':
        dataloader_args = dict(
            dataset=dataset,
            batch_size=1,
            shuffle=False,
            num_workers=0,
            pin_memory=False
        )

    return DataLoader(**dataloader_args)


def _worker_init_fn(worker_id, num_workers, rank, seed):
    # func for torch.utils.data.DataLoader
    # set the worker seed to num_workers * rank + worker_id + seed
    worker_seed = num_workers * rank + worker_id + seed
    np.random.seed(worker_seed)
    random.seed(worker_seed)


# ==========
# Loss & Metrics
# ==========
# 计算图像或信号之间的差异时常用的损失函数，它在平方损失的基础上引入了平方根以提高对异常值的鲁棒性。
class CharbonnierLoss(torch.nn.Module):
    def __init__(self, eps=1e-6):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps

    def forward(self, X, Y):
        diff = torch.add(X, -Y)
        error = torch.sqrt(diff * diff + self.eps)
        loss = torch.mean(error)
        return loss


class FFTCharbonnierLoss(torch.nn.Module):
    def __init__(self, lambda_param=1.0, eps=1e-6):
        super(FFTCharbonnierLoss, self).__init__()
        self.charbonnier_loss = CharbonnierLoss(eps=eps)
        self.lambda_param = lambda_param

    def forward(self, prediction, gt):
        fft_pred = fft.fft2(prediction, dim=(-2, -1))
        fft_gt = fft.fft2(gt, dim=(-2, -1))

        # Calculate amplitude and phase
        amp_pred = torch.sqrt(fft_pred.real**2 + fft_pred.imag**2)
        phase_pred = torch.atan2(fft_pred.imag, fft_pred.real)
        amp_gt = torch.sqrt(fft_gt.real**2 + fft_gt.imag**2)
        phase_gt = torch.atan2(fft_gt.imag, fft_gt.real)


        # Calculate Charbonnier Loss on amplitude and phase
        loss_amp = torch.mean(self.charbonnier_loss(amp_pred, amp_gt))
        loss_phase = torch.mean(self.charbonnier_loss(phase_pred, phase_gt))

        # Combine amplitude and phase losses
        total_loss = loss_amp + self.lambda_param * loss_phase

        return total_loss


class FFTL1Loss(torch.nn.Module):
    def __init__(self, lambda_param=1.0):
        super(FFTL1Loss, self).__init__()
        self.lambda_param = lambda_param

    def forward(self, pred, gt):
        fft_pred = torch.fft.fft2(pred, dim=(-2, -1))
        fft_gt = torch.fft.fft2(gt, dim=(-2, -1))

        amp_pred = torch.sqrt(fft_pred.real**2 + fft_pred.imag**2)
        phase_pred = torch.atan2(fft_pred.imag, fft_pred.real)

        amp_gt = torch.sqrt(fft_gt.real**2 + fft_gt.imag**2)
        phase_gt = torch.atan2(fft_gt.imag, fft_gt.real)

        # Calculate L1 Loss on amplitude and phase
        loss_amp = torch.mean(torch.abs(amp_pred - amp_gt))
        loss_phase = torch.mean(torch.abs(phase_pred - phase_gt))

        # Combine amplitude and phase losses
        total_loss = loss_amp + self.lambda_param * loss_phase

        return total_loss

class FFTLoss(torch.nn.Module):   #  FFT 损失函数  基于 L1 的
    def __init__(self):
        super(FFTLoss, self).__init__()
        self.L1 = nn.L1Loss(reduction='mean')
    def forward(self, img1, img2):
        img1=torch.stack([torch.fft.fft2(img1, dim=(-2, -1)).real, torch.fft.fft2(img1, dim=(-2, -1)).imag], -1)
        img2=torch.stack([torch.fft.fft2(img2, dim=(-2, -1)).real, torch.fft.fft2(img2, dim=(-2, -1)).imag], -1)
        return self.L1(img1,img2)

class PSNR(torch.nn.Module):
    def __init__(self, eps=1e-6):
        super(PSNR, self).__init__()
        self.eps = eps
        self.mse_func = nn.MSELoss()

    def forward(self, X, Y):
        mse = self.mse_func(X, Y)
        psnr = 10 * math.log10(1 / (mse.item() + self.eps))
        return psnr


# ==========
# Scheduler
# ==========
