
import math
import yaml
import argparse
import torch
import torch.optim as optim
import os.path as op
from torch.nn.parallel import DistributedDataParallel as DDP
import utils
from collections import OrderedDict
from models import *
from tqdm import tqdm
from pathlib import Path
from comet_ml import Experiment, ExistingExperiment
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils.loss import v8DetectionLoss
from ultralytics.utils.metrics import DetMetrics
from ultralytics.utils import ops
import ultralytics
from types import SimpleNamespace
import numpy as np
from torchmetrics.detection.mean_ap import MeanAveragePrecision

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def receive_arg():
    parser = argparse.ArgumentParser()
    parser.add_argument('--opt_path', type=str, default='configs/train_e2e.yml', help='Path to option YAML file.')
    parser.add_argument('--local_rank', type=int, default=0, help='Distributed launcher requires.')
    # parser.add_argument('--imgsz', type=int, default=64, help='Image size for training.')
    # parser.add_argument('--exp_name', type=str, default='Enhancer_LR_64', help='Experiment name for logging.')
    args = parser.parse_args()

    with open(args.opt_path, 'r') as fp:
        opts_dict = yaml.load(fp, Loader=yaml.FullLoader)

    opts_dict['opt_path'] = args.opt_path
    opts_dict['train']['rank'] = args.local_rank
    # opts_dict['dataset']['train']['imgsz'] = args.imgsz
    # opts_dict['train']['exp_name'] = args.exp_name
    if opts_dict['train']['exp_name'] is None:
        opts_dict['train']['exp_name'] = utils.get_timestr()

    opts_dict['train']['log_path'] = op.join("exp", opts_dict['train']['exp_name'], "log.log")
    opts_dict['train']['checkpoint_save_path_pre'] = op.join("exp", opts_dict['train']['exp_name'], "ckp_")
    opts_dict['train']['best_iqe_weight'] = op.join("exp", opts_dict['train']['exp_name'], "best_iqe_weight.pth")
    opts_dict['train']['best_detection_weight'] = op.join("exp", opts_dict['train']['exp_name'], "best_detection_weight.pth")

    # opts_dict['train']['num_gpu'] = torch.cuda.device_count()
    if opts_dict['train']['num_gpu'] > 1:
        opts_dict['train']['is_dist'] = True
    else:
        opts_dict['train']['is_dist'] = False
    return opts_dict

def main():
    # ==========
    # parameters
    # ==========

    opts_dict = receive_arg()
    rank = opts_dict['train']['rank']
    unit = opts_dict['train']['criterion']['unit']
    num_iter = int(opts_dict['train']['num_iter'])
    interval_train = int(opts_dict['train']['interval_train'])
    # interval_val = int(opts_dict['train']['interval_val'])

    # ==========
    # comet logging
    # ==========
    using_comet = opts_dict['comet_logging'].pop('using')
    previous_experiment = opts_dict['comet_logging'].pop('previous_experiment')

    if using_comet:
        if previous_experiment:
            experiment = ExistingExperiment(previous_experiment=previous_experiment, **opts_dict['comet_logging'])    
        else:
            experiment = Experiment(**opts_dict['comet_logging']) 

        experiment.set_name(opts_dict['train']['exp_name'])

    # ==========
    # init distributed training
    # ==========
    if opts_dict['train']['is_dist']:
        utils.init_dist(local_rank=rank, backend='nccl')
    pass

    if rank == 0:
        log_dir = op.join("exp", opts_dict['train']['exp_name'])
        if not previous_experiment:
            print("log_dir", log_dir)
            utils.mkdir(log_dir)
        log_fp = open(opts_dict['train']['log_path'], 'a')

        # log all parameters
        msg = (
                f"{'<' * 10} Hello {'>' * 10}\n"
                f"Timestamp: [{utils.get_timestr()}]\n"
                f"\n{'<' * 10} Options {'>' * 10}\n"
                f"{utils.dict2str(opts_dict)}"
                )
        print(msg)
        log_fp.write(msg + '\n')
        log_fp.flush()

    # ==========
    # TO-DO: init tensorboard
    # ==========
    pass

    seed = opts_dict['train']['random_seed']
    utils.set_random_seed(seed + rank)

    torch.backends.cudnn.benchmark = True  # speed up
    # torch.backends.cudnn.deterministic = True  # if reproduce


    # create datasets

    train_dataset = utils.TrainDataset(opts_dict['dataset']['train']['lr_train'],
                                        opts_dict['dataset']['train']['hr_train'],
                                        opts_dict['dataset']['train']['imgsz'],
                                        opts_dict['dataset']['train']['scale'],
                                        opts_dict['dataset']['train']['augment']  # augment=True for training
    )
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=opts_dict['dataset']['train']['batch_size_per_gpu'],
                                               shuffle=True,
    )
    valid_dataset = utils.TestDataset(opts_dict['dataset']['train']['lr_val'],
                                        opts_dict['dataset']['train']['hr_val'],
    )
    valid_loader = torch.utils.data.DataLoader(valid_dataset,)
    
    batch_size = opts_dict['dataset']['train']['batch_size_per_gpu'] * opts_dict['train']['num_gpu']  # divided by all GPUs
    num_iter_per_epoch = len(train_loader)
    num_epoch = math.ceil(num_iter / num_iter_per_epoch)

    # ==========
    # create model    ,find_unused_parameters=True
    # ==========
    if opts_dict['network']['iqe_type']=='Enhancer_Small':
        iqe = Enhancer(in_nc=3, out_nc=3,nf=40, level=2, num_blocks=[1, 2, 2])
    elif opts_dict['network']['iqe_type']=='Enhancer_Large':
        iqe = Enhancer(in_nc=3, out_nc=3,nf=64, level=2, num_blocks=[2, 4, 4])
    else:
        iqe = SwinIR()
        
    if opts_dict['network']['isr_type'] == 'ESR':
        isr = ESR()
    backbone = utils.resnet18_fpn(weights_backbone=None)
    backbone.load_state_dict(torch.load(opts_dict['train']['best_backbone'], map_location='cpu'))
    iqe = iqe.to(rank)
    isr = isr.to(rank)
    backbone = backbone.to(rank)
    # if opts_dict['train']['is_dist']:
    #     model = DDP(model, device_ids=[rank], find_unused_parameters=True)

    # # # # load pre-trained generator
    # ckp_path = opts_dict['train']['load_path']
    # checkpoint = torch.load(ckp_path)
    # state_dict = checkpoint['state_dict']
    #
    # if ('module.' in list(state_dict.keys())[0]) and (not opts_dict['train']['is_dist']):  # multi-gpu pre-trained -> single-gpu training
    #     new_state_dict = OrderedDict()
    #     for k, v in state_dict.items():
    #         name = k[7:]  # remove module
    #         new_state_dict[name] = v
    #     model.load_state_dict(new_state_dict)
    #     print(f'loaded from1 {ckp_path}')
    # elif ('module.' not in list(state_dict.keys())[0]) and (opts_dict['train']['is_dist']):  # single-gpu pre-trained -> multi-gpu training
    #     new_state_dict = OrderedDict()
    #     for k, v in state_dict.items():
    #         name = 'module.' + k  # add module
    #         new_state_dict[name] = v
    #     model.load_state_dict(new_state_dict)
    #     print(f'loaded from2 {ckp_path}')
    # else:  # the same way of training  ,strict=False
    #     model.load_state_dict(state_dict)
    #     print(f'loaded from3 {ckp_path}')

    # ==========
    # define loss func & optimizer & scheduler & scheduler & criterion 损失函数！！！！！！！
    # ==========
    assert opts_dict['train']['loss'].pop('type') == 'CharbonnierLoss', "Not implemented."
    iqe_loss = utils.MSELoss()
    detection_loss = utils.FPNLoss(backbone=backbone)
    # ép self.hyp trong loss thành namespace thay vì dict

        
    alpha = opts_dict['train']['alpha']  # alpha for human loss
    # define optimizer
    assert opts_dict['train']['optim'].pop('type') == 'Adam', "Not implemented."
    iqe_optimizer = optim.Adam(iqe.parameters(), **opts_dict['train']['optim'])
    
    if opts_dict['train']['scheduler']['is_on']:
        assert opts_dict['train']['scheduler'].pop('type') == 'CosineAnnealingRestartLR', "Not implemented."
        del opts_dict['train']['scheduler']['is_on']
        human_scheduler = utils.CosineAnnealingRestartLR(iqe_optimizer, **opts_dict['train']['scheduler'])
        opts_dict['train']['scheduler']['is_on'] = True
    
    
    if op.isfile(opts_dict['train']['best_iqe_model']):
        iqe.load_state_dict(torch.load(opts_dict['train']['best_iqe_model'])['model_state_dict'])
    if op.isfile(opts_dict['train']['best_isr_model']):
        isr.load_state_dict(torch.load(opts_dict['train']['best_isr_model'])['model_state_dict'])
    
    start_epoch, train_step, val_step, best_loss = utils.load_checkpoint(iqe, iqe_optimizer, human_scheduler, path=opts_dict['train']['load_path'])
    best_loss = 0 if best_loss==float('inf') else best_loss
    # display and log
    if rank == 0:
        msg = (
            f"\n{'<' * 10} Dataloader {'>' * 10}\n"
            f"total iters: [{num_iter}]\n"
            f"total epochs: [{num_epoch}]\n"
            f"iter per epoch: [{num_iter_per_epoch}]\n"
            f"start from epoch: [{start_epoch}]"
        )
        print(msg)
        if not previous_experiment:
            log_fp.write(msg + '\n')
            log_fp.flush()

    if opts_dict['train']['is_dist']:
        torch.distributed.barrier()  # all processes wait for ending

    if rank == 0:
        msg = f"\n{'<' * 10} Training {'>' * 10}"
        print(msg)
        log_fp.write(msg + '\n')

        # create timer
        total_train_timer = utils.system.Timer()  # total time of each epoch

    # Create a Timer object before training starts
    training_timer = utils.system.Timer()

    # ==========
    # start training
    # ==========
    # print(any(p.requires_grad for p in detection.parameters()))

    
    # num_iter_accum = start_iter
    isr.eval()
    for epoch in range(start_epoch, num_epoch):
        iqe.train()
        # if opts_dict['train']['is_dist']:
        #     train_sampler.set_epoch(current_epoch)
        train_loss, train_psnr = 0, 0
        training_timer.restart()
        # # # # fetch the first batch
        
        for i, (lr_images, hr_images, targets) in enumerate(tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epoch}', unit='batch')):
            # get data
            lr_images = lr_images.to(rank)
            hr_images = hr_images.to(rank)  # (B T C H W)s
            
            with torch.no_grad():
                isr_out = isr(lr_images)
            enhanced = iqe(isr_out)
            
            human_loss = iqe_loss(enhanced, hr_images)
            machine_loss = detection_loss(hr_images, enhanced)
            # print(human_loss, machine_loss)
            total_loss = human_loss * alpha + machine_loss * (1 - alpha)
            
            iqe_optimizer.zero_grad()  # zero grad
            total_loss.backward()  # cal grad
            
            iqe_optimizer.step()  # update parameters
            if opts_dict['train']['scheduler']['is_on']:
                human_scheduler.step()  # should after optimizer.step()
                
            with torch.no_grad():
                psnr = utils.calculate_psnr(enhanced, hr_images)

            train_loss += total_loss.item()
            train_psnr += psnr
            
            train_step += 1
            if using_comet:
                experiment.log_metric("train_loss", total_loss.item(), step=train_step)
                experiment.log_metric("train_psnr", psnr, step=train_step)
                experiment.log_metric("train_human_loss", human_loss.item(), step=train_step)
                experiment.log_metric("train_machine_loss", machine_loss.item(), step=train_step)

        # # # update learning rate
        iqe.eval()
        val_loss, val_psnr = 0, 0
        with torch.no_grad():
            # metrics = MeanAveragePrecision(iou_thresholds=[0.5], iou_type="bbox", class_metrics=True)
            # metrics = DetMetrics(save_dir='.', plot=False, names=opts_dict['train']['name_classes'])  # Thay detection.names bằng tên classes

            pbar = tqdm(valid_loader, desc=f'Val Epoch {epoch+1}: ', unit='batch', leave=False)
            for i, (lr_images, hr_images, labels) in enumerate(pbar):
                lr_images = lr_images.to(rank)
                hr_images = hr_images.to(rank)  # (B T C H W)s
                
                isr_out = isr(lr_images)
                enhanced = iqe(isr_out)
        
                human_loss = iqe_loss(enhanced, hr_images)
                machine_loss = detection_loss(hr_images, enhanced)

                total_loss = human_loss * alpha + machine_loss* (1 - alpha)
                    
                with torch.no_grad():
                    psnr = utils.calculate_psnr(enhanced, hr_images)

                val_loss += total_loss.item()
                val_psnr += psnr
                val_step += 1


                if using_comet:
                    experiment.log_metric("val_loss", total_loss.item(), step=val_step)
                    experiment.log_metric("val_psnr", psnr, step=val_step)
                    if val_step % 20 == 0:
                        experiment.log_image(utils.concat_triplet_yolo_batch(lr_images, enhanced, hr_images), name="Comparison", step=val_step+1)
        
        # results = metrics.compute()

        if using_comet:
            experiment.log_metrics({'avg_train_loss':train_loss/len(train_loader), 'avg_val_loss':val_loss/len(valid_loader)}, step=epoch+1)
            experiment.log_metrics({'avg_train_psnr':train_psnr/len(train_loader), 'avg_val_psnr':val_psnr/len(valid_loader)}, step=epoch+1)
            
        lr = iqe_optimizer.param_groups[0]['lr']
        # Get the training time for the current iteration
        iteration_time = training_timer.get_interval()
        # Estimated training time for the remaining iterations
        remaining_time = (num_iter - train_step) * iteration_time

        msg = (
            f'iterator: [{train_step}]/{num_iter}, '
            f'epoch: [{epoch+1}]/{num_epoch}, '
            f'lr: [{lr * 1e4:.3f}]x1e-4, ' 
            f'train loss: [{train_loss/len(train_loader):.6f}], '
            f'train psnr: [{train_psnr/len(train_loader):.2f}], '
            f'val loss: [{val_loss/len(valid_loader):.6f}], '
            f'val psnr: [{val_psnr/len(valid_loader):.2f}], '
            # f"val_map50: [{results['map_50']:.4f}] "
            # f"val_map50_95: [{results['map']:.4f}] "
            f'iteration time: [{iteration_time:.4f}] s'
        )

        print(msg)
        log_fp.write(msg + '\n')

        if ((epoch % interval_train == 0) or (epoch + 1 == num_epoch)) and (rank == 0):
            # save model
                iqe_checkpoint_save_path = (f"{opts_dict['train']['checkpoint_save_path_pre']}"
                                        f"{epoch+1}_iqe"
                                        ".pth")
                detection_checkpoint_save_path = (f"{opts_dict['train']['checkpoint_save_path_pre']}"
                                        f"{epoch+1}_detection"
                                        ".pth")
                utils.save_checkpoint(iqe, iqe_optimizer, human_scheduler, epoch+1, train_step, val_step, best_loss, iqe_checkpoint_save_path)
                # utils.save_checkpoint(detection, detection_optimizer, machine_scheduler, epoch+1, train_step, val_step, best_map, detection_checkpoint_save_path)
                # # log
                msg = "> iqe and detection model saved at {:s}\n".format(str(epoch+1))
                print(msg)
                log_fp.write(msg + '\n')
                log_fp.flush()
        if val_loss/len(valid_loader) < best_loss:
            best_loss = val_loss/len(valid_loader)
            utils.save_checkpoint(iqe, iqe_optimizer, human_scheduler, epoch+1, train_step, val_step, best_loss, opts_dict['train']['best_iqe_weight'])
            # utils.save_checkpoint(detection, detection_optimizer, machine_scheduler, epoch+1, train_step, val_step, best_map, opts_dict['train']['best_detection_weight'])

            msg = "> best model saved at {:s}\n".format(str(epoch+1))
            print(msg)
            log_fp.write(msg + '\n')
            log_fp.flush()
        if opts_dict['train']['is_dist']:
            torch.distributed.barrier()  # all processes wait for ending
    experiment.end()

    if rank == 0:
        total_time = total_train_timer.get_interval() / 3600
        total_day = total_train_timer.get_interval() / (24 * 3600)

        msg_hours = "TOTAL TIME: [{:.4f}] h".format(total_time)
        msg_days = "TOTAL TIME: [{:.4f}] days".format(total_day)

        print(msg_hours)
        print(msg_days)
        log_fp.write(msg_hours + '\n')
        log_fp.write(msg_days + '\n')

        goodbye_msg = (f"\n{'<' * 10} Goodbye {'>' * 10}\n"
                       f"Timestamp: [{utils.get_timestr()}]")
        print(goodbye_msg)
        log_fp.write(goodbye_msg + '\n')

        log_fp.close()


if __name__ == '__main__':
    main()
