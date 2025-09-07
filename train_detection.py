
import math
import yaml
import argparse
import torch
import torchvision
import torch.optim as optim
from torchvision import transforms
import os
import os.path as op
from torch.nn.parallel import DistributedDataParallel as DDP
import utils
from collections import OrderedDict
from models import *
from tqdm import tqdm
from pathlib import Path
from comet_ml import Experiment, ExistingExperiment
from types import SimpleNamespace
import numpy as np
from torchmetrics.detection.mean_ap import MeanAveragePrecision

from torchvision.models.detection import *
from torchvision.utils import make_grid



# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def receive_arg():
    parser = argparse.ArgumentParser()
    parser.add_argument('--opt_path', type=str, default='configs/train_detection.yml', help='Path to option YAML file.')
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

    train_dataset = utils.FasterRCNNTrainDataset(opts_dict['dataset']['train']['hr_train'],
                                            opts_dict['dataset']['train']['label_train'],
                                            opts_dict['dataset']['train']['augment']  # augment=True for training
    )
        
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=opts_dict['dataset']['train']['batch_size_per_gpu'],
                                               shuffle=True,
                                               collate_fn=utils.rcnn_collate_fn
    )
    valid_dataset = utils.FasterRCNNTestDataset(opts_dict['dataset']['train']['hr_val'],
                                        opts_dict['dataset']['train']['label_val'],
    )
    valid_loader = torch.utils.data.DataLoader(valid_dataset, collate_fn=utils.rcnn_collate_fn)
    
    batch_size = opts_dict['dataset']['train']['batch_size_per_gpu'] * opts_dict['train']['num_gpu']  # divided by all GPUs
    num_iter_per_epoch = len(train_loader)
    num_epoch = math.ceil(num_iter / num_iter_per_epoch)

    # ==========
    # create model    ,find_unused_parameters=True
    # ==========
    # detection = fasterrcnn_resnet50_fpn(weights=None,weights_backbone=torchvision.models.ResNet50_Weights.DEFAULT, num_classes=opts_dict['train']['num_classes']+1)
    detection = utils.fasterrcnn_resnet18_fpn(num_classes=opts_dict['train']['num_classes']+1)
    detection.to(rank)
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

    # ép self.hyp trong loss thành namespace thay vì dict

    if opts_dict['train']['detection_optim'].pop('type') == 'SGD':
        detection_optimizer = optim.SGD(detection.parameters(), **opts_dict['train']['detection_optim'])
    
    # define scheduler
    if opts_dict['train']['scheduler']['is_on']:
        assert opts_dict['train']['scheduler'].pop('type') == 'CosineAnnealingRestartLR', "Not implemented."
        del opts_dict['train']['scheduler']['is_on']
        opts_dict['train']['scheduler']['is_on'] = True
    
    machine_scheduler = utils.CosineAnnealingRestartLR(detection_optimizer, **opts_dict['train']['detection_scheduler'])
        
    start_epoch, train_step, val_step, best_map = utils.load_checkpoint(detection, detection_optimizer, machine_scheduler, path=opts_dict['train']['load_path'])
    best_map = 0 if best_map==float('inf') else best_map
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
 
    # num_iter_accum = start_iter
    for epoch in range(start_epoch, num_epoch):
        detection.train()
        # if opts_dict['train']['is_dist']:
        #     train_sampler.set_epoch(current_epoch)
        train_loss, train_psnr = 0, 0
        training_timer.restart()
        # # fetch the first batch
        for i, (hr_images, labels) in enumerate(tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epoch}', unit='batch')):
            # get data
            hr_images = list(img.to(rank) for img in hr_images)
            labels = [{k: v.to(rank) for k, v in t.items()} for t in labels]
            # print(type(labels), labels[0])  # check 1 sample target

            loss_dict = detection(hr_images, labels)                 
            total_loss = sum(loss for loss in loss_dict.values())   
            
            detection_optimizer.zero_grad()
            total_loss.backward()  # cal grad
            
            detection_optimizer.step()
            if opts_dict['train']['scheduler']['is_on']:
                machine_scheduler.step()

            train_loss += total_loss.item()
            
            train_step += 1
            if using_comet:
                experiment.log_metric("train_loss", total_loss.item(), step=train_step)

        # # # update learning rate
        detection.eval()
        val_loss, val_psnr = 0, 0
        with torch.no_grad():
            metrics = MeanAveragePrecision(iou_thresholds=[0.5], iou_type="bbox", class_metrics=True)
            # metrics = DetMetrics(save_dir='.', plot=False, names=opts_dict['train']['name_classes'])  # Thay detection.names bằng tên classes

            pbar = tqdm(valid_loader, desc=f'Val Epoch {epoch+1}: ', unit='batch', leave=False)
            for i, (hr_images, labels) in enumerate(pbar):
                # hr_images = hr_images.to(rank)  # (B T C H W)s
                hr_images = list(img.to(rank) for img in hr_images)
                labels = [{k: v.to(rank) for k, v in t.items()} for t in labels]

                pred = detection(hr_images)           
                # loss_dict = detection(hr_images, labels)     
                # print("lossdict: ",type(loss_dict), loss_dict)
                # total_loss = sum(loss for loss in loss_dict.values())   

                # val_loss += total_loss.item()
                val_step += 1

                # predictions, targets = utils.post_process(pred, labels, 608, 608)             

                metrics.update(pred, labels)

                if using_comet:
                    # experiment.log_metric("val_loss", total_loss.item(), step=val_step)
                    if val_step % 20 == 0:
                        # experiment.log_image(transforms.ToPILImage()(hr_images.squeeze(0).cpu()), name="Comparison", step=val_step+1)
                        # experiment.log_image([transforms.ToPILImage()(img.cpu()) for img in hr_images], name="Comparison", step=val_step+1)
                        grid = make_grid(hr_images, nrow=len(hr_images))  # ghép batch thành 1 ảnh
                        img = transforms.ToPILImage()(grid.cpu())
                        experiment.log_image(img, name="Comparison", step=val_step+1)

        results = metrics.compute()

        if using_comet:
            experiment.log_metrics({'avg_train_loss':train_loss/len(train_loader), 'avg_val_loss':val_loss/len(valid_loader)}, step=epoch+1)
            # experiment.log_metrics({'avg_train_psnr':train_psnr/len(train_loader), 'avg_val_psnr':val_psnr/len(valid_loader)}, step=epoch+1)
            experiment.log_metrics({
                "val_map50": results['map_50'],
                "val_map50_95": results['map']
            }, step=epoch+1)

        lr = detection_optimizer.param_groups[0]['lr']
        # Get the training time for the current iteration
        iteration_time = training_timer.get_interval()
        # Estimated training time for the remaining iterations
        remaining_time = (num_iter - train_step) * iteration_time

        msg = (
            f'iterator: [{train_step}]/{num_iter}, '
            f'epoch: [{epoch+1}]/{num_epoch}, '
            f'lr: [{lr * 1e4:.3f}]x1e-4, ' 
            f'train loss: [{train_loss/len(train_loader):.6f}], '
            # f'train psnr: [{train_psnr/len(train_loader):.2f}], '
            f'val loss: [{val_loss/len(valid_loader):.6f}], '
            # f'val psnr: [{val_psnr/len(valid_loader):.2f}], '
            f"val_map50: [{results['map_50']:.4f}] "
            f"val_map50_95: [{results['map']:.4f}] "
            f'iteration time: [{iteration_time:.4f}] s'
        )

        print(msg)
        log_fp.write(msg + '\n')

        if ((epoch % interval_train == 0) or (epoch + 1 == num_epoch)) and (rank == 0):
            # save model

                detection_checkpoint_save_path = (f"{opts_dict['train']['checkpoint_save_path_pre']}"
                                        f"{epoch+1}_detection"
                                        ".pth")
                backbone_checkpoint_save_path = (f"{opts_dict['train']['checkpoint_save_path_pre']}"
                                        f"{epoch+1}_backbone"
                                        ".pth")                
                utils.save_checkpoint(detection, detection_optimizer, machine_scheduler, epoch+1, train_step, val_step, best_map, detection_checkpoint_save_path)
                torch.save(detection.backbone.state_dict(), backbone_checkpoint_save_path)
                # log
                msg = "> detection model saved at epoch {:s}\n".format(str(epoch+1))
                print(msg)
                log_fp.write(msg + '\n')
                log_fp.flush()
        if results['map_50'] > best_map:
            best_map = results['map_50']
            utils.save_checkpoint(detection, detection_optimizer, machine_scheduler, epoch+1, train_step, val_step, best_map, opts_dict['train']['best_detection_weight'])
            torch.save(detection.backbone.state_dict(), os.path.dirname(opts_dict['train']['best_detection_weight']) +f'/best_backbone.pth')

            msg = "> best model saved at epoch {:s}\n".format(str(epoch+1))
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
# The above code is importing the `detection` module from the `torchvision.models`
# package in Python. This module likely contains classes and functions related to
# object detection models in computer vision tasks.
