import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
from tensorboardX import SummaryWriter
import argparse
import os
import _init_paths
import dataset
import models

from utils.utils import create_logger, load_config, get_optimizer, get_model_summary, save_checkpoint
import numpy as np
from core.function import train
from core.function import validate
from core.function import TVLoss
from config.cfg import get_cfg_defaults

from easydict import EasyDict as edict
import yaml

def parse_args():
    parser = argparse.ArgumentParser(description='Defocus map estimation')
    # general
    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        default="COCO_shuffle.yaml",
                        type=str)
    parser.add_argument('--pre_weight',
                        help='pre trained weight',
                        default="None",
                        type=str)

    args = parser.parse_args()

    return args



def main():

    args = parse_args()

    with open(os.path.join("./lib/config", args.cfg), 'r', encoding='utf-8') as f:
        cfg = edict(yaml.load(f.read(), Loader=yaml.FullLoader))
   
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = cfg.GPUS

    logger, final_output_dir, tb_log_dir = create_logger(cfg, args.cfg, 'train')
    logger.info(cfg)
    writer_dict = {
        'writer': SummaryWriter(log_dir=tb_log_dir),
        'train_global_steps': 0,
        'valid_global_steps': 0,
    }

    np.random.seed(cfg.RANDOMSEED)
    torch.manual_seed(cfg.RANDOMSEED)
    torch.cuda.manual_seed_all(cfg.RANDOMSEED)


    # cudnn related setting
    cudnn.benchmark = cfg.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = cfg.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = cfg.CUDNN.ENABLED

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    model = eval('models.'+cfg.MODEL.NAME+'.get_pose_net')(
        cfg, is_train=True
    )

    # define loss function (criterion) and optimizer
    #criterion = torch.nn.L1Loss().to(device)
    criterion = torch.nn.MSELoss().to(device)
    optimizer = get_optimizer(cfg, model)


    checkpoint_file = os.path.join(
        final_output_dir, 'checkpoint.pth'
    )
    
    if cfg.AUTO_RESUME and os.path.exists(checkpoint_file) and args.pre_weight == "None":
        logger.info("=> loading checkpoint '{}'".format(checkpoint_file))
        checkpoint = torch.load(checkpoint_file)
        begin_epoch = checkpoint['epoch']
        best_perf = checkpoint['perf']
        best_acc=checkpoint['acc']
        last_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])

        optimizer.load_state_dict(checkpoint['optimizer'])
        logger.info("=> loaded checkpoint '{}' (epoch {})".format(
            checkpoint_file, checkpoint['epoch']))
    if args.pre_weight != "None":
        # model_state_file = os.path.join(
        #     final_output_dir, 'model_best.pth'
        # )
        model_state_file = args.pre_weight
        logger.info('=> loading model from {}'.format(model_state_file))
        model.load_state_dict(torch.load(model_state_file))
    if len(cfg.GPUS)>1:
        device_list = []
        for id_device in cfg.GPUS.split(","):
            device_list.append(int(id_device))
        model = torch.nn.DataParallel(model, device_ids=device_list).to(device=device)
        num_GPU = len(device_list)
    else:
        model.to(device)
        num_GPU = 1
    
    train_dataset = eval('dataset.'+cfg.DATASET.DATASET)(
        cfg, "train"
    )
    valid_dataset = eval('dataset.'+cfg.DATASET.DATASET)(
        cfg, "val"
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=cfg.TRAIN.BATCH_SIZE_PER_GPU*num_GPU,
        shuffle=cfg.TRAIN.SHUFFLE,
        num_workers=cfg.WORKERS,
        pin_memory=cfg.PIN_MEMORY
    )
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=cfg.TEST.BATCH_SIZE_PER_GPU*num_GPU,
        shuffle=False,
        num_workers=cfg.WORKERS,
        pin_memory=cfg.PIN_MEMORY
    )

    best_perf = 1e5
    best_acc = 0
    best_model = False
    last_epoch = -1
    
    begin_epoch = cfg.TRAIN.BEGIN_EPOCH
    



    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, cfg.TRAIN.LR_STEP, cfg.TRAIN.LR_FACTOR,
        last_epoch=last_epoch
    )


    for epoch in range(begin_epoch, cfg.TRAIN.END_EPOCH):
        

        # train for one epoch
        train(cfg, train_loader, model, criterion, optimizer, epoch,
              final_output_dir, tb_log_dir, writer_dict, device, logger)

        lr_scheduler.step()
        # evaluate on validation set
        perf_indicator, acc = validate(
            cfg, valid_loader, valid_dataset, model, criterion,
            final_output_dir, tb_log_dir, device, logger, writer_dict,epoch
        )


        if acc >= best_acc:
            best_acc = acc
            best_model = True
        else:
            best_model = False

        logger.info('=> saving checkpoint to {}'.format(final_output_dir))
        save_checkpoint({
            'epoch': epoch + 1,
            'model': cfg.MODEL.NAME,
            'state_dict': model.state_dict(),
            'best_state_dict': model.state_dict(),
            'perf': perf_indicator,
            "acc":best_acc,
            'optimizer': optimizer.state_dict(),
        }, best_model, final_output_dir)

       

    final_model_state_file = os.path.join(
            final_output_dir, 'final_state.pth'
        )
    logger.info('=> saving final model state to {}'.format(
            final_model_state_file)
        )
    torch.save(model.state_dict(), final_model_state_file)
    writer_dict['writer'].close()
if __name__ == '__main__':
    main()
