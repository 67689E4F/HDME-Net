import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import argparse
import os, cv2
import _init_paths
import dataset
import models

from utils.utils import create_logger, load_config, get_optimizer, get_model_summary, save_checkpoint, save_batch_heatmaps
import numpy as np
from core.function import train
from core.function import validate
from config.cfg import get_cfg_defaults

from easydict import EasyDict as edict
import yaml

def parse_args():
    parser = argparse.ArgumentParser(description='Defocus map estimation')
    # general
    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        default="COCO_shuffle.yaml",  #"hr_ded_fine.yaml"
                        type=str)
    parser.add_argument('--path_data',
                        help='experiment test file name',
                        default="./image/BIT",#"./image/RTF"
                        type=str)
    parser.add_argument('--path_weight',
                        help='experiment test weight',
                        default="./weights/COCO_shuffle.pth",#"./weights/hr_ded_fine.pth"
                        type=str)
    parser.add_argument('--dataset',
                        default="False",
                        help = "False | test | val")
    parser.add_argument('--GPUS',
                        default="False",
                        help = "which gpu to be used, the index of gpus should writen in str, the default value is writen in config.yaml")
    parser.add_argument('--path_save',default='./vis')
    args = parser.parse_args()

    return args
def main():

    args = parse_args()
    with open(os.path.join("./lib/config", args.cfg), 'r', encoding='utf-8') as f:
        cfg = edict(yaml.load(f.read(), Loader=yaml.FullLoader))

    if args.GPUS != "False":
        cfg.GPUS = args.GPUS
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = cfg.GPUS

    logger, final_output_dir, tb_log_dir = create_logger(cfg, args.cfg, 'test')
    logger.info(cfg)

    np.random.seed(cfg.RANDOMSEED)
    torch.manual_seed(cfg.RANDOMSEED)
    torch.cuda.manual_seed_all(cfg.RANDOMSEED)


    # cudnn related setting
    cudnn.benchmark = cfg.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = cfg.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = cfg.CUDNN.ENABLED

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = eval('models.'+cfg.MODEL.NAME+'.get_pose_net')(
        cfg, is_train=False
    )
    data_mean = np.array([0.39989262, 0.44235793, 0.47678594],
                   dtype=np.float32).reshape(1, 1, 3)
    data_std  = np.array([0.17378885, 0.17431373, 0.17731254],
                   dtype=np.float32).reshape(1, 1, 3)
    if args.path_weight == "None":
        model_state_file = os.path.join(
                final_output_dir, 'model_best.pth'
            )
    else:
         model_state_file = args.path_weight
    # logger.info('=> loading model from {}'.format(model_state_file))
    model.load_state_dict(torch.load(model_state_file, map_location=device))
    if len(cfg.GPUS)>1:
        device_list = []
        for id_device in cfg.GPUS.split(","):
            device_list.append(int(id_device))
        model = torch.nn.DataParallel(model, device_ids=device_list).to(device=device)
    else:
        model.to(device)
    model.eval()



    with torch.no_grad():
        if args.dataset != "False":
            valid_dataset = eval('dataset.'+cfg.DATASET.DATASET)(cfg, args.dataset)
            valid_loader = torch.utils.data.DataLoader(
                valid_dataset,
                batch_size=cfg.TEST.BATCH_SIZE_PER_GPU*len(cfg.GPUS),
                shuffle=False,
                num_workers=cfg.WORKERS,
                pin_memory=cfg.PIN_MEMORY
            )
            criterion = torch.nn.MSELoss().to(device)

            perf_indicator = validate(
            cfg, valid_loader, valid_dataset, model, criterion,
            final_output_dir, tb_log_dir, device, logger, epoch=1000)
        else:


            img_list = os.listdir(args.path_data)
            for index_img, img_file in enumerate(img_list):
                img_org = cv2.imread(os.path.join(args.path_data, img_file))
                height, weight, chan = img_org.shape

                # img_cut = img_org[0:height - height%16, 0:weight - weight%16]
                img_cut = img_org[0:height - height%32, 0:weight - weight%32]

                img = (img_cut.astype(np.float32) / 255.)
                img = (img - data_mean) / data_std
                img = img.transpose(2, 0, 1)
                img = torch.from_numpy(img)

                img = img.unsqueeze(0).to(device)
                defocus_map = model(img)


                defocus_map = np.squeeze(defocus_map.cpu().numpy())
                defocus_map[np.where(defocus_map < 0)] = 0
                defocus_map[np.where(defocus_map > 1)] = 1

                
                heatmap_show = np.expand_dims(255 * defocus_map, axis=-1)


                prefix = os.path.join(args.path_save, img_file)

                result_out = np.concatenate([img_cut, heatmap_show.repeat(3, axis=2)] ,axis=1)
                cv2.imwrite(prefix, result_out)
                # cv2.imwrite(prefix, heatmap_show)



if __name__ == "__main__":
        main()
