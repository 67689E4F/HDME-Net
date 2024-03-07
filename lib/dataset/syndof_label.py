import cv2
import numpy as np
import shutil
import os

path_data = "/home/yangpeng/Subject/defocus/dataset/syndof/test/SYNDOF"

path_gt = os.path.join(path_data, "gt")
path_image = os.path.join(path_data, "image")
path_label = os.path.join(path_data, "label")

img_list = sorted(os.listdir(path_image))
gt_list = sorted(os.listdir(path_gt))
for i, file_name in enumerate(img_list):
    shutil.copy(os.path.join(path_gt, gt_list[i]), os.path.join(path_label, file_name))
