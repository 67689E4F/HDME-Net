import torch

from torch.utils.data import Dataset
import os, cv2, random, json
from utils.utils import gauss_kernel
import numpy as np
import imgaug.augmenters as iaa

class COCO(Dataset):
    def __init__(self, cfg, train_val_test):
        self.path_img = os.path.join(cfg.DATASET.PATH, "image", train_val_test)
        self.imgfile_list = os.listdir(self.path_img)
        self.path_label = os.path.join(cfg.DATASET.PATH, "label", train_val_test)

        self.cfg = cfg
        # self.mean = np.array([0.40789654, 0.44719302, 0.47026115],
        #            dtype=np.float32).reshape(1, 1, 3)
        # self.std  = np.array([0.28863828, 0.27408164, 0.27809835],
        #            dtype=np.float32).reshape(1, 1, 3)
        
        self.mean = np.array([0.39989262, 0.44235793, 0.47678594],
                    dtype=np.float32).reshape(1, 1, 3)
        self.std  = np.array([0.17378885, 0.17431373, 0.17731254],
                   dtype=np.float32).reshape(1, 1, 3)
        self.color_rgb = False



        self.seq = iaa.Sequential([
        iaa.SaltAndPepper(0.01), #椒盐噪声
        iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 15), per_channel=0.01), #高斯白噪声
        # iaa.PiecewiseAffine(scale=(0.0, 0.01)), #变形
        iaa.AdditivePoissonNoise(5), #泊松噪声
        ])

    def __len__(self,):
        return len(self.imgfile_list)

    def __getitem__(self, idx):
        img_filename = self.imgfile_list[idx]
        img = cv2.imread(os.path.join(self.path_img, img_filename))
        # height, weight = img.shape[0], img.shape[1]
        # s = np.array([width, height], dtype=np.float32)

        if random.random() < 0.1:
            img = self.seq(image=img)        
        if self.color_rgb:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        #label = np.load(os.path.join(self.path_label, img_filename.replace("jpg", "npy")))
        #label = label.astype(np.float32) / 15

        label = cv2.imread(os.path.join(self.path_label, img_filename.replace("jpg", "png")), 0)
        label = np.expand_dims(label, axis=-1)
        label = label.astype(np.float32) / 255.
        
        img = img.astype(np.float32)/255.
        img = (img - self.mean) / self.std
        img = img.transpose(2, 0, 1)
        label = label.transpose(2, 0, 1)
        label = torch.from_numpy(label)

        return img, label
    


