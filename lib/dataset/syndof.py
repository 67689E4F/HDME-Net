import torch

from torch.utils.data import Dataset
import os, cv2, random, json

import numpy as np
from pathlib import Path
from utils.trans_img import get_affine_transform, affine_transform

class SYNDOF(Dataset):
    def __init__(self, cfg, train_val_test, dataname="SYNDOF"):
        
        self.color_rgb = False
        self.train_val_test = train_val_test

        if train_val_test == "train":
            self.path_img = os.path.join(cfg.DATASET.PATH, train_val_test, dataname, "image")
            self.path_label = os.path.join(cfg.DATASET.PATH, train_val_test, dataname, "blur_map")
            self.label_mode = "DEPTH"
            
        else:
            self.path_img = os.path.join(cfg.DATASET.PATH, "test", "syndof_crop", "image")
            self.path_label = os.path.join(cfg.DATASET.PATH, "test", "syndof_crop", "label")
            self.label_mode = "GRAY"

        self.imgfile_list = os.listdir(self.path_img)

        self.cfg = cfg
        self.heatmap_size = [256, 256] #w h
        self.mean = np.array([0.40789654, 0.44719302, 0.47026115],
                   dtype=np.float32).reshape(1, 1, 3)
        self.std  = np.array([0.28863828, 0.27408164, 0.27809835],
                   dtype=np.float32).reshape(1, 1, 3)


    def __len__(self,):
        return len(self.imgfile_list)

    def __getitem__(self, idx):
        img_filename = self.imgfile_list[idx]
        img = cv2.imread(os.path.join(self.path_img, img_filename), cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
        height, width = img.shape[0], img.shape[1]
        center_img = np.array([img.shape[1] / 2., img.shape[0] / 2.], dtype=np.float32)
        s = np.array([width, height], dtype=np.float32)


        if self.color_rgb:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        label = self.get_images(os.path.join(self.path_label, img_filename), self.label_mode)

        if self.train_val_test == "train":
            cv_blur = random.random()
            if 0< cv_blur <0.2 :
                sigma_blur = random.randint(1,5)
                img = cv2.GaussianBlur(img,[255,255], sigmaX=sigma_blur, sigmaY=sigma_blur)
                label = ((label * 15)**2 + sigma_blur**2)**0.5 / 15.8114

            if 0 < random.random() <= self.cfg.FLIP:
                img = img[:, ::-1, :]
                label = label[:,::-1,:]
                center_img[0] =  width - center_img[0] - 1

            rotation = np.clip(np.random.randn()*self.cfg.ROTATION, -self.cfg.ROTATION, self.cfg.ROTATION) \
                if random.random() <= 0.6 else 0
            trans_input = get_affine_transform(center_img, s, rotation, [width, height])
            img = cv2.warpAffine(img, trans_input, 
                         (width, height),
                         flags=cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_REFLECT
                         )
            label = cv2.warpAffine(label, trans_input, 
                         (width, height),
                         flags=cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_REFLECT
                         )
            label = np.expand_dims(label, axis = 2)
        
            cut_w = np.random.randint(low=0, high=img.shape[1] - self.heatmap_size[0])
            cut_h = np.random.randint(low=0, high=img.shape[0] - self.heatmap_size[1])
            img = img[cut_h:cut_h+self.heatmap_size[1], cut_w:cut_w+self.heatmap_size[0]]
            label = label[cut_h:cut_h+self.heatmap_size[1], cut_w:cut_w+self.heatmap_size[0]]

        img = (img.astype(np.float32) / 255.)
        img = (img - self.mean) / self.std
        img = img.transpose(2, 0, 1)
        label = label.transpose(2, 0, 1)
        label = torch.from_numpy(label)


        return img, label
    
    
    def get_images(self, path_img, mode):
        """ 
        @InProceedings{Lee2019DMENet,
            author    = {Junyong Lee and Sungkil Lee and Sunghyun Cho and Seungyong Lee},
            title     = {Deep Defocus Map Estimation Using Domain Adaptation},
            booktitle = {IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
            year      = {2019}
        } 
        """
        # return scipy.misc.imread(path + file_name).astype(np.float)
        if mode == 'RGB':
            image = (scipy.misc.imread(path_img, mode='RGB')/255.).astype(np.float32)
        elif mode == 'GRAY':
            image = ((cv2.imread(path_img, cv2.IMREAD_UNCHANGED))/255.).astype(np.float32)
            image = np.expand_dims(image, axis = 2)
        elif mode == 'NPY':
            image = np.load(path_img)
            image = image / 3.275
            image = np.expand_dims(image, axis = 2)
        elif mode == 'DEPTH':
            image = (np.float32(cv2.imread(path_img, cv2.IMREAD_UNCHANGED))/10.)[:, :, 1]
            ## If you train the network with the SYNDOF dataset (this is the original SYNDOF dataset) shared in this repository.
            ## The SYNDOF datasets's maximum COC value is 15 and we saved the defocus map with the COC value.
            ## (The paper say that maximum COC value is 28, becuase the blur kernel of orignal SYNDOF dataset visually had the maximaum coc value of 28 when it was generated with max_coc=15.)

            image = image / 15

            ## If you train the network with the new SYNDOF dataset generated with the codes in "https://github.com/codeslake/SYNDOF".
            ## We save the sigma value (max=7) in the code, where
            ## sigma = (max_coc-1)/4, when max_coc = 29, max_sigma = 7

            # image = image / 7

            image = np.expand_dims(image, axis = 2)

        return image





