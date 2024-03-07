import cv2, os, math
import numpy as np
from tqdm import tqdm
from pathlib import Path

def crop_coco(path_orgdata, path_save, size_save=(256,256)):
    """
    @description  :裁剪图片
    ---------
    @param  :
    -------
    @Returns  :
    -------
    """
    size_save_half = np.array(size_save)//2
    file_list = os.listdir(path_orgdata)
    num_all = len(file_list)
    id_trains = [0, int(num_all * 0.8)]
    id_test = [int(num_all * 0.8), int(num_all * 0.9)]

    np.random.seed(1221)
    np.random.shuffle(file_list)

    with tqdm(total=num_all) as pbar:
        for id_img, name_img in enumerate(file_list):
            img = cv2.imread(os.path.join(path_orgdata, name_img))
            row, col, channel = img.shape
            if row< size_save[0] or col < size_save[1]:
                continue
            else:
                img_crop = img[row//2 - size_save_half[0] : row//2 + size_save_half[0], col//2 - size_save_half[1] : col//2 + size_save_half[1], :]
                if id_img<id_trains[1]:
                    cv2.imwrite(os.path.join(path_save, "train", name_img), img_crop)
                elif id_test[0] <= id_img < id_test[1]:
                    cv2.imwrite(os.path.join(path_save, "test", name_img), img_crop)
                else:
                    cv2.imwrite(os.path.join(path_save, "val", name_img), img_crop)
            pbar.update(1)
    
def general_label():
    pass


def div_test(path_dataset, dst_size, path_save):
    """
    @description  :将测试集裁剪为小块，小块之间有重叠，确保每个位置都有保留
    ---------
    @path_dataset  :数据集路径
    @dst_size  :小块的大小 row, clo
    @path_save  :保存路径
    -------
    @Returns  :
    -------
    """
    
    path_r = Path(path_dataset)
    path_img = path_r / "image"
    path_label = path_r / "label"
    path_save = Path(path_save)
    Path(path_save / "image").mkdir(exist_ok=True)
    Path(path_save / "label").mkdir(exist_ok=True)


    for single_img_path in path_img.iterdir():
        img_name = single_img_path.name
        single_label_path = path_label / img_name

        img = cv2.imread(str(single_img_path), cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
        label = cv2.imread(str(single_label_path), cv2.IMREAD_UNCHANGED)

        height, weight  = label.shape

        num_h = math.ceil(height / dst_size[0])
        num_w = math.ceil(weight / dst_size[1])

        step_h = math.floor((height - dst_size[0])/(num_h-1))
        step_w = math.floor((weight - dst_size[1])/(num_w-1))
        for mv_w in range(num_w):
            for mv_h in range(num_h):
                img_cut = img[mv_h*step_h:mv_h*step_h+dst_size[0], mv_w*step_w:mv_w*step_w+dst_size[1], :]
                label_cut = label[mv_h*step_h:mv_h*step_h+dst_size[0], mv_w*step_w:mv_w*step_w+dst_size[1]]
                save_name = "{}_h{}w{}.png".format(single_img_path.stem, mv_h, mv_w)
                p_img_s = path_save / "image" / save_name
                p_label_s = path_save / "label" / save_name
          
                cv2.imwrite(str(p_img_s), img_cut)
                cv2.imwrite(str(p_label_s), label_cut)

if __name__ == "__main__":

    div_test("/mnt/sdb1/67689e4f/dataset/SYNDOF/test/SYNDOF/",
             [256, 256], 
             "/mnt/sdb1/67689e4f/dataset/SYNDOF/test/syndor_crop/")
    
