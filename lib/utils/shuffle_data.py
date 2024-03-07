import os, cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm, trange
import random
import json


def shuffle_img(img):
    h, w, channel = img.shape
    range_h = np.arange(h)
    np.random.shuffle(range_h)
    range_w = np.arange(w)
    np.random.shuffle(range_w)
    index_i, index_j = np.meshgrid(range_h, range_w, indexing="ij")
    img_new = img[index_i, index_j]
    return img_new

def hist_shuffle(img):
    img_new = shuffle_img(img)
    hist_org = cv2.calcHist([img],[0],None,[256],[0,256])
    hist_new = cv2.calcHist([img_new],[0],None,[256],[0,256])

    plt.figure()#新建一个图像
    plt.title("Grayscale Histogram")#图像的标题
    plt.xlabel("Bins")#X轴标签
    plt.ylabel("# of Pixels")#Y轴标签
    plt.plot(hist_org)#画图
    plt.plot(hist_new)#画图
    plt.xlim([0,256])#设置x坐标轴范围
    plt.show()#显示图像

def color_hist(img):
    chans = cv2.split(img)
    colors = ("b","g","r")
    plt.figure()
    plt.title("Flattened Color Histogram")
    plt.xlabel("Bins")
    plt.ylabel("# of Pixels")

    for i, color in enumerate(colors):
        hist = cv2.calcHist([img], [i], None, [256], [0, 256])
        # hist = cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX, -1)  # 对图片进行归一化处理
        plt.plot(hist, color=color)
    plt.xlim(0,256)
    plt.show()


def set_sigma(sigma_map, row_point, col_point, max_sigma):
    sigma = np.random.randint(0, max_sigma)
    sigma_map[0:row_point, 0:col_point] = sigma
    sigma = np.random.randint(0, max_sigma)
    sigma_map[0:row_point, col_point:] = sigma
    sigma = np.random.randint(0, max_sigma)
    sigma_map[row_point:, 0:col_point] = sigma
    sigma = np.random.randint(0, max_sigma)
    sigma_map[row_point:, col_point:] = sigma

    return sigma_map

def find_divide_point(sigma_map, min_area, max_sigma):
    
    row_img, col_img, channel = sigma_map.shape
    # print(row_img, col_img)
    row_0 = np.random.randint(min_area, row_img-min_area)
    col_0 = np.random.randint(min_area, col_img-min_area)
    sigma_map = set_sigma(sigma_map, row_0, col_0, max_sigma)

    return sigma_map
def divide_sigma_map(sigma_map, min_area_0=50,min_area_1=20, max_sigma=15):
    """
    @description  :将离焦深度图进行划分，划分工分为两级，
    第一次根据min_area_0的限制，在图上随机取点，根据点将图像分为四份，
    再根据min_area_1将这四份依次再分四份，共得到16份
    ---------
    @param  :
    -------
    @Returns  :
    -------
    """
    
    
    row_img, col_img, channel = sigma_map.shape
    row_0 = np.random.randint(min_area_0, row_img-min_area_0)
    col_0 = np.random.randint(min_area_0, col_img-min_area_0)

    sigma_map[0:row_0, 0:col_0] = find_divide_point(sigma_map[0:row_0, 0:col_0], min_area_1, max_sigma)
    sigma_map[0:row_0, col_0:] = find_divide_point(sigma_map[0:row_0, col_0:], min_area_1, max_sigma)

    sigma_map[row_0:, 0:col_0] = find_divide_point(sigma_map[row_0:, 0:col_0], min_area_1, max_sigma)

    sigma_map[row_0:, col_0:] = find_divide_point(sigma_map[row_0:, col_0:], min_area_1, max_sigma)
    return sigma_map

def blur_img(img, max_sigma, ksize=[255,255]):
    img_blur_all = [img]
    for sigma in range(1, max_sigma):
        img_blur = cv2.GaussianBlur(img, ksize, sigmaX=sigma, sigmaY=sigma)
        img_blur_all.append(img_blur)
    img_blur_all = np.stack(img_blur_all, axis=0)
    return img_blur_all

def mksigma(img, max_sigma=15):
    #max_sigma 如果是15，则包含15
    row, col, chan= img.shape
    row_mesh,col_mesh = np.meshgrid(np.arange(row),np.arange(col),indexing = 'ij')
    sigma_map = np.zeros((row, col, 1), dtype=int)
    #因为使用range函数的时候，不取最右侧值，所以需要加1，使得max_sigma包含在内
    sigma_map = divide_sigma_map(sigma_map, max_sigma=max_sigma+1)
    img_all = blur_img(img, max_sigma+1)
    # 因为img_all包含sigma为0的图像
    img_blur = img_all[np.squeeze(sigma_map), row_mesh, col_mesh]
    return img_blur, sigma_map


def mk_coco_sigma(path_coco, path_save):
    data_type_list = ["test", "train", "val"]
    for data_type in data_type_list:
        path_img = os.path.join(path_coco, "image", data_type)

        img_list = tqdm(os.listdir(path_img))
        for img_name in img_list:
            if not os.path.exists(os.path.join(path_save, "image", data_type, img_name)):
                img = cv2.imread(os.path.join(path_img, img_name))
                img = shuffle_img(img)
                img_blur, sigma_map = mksigma(img)
                cv2.imwrite(os.path.join(path_save, "image", data_type, img_name), img_blur)
                np.save(os.path.join(path_save, "label", data_type, img_name.replace("jpg","npy")), sigma_map)
                np.save(os.path.join(path_coco, "label", data_type, img_name.replace("jpg","npy")), sigma_map)


def shuffle_data(path_org, path_save):
    data_type_list = ["test", "train", "val"]
    for data_type in data_type_list:
        path_img = os.path.join(path_org, "image", data_type)
        label_dict = {}
        img_list = tqdm(os.listdir(path_img))
        for img_name in img_list:
            if not os.path.exists(os.path.join(path_save, "image", data_type, img_name)):
                img = cv2.imread(os.path.join(path_img, img_name))
                img = shuffle_img(img)
                sigma = random.randint(0, 10)
                label_dict[str(img_name)] = sigma

                cv2.imwrite(os.path.join(path_save, "image", data_type, img_name), img)
        with open(os.path.join(path_save, "label", data_type+".json"), "w") as f:
            f.write(json.dumps(label_dict))

if __name__ == "__main__":
    # mk_coco_sigma("/home/yangpeng/Subject/defocus/dataset/coco/coco_256", "/home/yangpeng/Subject/defocus/dataset/coco/coco_shuffle")
    shuffle_data("/mnt/sdb1/67689e4f/dataset/coco_256", "/mnt/sdb1/67689e4f/dataset/coco_shuffle")


