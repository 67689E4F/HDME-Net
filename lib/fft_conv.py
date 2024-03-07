import cv2
import numpy as np
import matplotlib.pylab as plt


def gauss_kernel(size, sigma):
    x, y = np.mgrid[-size//2 + 1:size//2 + 1, -size//2 + 1:size//2 + 1]
    # x = np.expand_dims(x_data, axis=-1)
    # y = np.expand_dims(y_data, axis=-1)

    g_kernel = np.exp(-((x**2+ y**2)/(2*sigma**2)), dtype="float32")
    return g_kernel


img = cv2.imread("/home/yangpeng/Subject/defocus/OIDMENet/lib/IMG_2159.JPG", 0)
# img = cv2.resize(img, (32, 32))
gk_filter_3 = gauss_kernel(3, 2)
gk_filter = gauss_kernel(32, 2)

img_blur = cv2.filter2D(img, -1, gk_filter_3)
cv2.imshow("img_blur", img_blur)
cv2.waitKey(0)
cv2.destroyAllWindows()

img_fft = np.fft.fft2(img)
img_fft_shift = np.fft.fftshift(img_fft)

gk_filter_fft = np.fft.fft2(gk_filter)
gk_filter_fft_shift = np.fft.fftshift(gk_filter_fft)


img_fft_blur = img_fft_shift * gk_filter_fft_shift
# img_fft_blur = img_fft_blur + np.imag(img_fft_shift)
#傅里叶逆变换
ishift = np.fft.ifftshift(img_fft_blur)
iimg = np.fft.ifft2(ishift)
iimg = np.abs(iimg).astype(np.uint8)

error_img = iimg - img_blur
mean_img = np.mean(error_img)