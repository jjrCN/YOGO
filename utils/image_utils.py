import torch
import cv2
import numpy as np

def mse(img1, img2):
    return (((img1 - img2)) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)

def psnr(img1, img2):
    mse = (((img1 - img2)) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)
    return 20 * torch.log10(1.0 / torch.sqrt(mse))

def high_frequency_heatmap_fourier(img):
    dft = cv2.dft(np.float32(img), flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)
    rows, cols = img.shape
    crow, ccol = rows // 2, cols // 2

    sigma = 30 # 高斯函数的标准差，sigma越大，保留的高频越多
    x, y = np.meshgrid(np.arange(cols), np.arange(rows))
    d = np.sqrt((x - ccol)**2 + (y - crow)**2)
    high_pass_filter = 1 - np.exp(-(d**2) / (2 * (sigma**2)))
    mask = np.stack([high_pass_filter, high_pass_filter], axis=2)

    fshift = dft_shift * mask

    f_ishift = np.fft.ifftshift(fshift)
    img_back = cv2.idft(f_ishift)
    img_back = cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1]) # 计算幅值

    # img_back_normalized = cv2.normalize(img_back, None, 0, 1, cv2.NORM_MINMAX)
    # high_freq_heatmap_torch = torch.from_numpy(img_back_normalized).float().unsqueeze(0) # (1, H, W)
    high_freq_heatmap_torch = torch.from_numpy(img_back).float().unsqueeze(0) # (1, H, W)

    return high_freq_heatmap_torch
