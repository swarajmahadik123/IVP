import cv2
import numpy as np
import os

def load_image(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    return img

def save_image(img, path):
    if not path.endswith(('.png', '.jpg', '.jpeg')):
        path += '.png'
    cv2.imwrite(path, img)

def calculate_metrics(original, reconstructed):
    mse = np.mean((original - reconstructed) ** 2)
    psnr = 20 * np.log10(255 / np.sqrt(mse)) if mse != 0 else float('inf')
    return mse, psnr