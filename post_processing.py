import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import re


def gamma_correction(image, gamma=1.0):
    inv_gamma = 1 / gamma
    output = np.uint8(((image / 255) ** inv_gamma) * 255)
    return output


def get_gamma_image(path, output_path, label, gamma_size):
    imglist = os.listdir(f'{path}/{label}')
    for imgName in imglist:
        output_folder = f'{output_path}/{label}'
        os.makedirs(output_folder, exist_ok=True)

        imgPath = f'{path}/{label}/{imgName}'
        img = cv2.imread(imgPath)

        if label == 'fake':
            img = gamma_correction(img, gamma=gamma_size)

        output = f'{output_folder}/{imgName}'
        cv2.imwrite(output, img)

    return 0


def get_gaussian_image(path, output_path, label, kernel):
    imglist = os.listdir(f'{path}/{label}')
    for imgName in imglist:
        output_folder = f'{output_path}/{label}'
        os.makedirs(output_folder, exist_ok=True)

        imgPath = f'{path}/{label}/{imgName}'
        img = cv2.imread(imgPath)

        if label == 'fake':
            img = cv2.GaussianBlur(img, (kernel, kernel), 1.0)

        output = f'{output_folder}/{imgName}'
        cv2.imwrite(output, img)

    return 0


def get_median_image(path, output_path, label, kernel):
    imglist = os.listdir(f'{path}/{label}')
    for imgName in imglist:
        output_folder = f'{output_path}/{label}'
        os.makedirs(output_folder, exist_ok=True)

        imgPath = f'{path}/{label}/{imgName}'
        img = cv2.imread(imgPath)

        if label == 'fake':
            img = cv2.medianBlur(img, kernel, 1.0)

        output = f'{output_folder}/{imgName}'
        cv2.imwrite(output, img)

    return 0


basePath = f''

db = ['A', 'B']
gamma_sizes = [0.8, 0.9, 1.2]
kernel = [3, 9, 11]
quality = [85, 90, 95]
ahe_ = ['one']
classes = ['fake']

for d in db:
    if d == 'A':
        fold = '2-fold'
    else:
        fold = '1-fold'

    iris_B = f'{basePath}/{fold}/{d}'
    iris_B_folders = os.listdir(iris_B)
    print(f'input folder : {iris_B_folders}')

    for gamma_size in gamma_sizes:
        iris_B_output = f'{basePath}/Attack/Gamma/{d}/gamma_{gamma_size}'
        os.makedirs(iris_B_output, exist_ok=True)
        iris_B_output_folders = os.listdir(iris_B_output)

        print(f'output folder : {iris_B_output_folders}')
        get_gamma_image(iris_B, iris_B_output, 'fake', gamma_size)

    for kernel_size in kernel:
        iris_B_output = f'{basePath}/Attack/Gaussian/{d}/blur_{kernel_size}'
        os.makedirs(iris_B_output, exist_ok=True)
        iris_B_output_folders = os.listdir(iris_B_output)

        print(f'output folder : {iris_B_output_folders}')
        get_gaussian_image(iris_B, iris_B_output, 'fake', kernel_size)

    for kernel_size in kernel:
        iris_B_output = f'{basePath}/Attack/Median/{d}/blur_{kernel_size}'
        os.makedirs(iris_B_output, exist_ok=True)
        iris_B_output_folders = os.listdir(iris_B_output)

        print(f'output folder : {iris_B_output_folders}')
        get_median_image(iris_B, iris_B_output, 'fake', kernel_size)
