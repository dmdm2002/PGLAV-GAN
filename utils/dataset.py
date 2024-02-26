import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import PIL.Image as Image

import glob
import os
import random
import numpy as np


class CustomDataset(data.Dataset):
    def __init__(self, cfg, transform):
        super(CustomDataset, self).__init__()
        self.cfg = cfg

        folder_A = glob.glob(f"{os.path.join(self.cfg['dataset_path'], self.cfg['data_folder'][0])}/*")
        folder_B = glob.glob(f"{os.path.join(self.cfg['dataset_path'], self.cfg['data_folder'][0])}/*")

        self.transform = transform

        self.image_path_A = []
        self.image_path_B = []

        """
        inner class image 셔플
        """
        for i in range(len(folder_A)):
            A = glob.glob(f"{folder_A[i]}/*.png")
            B = glob.glob(f"{folder_B[i]}/*.png")
            B = self.shuffle_image(A, B)

            self.image_path_A = self.image_path_A + A
            self.image_path_B = self.image_path_B + B

    def shuffle_image(self, A, B):
        random.shuffle(B)
        for i in range(len(A)):
            if A[i] == B[i]:
                return self.shuffle_image(A, B)
        return B

    def __getitem__(self, index):
        item_A = self.transform(Image.open(self.image_path_A[index]))
        item_B = self.transform(Image.open(self.image_path_B[index]))

        return [item_A, item_B, self.image_path_A[index]]

    def __len__(self):
        return len(self.image_path_A)