import os
import random

import torch
import torch.nn as nn
import torchvision.transforms as transforms


class ImagePool:
    """This class implements an image buffer that stores previously generated images.
    This buffer enables us to update discriminators using a history of generated images
    rather than the ones produced by the latest generators.
    """

    """https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/util/image_pool.py"""

    def __init__(self, pool_size):
        """Initialize the ImagePool class
        Parameters:
            pool_size (int) -- the size of image buffer, if pool_size=0, no buffer will be created
        """
        self.pool_size = pool_size
        if self.pool_size > 0:  # create an empty pool
            self.num_imgs = 0
            self.images = []

    def query(self, images):
        """Return an image from the pool.
        Parameters:
            images: the latest generated images from the generator
        Returns images from the buffer.
        By 50/100, the buffer will return input images.
        By 50/100, the buffer will return images previously stored in the buffer,
        and insert the current images to the buffer.
        """
        if self.pool_size == 0:  # if the buffer size is 0, do nothing
            return images
        return_images = []
        for image in images:
            image = torch.unsqueeze(image.data, 0)
            if self.num_imgs < self.pool_size:   # if the buffer is not full; keep inserting current images to the buffer
                self.num_imgs = self.num_imgs + 1
                self.images.append(image)
                return_images.append(image)
            else:
                p = random.uniform(0, 1)
                if p > 0.5:  # by 50% chance, the buffer will return a previously stored image, and insert the current image into the buffer
                    random_id = random.randint(0, self.pool_size - 1)  # randint is inclusive
                    tmp = self.images[random_id].clone()
                    self.images[random_id] = image
                    return_images.append(tmp)
                else:       # by another 50% chance, the buffer will return the current image
                    return_images.append(image)
        return_images = torch.cat(return_images, 0)   # collect all the images and return
        return return_images


class CkpManager:
    def __init__(self, cfg):
        super(CkpManager, self).__init__()
        self.cfg = cfg

    def init_weight(self, module):
        if isinstance(module, nn.Conv2d):
            nn.init.normal_(module.weight.data, 0.0, 0.02)
            if module.bias is not None:
                nn.init.constant(module.bias.data, 0.0)

        elif isinstance(module, nn.BatchNorm2d):
            nn.init.normal_(module.weight.data, 1.0, 0.02)
            if module.bias is not None:
                nn.init.constant(module.bias.data, 0.0)

        elif isinstance(module, nn.Linear):
            nn.init.normal_(module.weight.data, 0.0, 0.02)
            if module.bias is not None:
                nn.init.constant(module.bias.data, 0.0)

    def load_ckp(self, netG_A2B, netG_B2A, netD_A, netD_B, imagenet=False):
        if imagenet:
            print(f"BackBone Pretrained [IMAGE NET] Loading..")
            return netG_A2B, netG_B2A, netD_A, netD_B, 0

        else:
            if self.cfg['do_ckp_load']:
                print(f"Check Point [{self.cfg['load_ckp_epoch']}] Loading...")
                ckp = torch.load(f"{self.cfg['output_ckp']}/{self.cfg['load_ckp_epoch']}.pth")
                netG_A2B.load_state_dict(ckp["netG_A2B_state_dict"])
                netG_B2A.load_state_dict(ckp["netG_A2B_state_dict"])
                netD_A.load_state_dict(ckp["netG_A2B_state_dict"])
                netD_B.load_state_dict(ckp["netG_A2B_state_dict"])
                epoch = ckp["epoch"] + 1

            else:
                print(f"Initialize Model Weight...")
                netG_A2B.apply(self.init_weight)
                netG_B2A.apply(self.init_weight)
                netD_A.apply(self.init_weight)
                netD_B.apply(self.init_weight)
                epoch = 0

            return netG_A2B, netG_B2A, netD_A, netD_B, epoch

    def save_ckp(self, netG_A2B, netG_B2A, netD_A, netD_B, epoch):
        torch.save(
            {
                "netG_A2B_state_dict": netG_A2B.state_dict(),
                "netG_B2A_state_dict": netG_B2A.state_dict(),
                "netD_A2_state_dict": netD_A.state_dict(),
                "netD_B": netD_B.state_dict(),
                "epoch": epoch,
            },
            os.path.join(f"{self.cfg['output_ckp']}", f"{epoch}.pth")
        )


class TransformManager:
    def __init__(self, cfg):
        super(TransformManager, self).__init__()
        self.cfg = cfg

    def set_train_transform(self, to_image=False):
        assert type(to_image) is bool, 'Only boolean type is available for self.AUG.'

        if to_image:
            transform = transforms.Compose([
                transforms.Normalize(mean=(-1, -1, -1), std=(2, 2, 2)),
                transforms.ToPILImage(),
            ])
        else:
            transform = transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.RandomCrop((self.cfg['input_size'], self.cfg['input_size'])),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ])

        return transform