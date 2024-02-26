import os
import re
import tqdm
import yaml

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader

from run.loss import calc_gradient_penalty, fourier_transform_loss

from model.plga_vit_generator import Gen
from model.discriminator import Disc

from utils.dataset import CustomDataset
from utils.func import ImagePool, CkpManager, TransformManager
from utils.logging import LossDisplayer, save_model, sampling, save_yaml


class Trainer(object):
    def __init__(self, config):
        super(Trainer, self).__init__()
        self.cfg = config
        self.device = torch.device(f"{self.cfg['device']}")

        os.makedirs(f"{self.cfg['output_ckp']}", exist_ok=True)
        os.makedirs(f"{self.cfg['output_sample']}", exist_ok=True)
        os.makedirs(f"{self.cfg['output_log']}", exist_ok=True)

        self.ckp_manager = CkpManager(self.cfg)
        _transform_manager = TransformManager(self.cfg)

        self.transform = _transform_manager.set_train_transform(to_image=False)
        self.transform_to_image = _transform_manager.set_train_transform(to_image=True)

        self.pool_fake_A = ImagePool(self.cfg['pool_size'])
        self.pool_fake_B = ImagePool(self.cfg['pool_size'])

        # Build Model
        self.netG_A2B = Gen(self.cfg).to(self.device)
        self.netG_B2A = Gen(self.cfg).to(self.device)

        self.netD_A = Disc().to(self.device)
        self.netD_B = Disc().to(self.device)

        # Loss setting
        self.criterion_cycle = nn.L1Loss()
        self.criterion_identity = nn.L1Loss()
        self.criterion_gan = nn.MSELoss()

        if self.cfg['do_high_frequency_loss']:
            self.criterion_hf = nn.L1Loss()

        # Optimizer & scheduler setting
        self.optim_G = optim.Adam(
            list(self.netG_A2B.parameters()) + list(self.netG_B2A.parameters()),
            lr=float(self.cfg['lr']),
            betas=(0.5, 0.999)
        )
        self.optim_D_A = optim.Adam(self.netD_A.parameters(), lr=float(self.cfg['lr']))
        self.optim_D_B = optim.Adam(self.netD_B.parameters(), lr=float(self.cfg['lr']))

        self.scheduler_G = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer=self.optim_G, T_0=50, T_mult=2, eta_min=0.001)
        self.scheduler_D_A = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer=self.optim_D_A, T_0=50, T_mult=2, eta_min=0.001)
        self.scheduler_D_B = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer=self.optim_D_B, T_0=50, T_mult=2, eta_min=0.001)

        self.disp = LossDisplayer(self.cfg['display_loss_list'])

    def run(self):
        self.netG_A2B, self.netG_B2A, self.netD_A, self.netD_B, epoch = self.ckp_manager.load_ckp(self.netG_A2B, self.netG_B2A, self.netD_A, self.netD_B, imagenet=False)

        self.netG_A2B.train()
        self.netG_B2A.train()
        self.netD_A.train()
        self.netD_B.train()

        summary = SummaryWriter(f"{self.cfg['output_log']}/loss")
        dataset = CustomDataset(cfg=self.cfg, transform=self.transform)
        data_loader = DataLoader(dataset=dataset, batch_size=self.cfg['batch_size'], shuffle=True)

        for ep in range(epoch, self.cfg['full_epoch']):
            self.run_epoch(data_loader, ep)
            self. scheduler_G.step()
            self.scheduler_D_A.step()
            self.scheduler_D_B.step()

            avg_losses = self.disp.get_avg_losses()

            summary.add_scalar("Gen/loss_G_GAN", avg_losses[0], ep)
            summary.add_scalar("Gen/loss_G_cycle", avg_losses[1], ep)
            summary.add_scalar("Gen/loss_G", avg_losses[2], ep)
            # summary.add_scalar("Gen/loss_hf", avg_losses[3], epoch)
            # summary.add_scalar("Gen/loss_P", avg_losses[4], epoch)

            summary.add_scalar("Disc/loss_D_A", avg_losses[3], ep)
            summary.add_scalar("Disc/loss_D_B", avg_losses[4], ep)
            summary.add_scalar("Disc/loss_D", avg_losses[5], ep)

    def run_epoch(self, data_loader, epoch):
        for idx, (real_A, real_B, name) in enumerate(tqdm.tqdm(data_loader, desc=f"Epcoh : [{epoch}/{self.cfg['full_epoch']}]")):
            loss_list = []

            real_A = real_A.to(self.device)
            real_B = real_B.to(self.device)

            # Foard Model
            # A -> B
            fake_B = self.netG_A2B(real_A)
            # B -> A
            fake_A = self.netG_B2A(real_B)

            # A -> B -> A
            cycle_A = self.netG_B2A(fake_B)
            # B -> A -> B
            cycle_B = self.netG_A2B(fake_A)

            # Identity mapping
            same_A = self.netG_B2A(real_A)
            same_B = self.netG_A2B(real_B)

            disc_fake_A = self.netD_A(fake_A)
            disc_fake_B = self.netD_B(fake_B)

            # Calculate and backward generator
            loss_cycle_A = self.criterion_cycle(cycle_A, real_A)
            loss_cycle_B = self.criterion_cycle(cycle_B, real_B)

            loss_gan_A = self.criterion_gan(disc_fake_A, torch.ones_like(disc_fake_A))
            loss_gan_B = self.criterion_gan(disc_fake_B, torch.ones_like(disc_fake_B))

            loss_id_A = self.criterion_identity(same_A, real_A)
            loss_id_B = self.criterion_identity(same_B, real_B)

            loss_g = (
                self.cfg['lamda_cycle'] * (loss_cycle_A + loss_cycle_B)
                + (loss_gan_A + loss_gan_B)
                + self.cfg['lamda_id'] * self.cfg['lamda_cycle'] * (loss_id_A + loss_id_B)
            )

            loss_G_GAN = loss_gan_A + loss_gan_B
            loss_G_cycle = loss_g - loss_G_GAN
            loss_list += [loss_G_GAN, loss_G_cycle, loss_g]

            if self.cfg['do_fourier_transform_loss']:
                hf_loss = fourier_transform_loss(self.criterion_hf, self.cfg['fourier_transform_loss_weight'],
                                              real_A, fake_A, cycle_A, real_B, fake_B, cycle_B)
                loss_g = loss_g + hf_loss
                loss_list += hf_loss

            self.optim_G.zero_grad()
            loss_g.backward()
            self.optim_G.step()

            # Calculate and backward discriminator
            disc_real_A = self.netD_A(real_A)
            disc_fake_A = self.netD_A(self.pool_fake_A.query(fake_A))
            grad_penalty_A = calc_gradient_penalty(self.netD_A, fake_A, real_A)

            loss_D_A = 0.5 * (
                    self.criterion_gan(disc_fake_A, torch.zeros_like(disc_fake_A))
                    + self.criterion_gan(disc_real_A, torch.ones_like(disc_real_A))
                    + grad_penalty_A)

            self.optim_D_A.zero_grad()
            loss_D_A.backward()
            self.optim_D_A.step()

            disc_real_B = self.netD_B(real_B)
            disc_fake_B = self.netD_B(self.pool_fake_B.query(fake_B))
            grad_penalty_B = calc_gradient_penalty(self.netD_B, fake_B, real_B)

            loss_D_B = 0.5 * (
                    self.criterion_gan(disc_fake_B, torch.zeros_like(disc_fake_B))
                    + self.criterion_gan(disc_real_B, torch.ones_like(disc_real_B))
                    + grad_penalty_B)

            loss_D = loss_D_A + loss_D_B
            loss_list += [loss_D_A, loss_D_B, loss_D]

            self.optim_D_B.zero_grad()
            loss_D_B.backward()
            self.optim_D_B.step()

            self.disp.record(loss_list)
            if self.cfg['do_sampling']:
                if idx % 100 == 0:
                    name = name[0].split("\\")[-1]
                    name = re.compile(".png").sub('', name)
                    sampling(self.cfg['output_sample'], fake_B[0], self.transform_to_image, name, epoch, "A2B")
                    sampling(self.cfg['output_sample'], cycle_A[0], self.transform_to_image, name, epoch, 'A2B2A')