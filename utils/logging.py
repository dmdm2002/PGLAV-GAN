import yaml
import os

import torch


def save_yaml(path, yaml_name, config):
    with open(f"{path}/{yaml_name}.yaml") as f:
        yaml.dump(config, f)


def save_model(path, model_state_dict):
    torch.save(model_state_dict(), f'{path}/backup_model')


def sampling(path, output, transform_to_image, name, epoch, gen_type):
    output_path = f"{path}/{gen_type}/{epoch}"
    os.makedirs(output_path, exist_ok=True)
    output = transform_to_image(output.squeeze())
    output.save(f'{output_path}/{name}_{gen_type}.png')


class LossDisplayer:
    def __init__(self, name_list):
        self.count = 0
        self.name_list = name_list
        self.loss_list = [0] * len(self.name_list)

    def record(self, losses):
        self.count += 1
        for i, loss in enumerate(losses):
            self.loss_list[i] += loss.item()

    def get_avg_losses(self):
        return [loss / self.count for loss in self.loss_list]

    def display(self):
        for i, total_loss in enumerate(self.loss_list):
            avg_loss = total_loss / self.count
            print(f"{self.name_list[i]}: {avg_loss:.4f}   ", end="")

    def reset(self):
        self.count = 0
        self.loss_list = [0] * len(self.name_list)

    def logging(self, display_loss_name_list, summary, ep):
        display_loss_value_list = self.get_avg_losses()
        for i in range(len(display_loss_name_list)):
            summary.add_scalar(display_loss_name_list[i], display_loss_value_list[i], ep)