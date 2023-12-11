# create a data class that ingests a file with the following structure:
# <name>=<value> where <value> can either be a "string" or a number
# the class creates properties for each name and assigns the value to it

import re
from typing import Literal


class Settings:
    """
    This class loads a settings file and creates properties for each name in the file.
    The values are either strings or numbers.
    If the name starts with a star, the value is saved to self.cfg_dict to be used for wandb config.
    """

    project: str
    group: str
    name: str
    notes: str
    model_notes: str
    resume: str
    mode: str
    log_frequency: int
    log_dir: str

    # Data location
    model_root: str
    data_root: str
    data_train_he: str
    data_train_p63: str
    data_test_he: str
    data_test_p63: str
    size: int
    crop: int
    flip_vertical: bool
    flip_horizontal: bool
    norm_dict: dict
    channels: int
    pool_size: int

    # Model
    checkpoint_frequency_steps: int
    batch_size: int
    generator_downconv_filters: int
    discriminator_downconv_filters: int
    num_resnet_blocks: int
    lr_generator: float
    lr_discriminator: float
    epochs: int
    decay_epoch: int
    lambda_cycle: int
    lambda_identity: int
    lambda_adversarial: float
    lambda_mask_adversarial_ratio: float
    lambda_mask_cycle_ratio: float
    lambda_context: int
    lambda_cycle_context: int
    mask_type: Literal['binary_rec', 'entropy', 'noise']
    explanation_ramp_type: str
    beta1: float
    beta2: float

    def __init__(self, path):
        self.path = path
        self.cfg_dict = {}
        self.load_settings()

    def load_settings(self):
        with open(self.path, 'r') as file:
            for line in file:
                # if line is empty or starts with a comment, skip it
                if not line.strip() or line.strip().startswith('#'):
                    continue

                if '=' in line:
                    name, value = line.split('=')
                    name = name.strip()
                    value = value.strip()

                    if re.match(r'^-?\d+$', value):
                        value = int(value)
                    elif re.match(r'^-?\d+\.\d+$', value):
                        value = float(value)
                    elif value == 'None':
                        value = None
                    elif value == 'True':
                        value = True
                    elif value == 'False':
                        value = False
                    else:
                        value = value.strip('"').strip("'")

                    # check if name has a star on the beginning and if so, save it to self.cfg_dict
                    if name.startswith('*'):
                        name = name[1:]
                        self.cfg_dict[name] = value

                    if name not in Settings.__dict__['__annotations__'].keys():
                        raise KeyError(f"Unknown setting: {name}")

                    setattr(self, name, value)
