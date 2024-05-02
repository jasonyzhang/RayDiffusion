import os
import random

import ipdb  # noqa: F401
import numpy as np
import torch

os.umask(000)  # Default to 777 permissions


class Trainer(object):
    def __init__(self, cfg):
        seed = cfg.training.seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        self.cfg = cfg
        self.debug = cfg.debug
        self.resume = cfg.training.resume
        self.pretrain_path = cfg.training.pretrain_path
