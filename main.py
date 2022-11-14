import torch.distributed as dist
import torch.multiprocessing as mp

import os
import os.path as osp
import sys
import numpy as np
import copy

from lib.cfg_holder import cfg_unique_holder as cfguh
from lib.cfg_helper import \
    get_command_line_args, \
    cfg_initiates

from lib.model_zoo.sd import version
from lib.utils import get_obj_from_str

if __name__ == "__main__":
    cfg = get_command_line_args()
    cfg = cfg_initiates(cfg)

    evaler = get_obj_from_str(cfg.eval.main)(cfg)
    estage = get_obj_from_str(cfg.eval.stage)()
    evaler.register_stage(estage)
    evaler(0)
