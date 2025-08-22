import random
import numpy as np
import torch
import logging
from pathlib import Path

log = logging.getLogger(__name__)

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    log.info(f"Set random seed to {seed}")
