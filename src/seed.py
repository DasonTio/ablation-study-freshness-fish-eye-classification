"""Seed control for reproducible multi-seed experiments."""
import random

import numpy as np
import torch


def set_seed(seed: int):
    """Seed python, numpy, and torch. Used to vary full pipeline (split + init + aug)
    across runs so we can report mean +/- std error bars."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
