# utils.py
from __future__ import annotations
import random
import numpy as np
import torch

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # reproducibilitate mai strictă (ușor mai lent)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class EMA:
    def __init__(self, module, decay: float = 0.999):
        import copy
        self.decay = decay
        self.shadow = copy.deepcopy(module).eval()
        for p in self.shadow.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, module):
        for s, p in zip(self.shadow.parameters(), module.parameters()):
            s.data.lerp_(p.data, 1.0 - self.decay)
