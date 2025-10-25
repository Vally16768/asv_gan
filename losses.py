# losses.py
from __future__ import annotations
import torch

def wgan_g_loss(d_fake: torch.Tensor) -> torch.Tensor:
    return -d_fake.mean()

def wgan_d_loss(d_real: torch.Tensor, d_fake: torch.Tensor) -> torch.Tensor:
    return (d_fake.mean() - d_real.mean())

def r1_regularizer(d_real: torch.Tensor, x_real: torch.Tensor) -> torch.Tensor:
    """
    R1 penalty: ||∇_x D(x)||^2 pe real; d_real: [B], x_real: [B,C,T]
    """
    grads = torch.autograd.grad(
        outputs=d_real.sum(),
        inputs=x_real,
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]
    return grads.pow(2).sum(dim=[1, 2]).mean()
