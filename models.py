# models.py
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------
# Utils: pad/unpad la multiplu de 8 + aliniere skip
# ---------------------------------------------------------

def _pad_to_multiple(x: torch.Tensor, mult: int = 8):
    """
    x: [B, C, H, W]
    întoarce: x_pad, (pad_h, pad_w) cu pad adăugat la dreapta/jos
    """
    B, C, H, W = x.shape
    Ht = ((H + mult - 1) // mult) * mult
    Wt = ((W + mult - 1) // mult) * mult
    pad_h = Ht - H
    pad_w = Wt - W
    if pad_h or pad_w:
        x = F.pad(x, (0, pad_w, 0, pad_h))  # (left,right,top,bottom) dar 2D: (wL,wR,hT,hB)
    return x, (pad_h, pad_w)

def _unpad(x: torch.Tensor, pads: tuple[int,int]):
    """
    inversează _pad_to_multiple; x: [B, C, H, W]
    """
    pad_h, pad_w = pads
    if pad_h or pad_w:
        H = x.shape[2] - pad_h
        W = x.shape[3] - pad_w
        x = x[:, :, :H, :W]
    return x

def _match_spatial(x_up: torch.Tensor, x_skip: torch.Tensor):
    """
    Aduce x_up la aceeași dimensiune spațială ca x_skip.
    """
    _, _, Hs, Ws = x_skip.shape
    _, _, Hu, Wu = x_up.shape
    if (Hu, Wu) != (Hs, Ws):
        x_up = F.interpolate(x_up, size=(Hs, Ws), mode="bilinear", align_corners=False)
    return x_up

# ---------------------------------------------------------
# Blocuri U-Net 2D
# ---------------------------------------------------------

class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x):
        return self.net(x)

class Down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x):
        return self.conv(self.pool(x))

class Up(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, in_ch // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x, skip):
        x = self.up(x)
        x = _match_spatial(x, skip)
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)

# ---------------------------------------------------------
# Generator: UNetMel
# Input mel: [B, n_mels, T]  -> reshape la [B, 1, n_mels, T]
# Output delta: [B, n_mels, T] (aceeași dimensiune ca inputul)
# ---------------------------------------------------------

class UNetMel(nn.Module):
    def __init__(self, n_mels: int, base_ch: int = 64):
        super().__init__()
        self.n_mels = n_mels

        self.inc = DoubleConv(1, base_ch)
        self.down1 = Down(base_ch, base_ch * 2)    # 64 -> 128
        self.down2 = Down(base_ch * 2, base_ch * 4) # 128 -> 256
        self.down3 = Down(base_ch * 4, base_ch * 8) # 256 -> 512

        self.up1 = Up(base_ch * 8, base_ch * 4)    # 512 -> 256
        self.up2 = Up(base_ch * 4, base_ch * 2)    # 256 -> 128
        self.up3 = Up(base_ch * 2, base_ch)        # 128 -> 64

        self.outc = nn.Conv2d(base_ch, 1, kernel_size=1)

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        """
        mel: [B, n_mels, T]
        return delta: [B, n_mels, T]
        """
        B, M, Tm = mel.shape
        assert M == self.n_mels, f"UNetMel expected n_mels={self.n_mels}, got {M}"

        x = mel.unsqueeze(1)  # [B, 1, n_mels, T]
        x, pads = _pad_to_multiple(x, mult=8)  # asigură divizibilitate pentru 3 downs (x/8)

        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)

        u1 = self.up1(x4, x3)
        u2 = self.up2(u1, x2)
        u3 = self.up3(u2, x1)

        out = self.outc(u3)               # [B, 1, H, W]
        out = _unpad(out, pads)           # revino la [B, 1, n_mels, T]
        out = out.squeeze(1)              # [B, n_mels, T]

        # clamp mic pentru stabilitate numerică (opțional)
        return torch.tanh(out)

# ---------------------------------------------------------
# Discriminator: WGAN critic pe mel ca imagine
# Primește [B, n_mels, T] -> mapă scor, o agregă în [B]
# ---------------------------------------------------------

class WDiscriminator(nn.Module):
    def __init__(self, n_mels: int, base_ch: int = 64):
        super().__init__()
        self.n_mels = n_mels
        # lucrăm pe canal unic (imagine mel)
        def block(in_ch, out_ch, k=4, s=2, p=1, bn=True):
            layers = [nn.Conv2d(in_ch, out_ch, kernel_size=k, stride=s, padding=p)]
            if bn:
                layers.append(nn.InstanceNorm2d(out_ch, affine=True))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return nn.Sequential(*layers)

        self.net = nn.Sequential(
            block(1, base_ch, bn=False),         # 1 -> 64
            block(base_ch, base_ch * 2),         # 64 -> 128
            block(base_ch * 2, base_ch * 4),     # 128 -> 256
            block(base_ch * 4, base_ch * 8),     # 256 -> 512
            nn.Conv2d(base_ch * 8, 1, kernel_size=3, padding=1)  # [B,1,H',W']
        )

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        """
        mel: [B, n_mels, T] -> scor [B]
        """
        x = mel.unsqueeze(1)  # [B,1,n_mels,T]
        feat = self.net(x)    # [B,1,H',W']
        # WGAN critic: agregare medie spațială
        score = feat.mean(dim=[1, 2, 3])  # [B]
        return score
