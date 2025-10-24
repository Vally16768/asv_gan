# models.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel=3, stride=1, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel, stride, padding)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.LeakyReLU(0.2)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

class Down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            ConvBlock(in_ch, out_ch, kernel=4, stride=2, padding=1),
            ConvBlock(out_ch, out_ch)
        )
    def forward(self, x): return self.conv(x)

class Up(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1)
        self.conv = nn.Sequential(ConvBlock(in_ch, out_ch), ConvBlock(out_ch, out_ch))
    def forward(self, x, skip):
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)

class UNetMel(nn.Module):
    """
    Input: mel [B, n_mels, T] -> we unsqueeze channel dim
    Output: delta_mel [B, n_mels, T] (same shape)
    """
    def __init__(self, n_mels=80, base_ch=32):
        super().__init__()
        self.inc = nn.Sequential(ConvBlock(1, base_ch), ConvBlock(base_ch, base_ch))
        self.down1 = Down(base_ch, base_ch*2)
        self.down2 = Down(base_ch*2, base_ch*4)
        self.down3 = Down(base_ch*4, base_ch*8)
        self.bottleneck = nn.Sequential(ConvBlock(base_ch*8, base_ch*16), nn.Dropout(0.2))
        self.up3 = Up(base_ch*16, base_ch*8)
        self.up2 = Up(base_ch*8, base_ch*4)
        self.up1 = Up(base_ch*4, base_ch*2)
        self.outc = nn.Sequential(nn.Conv2d(base_ch*2, 1, kernel_size=1), nn.Tanh())

    def forward(self, mel):
        # mel: [B, n_mels, T]
        x = mel.unsqueeze(1)  # [B,1,n_mels,T]
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        b = self.bottleneck(x4)
        u3 = self.up3(b, x4)
        u2 = self.up2(u3, x3)
        u1 = self.up1(u2, x2)
        out = self.outc(u1)  # [B,1,n_mels,T]
        return out.squeeze(1)  # [B,n_mels,T]

class WDiscriminator(nn.Module):
    def __init__(self, n_mels=80, base_ch=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, base_ch, 4, 2, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(base_ch, base_ch*2, 4, 2, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(base_ch*2, base_ch*4, 4, 2, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(base_ch*4, 1, 3, 1, 1)
        )
    def forward(self, mel):
        # mel [B, n_mels, T] -> unsqueeze
        return self.net(mel.unsqueeze(1)).mean(dim=[1,2,3])  # scalar per batch element
