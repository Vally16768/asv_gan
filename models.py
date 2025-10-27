# models.py — ResUNet-1D generator + self-attention + 3-scale critic + surrogate detector
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm as SN

# ----------------- Small helpers -----------------
def conv1x1(in_ch, out_ch):
    return nn.Conv1d(in_ch, out_ch, kernel_size=1)

def norm_act(ch):
    return nn.Sequential(nn.GroupNorm(8, ch), nn.SiLU())

class ResBlock1D(nn.Module):
    def __init__(self, ch, k=3, dilation=1, dropout=0.0):
        super().__init__()
        p = (k - 1) // 2 * dilation
        self.block = nn.Sequential(
            norm_act(ch),
            nn.Conv1d(ch, ch, kernel_size=k, padding=p, dilation=dilation),
            norm_act(ch),
            nn.Conv1d(ch, ch, kernel_size=k, padding=p, dilation=dilation),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
        )

    def forward(self, x):
        return x + self.block(x)

class UpBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.ups = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="linear", align_corners=False),
            nn.Conv1d(in_ch, out_ch, kernel_size=3, padding=1),
        )
        self.res = ResBlock1D(out_ch)

    def forward(self, x):
        x = self.ups(x)
        return self.res(x)

class DownBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.down = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size=4, stride=2, padding=1),
            nn.SiLU(),
        )
        self.res = ResBlock1D(out_ch)

    def forward(self, x):
        x = self.down(x)
        return self.res(x)

# ----------------- Self-Attention 1D -----------------
class SelfAttention1D(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.query = nn.Conv1d(in_dim, max(1, in_dim // 8), 1)
        self.key   = nn.Conv1d(in_dim, max(1, in_dim // 8), 1)
        self.value = nn.Conv1d(in_dim, in_dim, 1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        # x: [B, C, T]
        B, C, T = x.shape
        q = self.query(x).view(B, -1, T)       # [B, Cq, T]
        k = self.key(x).view(B, -1, T)
        v = self.value(x).view(B, -1, T)
        attn = torch.bmm(q.transpose(1,2), k)  # [B, T, T]
        attn = torch.softmax(attn / (q.size(1) ** 0.5), dim=-1)
        out = torch.bmm(v, attn.transpose(1,2)).view(B, C, T)
        return self.gamma * out + x

# ----------------- Generator (ResUNet-1D) -----------------
class Generator(nn.Module):
    """
    Input: waveform [B, T]
    Output: waveform [B, T] (residual enhancement-style: out = in + 0.5*delta)
    """
    def __init__(self, base=64, depth=4):
        super().__init__()
        self.inp = nn.Conv1d(1, base, 7, padding=3)

        downs, ups = [], []
        ch = base
        self.skip_channels = []
        # encoder
        for _ in range(depth):
            downs.append(DownBlock(ch, ch * 2))
            ch *= 2
            self.skip_channels.append(ch)
        self.downs = nn.ModuleList(downs)

        # bottleneck
        self.bot = nn.Sequential(
            ResBlock1D(ch, k=3, dilation=1),
            ResBlock1D(ch, k=3, dilation=3),
            SelfAttention1D(ch),
            ResBlock1D(ch, k=3, dilation=9),
        )

        # decoder: we will concat skip features
        for _ in range(depth):
            # after concat channels double -> ups expect in_ch = current_ch * 2
            ups.append(UpBlock(in_ch=ch * 2, out_ch=ch // 2))
            ch //= 2
        self.ups = nn.ModuleList(ups)

        self.outp = nn.Sequential(
            norm_act(ch),
            nn.Conv1d(ch, 1, 7, padding=3),
            nn.Tanh()
        )

    def forward(self, wav):
        x = wav.unsqueeze(1)          # [B,1,T]
        x = self.inp(x)               # [B,C,T]
        feats = []
        for d in self.downs:
            x = d(x)
            feats.append(x)
        x = self.bot(x)

        # go back up in the same order as we created 'ups'
        for u in self.ups:
            skip = feats.pop()  # last encoder feature (matches current 'x' scale)
            # align lengths (safety for odd lengths)
            x = F.interpolate(x, size=skip.size(-1), mode='nearest')
            x = torch.cat([x, skip], dim=1)  # channels double here
            x = u(x)

        delta = self.outp(x).squeeze(1)
        return (wav + 0.5 * delta).clamp(-1, 1)

# ----------------- Multi-Scale Discriminator (3-scale) -----------------
class Critic1D(nn.Module):
    def __init__(self, in_ch=1, base=64):
        super().__init__()
        ch = base
        self.net = nn.ModuleList([
            SN(nn.Conv1d(in_ch, ch, 15, stride=1, padding=7)),
            nn.LeakyReLU(0.2, inplace=True),

            SN(nn.Conv1d(ch, ch*2, 15, stride=2, padding=7)),
            nn.LeakyReLU(0.2, inplace=True),

            SN(nn.Conv1d(ch*2, ch*4, 15, stride=2, padding=7)),
            nn.LeakyReLU(0.2, inplace=True),

            SN(nn.Conv1d(ch*4, ch*4, 15, stride=2, padding=7)),
            nn.LeakyReLU(0.2, inplace=True),
        ])
        self.head = SN(nn.Conv1d(ch*4, 1, 3, padding=1))

    def forward(self, x):
        # x: [B,1,T]
        feats = []
        h = x
        for layer in self.net:
            h = layer(h)
            feats.append(h)
        out = self.head(h)  # [B,1,T']
        score = out.mean(dim=[1, 2])  # WGAN scalar
        return score, feats

class MultiScaleCritic(nn.Module):
    def __init__(self):
        super().__init__()
        # stronger critics (base increased)
        self.d1 = Critic1D(base=64)
        self.d2 = Critic1D(base=64)
        self.d3 = Critic1D(base=64)
        self.avgpool = nn.AvgPool1d(4, 2, 1)

    def forward(self, x):
        s1 = x
        s2 = self.avgpool(x)
        s3 = self.avgpool(self.avgpool(x))
        s = []
        f = []
        for d, inp in [(self.d1, s1), (self.d2, s2), (self.d3, s3)]:
            sc, feats = d(inp)
            s.append(sc)
            f.append(feats)
        score = torch.stack(s, dim=0).mean(dim=0)
        return score, f

# ----------------- Optional Surrogate Detector -----------------
class SurrogateDetector(nn.Module):
    def __init__(self, mel_bins=160, hidden=2048):
        super().__init__()
        in_dim = mel_bins * 2  # mean + std pooling
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, 1),  # logits for bona_fide
        )

    def forward(self, mel):  # mel: [B, M, Tm]
        mu = mel.mean(dim=-1)
        sd = mel.std(dim=-1)
        x = torch.cat([mu, sd], dim=1)  # [B, 2M]
        return self.net(x).squeeze(-1)  # [B]
