import torch
from torch import nn
import torch.nn.functional as F
import math
import torchaudio

# ----------------- GAN losses (WGAN) -----------------
def d_loss_wgan(real_scores, fake_scores):
    return (fake_scores - real_scores).mean()

def g_loss_wgan(fake_scores):
    return (-fake_scores).mean()

def r1_penalty(real_wave, real_scores):
    grad = torch.autograd.grad(
        outputs=real_scores.sum(),
        inputs=real_wave,
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]
    penalty = grad.pow(2).reshape(grad.size(0), -1).sum(dim=1).mean()
    return penalty

# ----------------- Multi-Resolution STFT -----------------
class MRSTFTLoss(nn.Module):
    def __init__(self, fft_sizes=(512, 1024, 2048), hops=(160, 320, 640), wins=(400, 800, 1200)):
        super().__init__()
        self.fft_sizes = fft_sizes
        self.hops = hops
        self.wins = wins
        self.window_cache = {}

    def stft_mag(self, x, n_fft, hop, win):
        key = (n_fft,)
        if key not in self.window_cache:
            self.window_cache[key] = torch.hann_window(win).to(x.device)
        w = self.window_cache[key]
        X = torch.stft(x, n_fft=n_fft, hop_length=hop, win_length=win, window=w, center=True, return_complex=True)
        mag = torch.abs(X)
        return mag

    def forward(self, x, y):
        loss = 0.0
        for n_fft, hop, win in zip(self.fft_sizes, self.hops, self.wins):
            X = self.stft_mag(x, n_fft, hop, win)
            Y = self.stft_mag(y, n_fft, hop, win)
            loss += F.l1_loss(torch.log(X + 1e-6), torch.log(Y + 1e-6)) + F.l1_loss(X, Y)
        return loss / len(self.fft_sizes)

# ----------------- Feature Matching -----------------
def feature_matching_loss(real_feats, fake_feats):
    tot = 0.0
    count = 0
    for rf_s, ff_s in zip(real_feats, fake_feats):
        for r, f in zip(rf_s, ff_s):
            r = r.detach()  # asigură 0-grad pe ramura real (evită orice retenție accidentală)
            tot += F.l1_loss(r, f)
            count += 1
    return tot / max(1, count)

# ----------------- Evasion Loss (with surrogate) -----------------
def evasion_loss_from_logits(bona_logits, weight=1.0):
    target = torch.ones_like(bona_logits)
    loss = F.binary_cross_entropy_with_logits(bona_logits, target)
    return weight * loss
