import torch
from torch import nn
import torch.nn.functional as F

# ----------------- GAN losses (WGAN) -----------------
def d_loss_wgan(real_scores, fake_scores):
    # maximize real - fake -> minimize fake - real
    return (fake_scores - real_scores).mean()

def g_loss_wgan(fake_scores):
    return (-fake_scores).mean()

# R1 penalty on real
def r1_penalty(real_wave, real_scores):
    # real_wave: [B,1,T] requires grad, real_scores: [B]
    grad = torch.autograd.grad(
        outputs=real_scores.sum(),
        inputs=real_wave,
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]  # [B,1,T]
    penalty = grad.pow(2).reshape(grad.size(0), -1).sum(dim=1).mean()
    return penalty

# ----------------- Multi-Resolution STFT -----------------
class MRSTFTLoss(nn.Module):
    def __init__(self, fft_sizes=(512, 1024, 2048), hops=(160, 320, 640), wins=(400, 800, 1200)):
        super().__init__()
        self.fft_sizes = fft_sizes
        self.hops = hops
        self.wins = wins
        self._windows = {}  # cache per n_fft

    def _stft_mag(self, x, n_fft, hop, win):
        if n_fft not in self._windows:
            self._windows[n_fft] = torch.hann_window(win, periodic=True, dtype=x.dtype, device=x.device)
        w = self._windows[n_fft]
        X = torch.stft(x, n_fft=n_fft, hop_length=hop, win_length=win, window=w,
                       center=True, return_complex=True)
        return torch.abs(X)

    def forward(self, x, y):  # wave [B,T]
        loss = 0.0
        for n_fft, hop, win in zip(self.fft_sizes, self.hops, self.wins):
            X = self._stft_mag(x, n_fft, hop, win)
            Y = self._stft_mag(y, n_fft, hop, win)
            loss += F.l1_loss(torch.log(X + 1e-6), torch.log(Y + 1e-6)) + F.l1_loss(X, Y)
        return loss / len(self.fft_sizes)

# ----------------- Feature Matching -----------------
def feature_matching_loss(real_feats, fake_feats):
    # real_feats / fake_feats: list[list[tensors]] per scale
    tot = 0.0
    cnt = 0
    for rf_s, ff_s in zip(real_feats, fake_feats):
        for r, f in zip(rf_s, ff_s):
            tot += F.l1_loss(r, f)
            cnt += 1
    return tot / max(1, cnt)

# ----------------- Evasion Loss (with surrogate) -----------------
def evasion_loss_from_logits(bona_logits, weight=1.0):
    # vrem bona_fide => target=1
    target = torch.ones_like(bona_logits)
    loss = F.binary_cross_entropy_with_logits(bona_logits, target)
    return weight * loss
