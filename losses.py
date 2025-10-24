# losses.py
import torch
import torch.nn.functional as F

def gradient_penalty(D, real, fake, device):
    alpha = torch.rand(real.size(0), 1, 1, 1, device=device)
    interp = (alpha * real.unsqueeze(1) + (1 - alpha) * fake.unsqueeze(1)).requires_grad_(True)
    # D expects [B, n_mels, T] wants unsqueeze inside
    interp = interp.squeeze(1)
    out = D(interp)
    grad = torch.autograd.grad(outputs=out, inputs=interp, grad_outputs=torch.ones_like(out).to(device),
                               create_graph=True, retain_graph=True, only_inputs=True)[0]
    grad = grad.view(grad.size(0), -1)
    gp = ((grad.norm(2, dim=1) - 1) ** 2).mean()
    return gp

def wgan_d_loss(real_scores, fake_scores):
    return torch.mean(fake_scores) - torch.mean(real_scores)

def wgan_g_loss(fake_scores):
    return -torch.mean(fake_scores)

# evasion loss: if detector outputs a scalar score (higher = spoof prob),
# we want to minimize probability of detector flagging as spoof.
# assume detector(wave) -> prob_spoof in [0,1] (if logits, adjust)
def evasion_loss_from_detector(detector_scores, target_label=0.0):
    # simplest: L2 to target_label (0 bona fide)
    return F.mse_loss(detector_scores, detector_scores.new_full(detector_scores.shape, target_label))

# spectral L1 loss on mel
def spec_loss(gen_mel, ref_mel):
    return F.l1_loss(gen_mel, ref_mel)

# speaker loss: requires speaker embedding model
def speaker_loss_fn(emb_gen, emb_ref):
    return F.mse_loss(emb_gen, emb_ref)
