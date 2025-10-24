# losses.py
import torch
import torch.nn.functional as F

def gradient_penalty(D, real, fake, device):
    # real/fake: [B, n_mels, T]
    alpha = torch.rand(real.size(0), 1, 1, device=device)
    interp = (alpha * real + (1 - alpha) * fake).requires_grad_(True)
    out = D(interp)
    grad = torch.autograd.grad(out, interp, grad_outputs=torch.ones_like(out),
                               create_graph=True, retain_graph=True, only_inputs=True)[0]
    grad = grad.view(grad.size(0), -1)
    return ((grad.norm(2, dim=1) - 1)**2).mean()

def wgan_d_loss(real_scores, fake_scores):
    return fake_scores.mean() - real_scores.mean()

def wgan_g_loss(fake_scores):
    return -fake_scores.mean()

def evasion_loss_ensemble(score_list, target=0.0, reduce="mean"):
    if not score_list: 
        return torch.tensor(0.0, device="cuda" if torch.cuda.is_available() else "cpu")
    losses = []
    for s in score_list:
        if s.dim() > 1: s = s.squeeze()
        losses.append(F.mse_loss(s, s.new_full(s.shape, float(target))))
    if reduce == "sum": return torch.stack(losses).sum()
    return torch.stack(losses).mean()

def spec_l1(gen_mel, ref_mel):
    return F.l1_loss(gen_mel, ref_mel)

# speaker loss = MSE între embeddings
def speaker_loss(emb_gen, emb_ref):
    return F.mse_loss(emb_gen, emb_ref)
